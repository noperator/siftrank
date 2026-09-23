package siftrank

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

func TestOpenAIProviderConcurrentResponseState(t *testing.T) {
	arrived := make(chan struct{}, 2)
	release := make(chan struct{})
	var rateRequests, badRequests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if len(request.Messages) != 1 {
			t.Errorf("got %d request messages", len(request.Messages))
			return
		}

		w.Header().Set("Content-Type", "application/json")
		switch request.Messages[0].Content {
		case "rate-limit":
			if rateRequests.Add(1) == 1 {
				arrived <- struct{}{}
				<-release
				w.Header().Set("X-RateLimit-Reset-Tokens", "1ms")
				w.WriteHeader(http.StatusTooManyRequests)
				_, _ = io.WriteString(w, `{"error":{"message":"rate limited","type":"rate_limit_error"}}`)
				return
			}
		case "bad-request":
			if badRequests.Add(1) == 1 {
				arrived <- struct{}{}
				<-release
				w.WriteHeader(http.StatusBadRequest)
				_, _ = io.WriteString(w, `{"error":{"message":"invalid request","type":"invalid_request_error"}}`)
				return
			}
		default:
			t.Errorf("unexpected prompt %q", request.Messages[0].Content)
			return
		}
		_, _ = io.WriteString(w, `{"id":"chatcmpl-test","object":"chat.completion","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer server.Close()

	transport := &customTransport{Transport: http.DefaultTransport}
	httpClient := &http.Client{Transport: transport}
	client := openai.NewClient(
		option.WithAPIKey("test-key"),
		option.WithHTTPClient(httpClient),
		option.WithBaseURL(server.URL+"/v1/"),
		option.WithMaxRetries(0),
	)
	provider := &OpenAIProvider{
		client: &client,
		model:  openai.ChatModelGPT4oMini,
		logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	}

	rateDone := make(chan error, 1)
	badDone := make(chan error, 1)
	go func() {
		result, err := provider.Complete(context.Background(), "rate-limit", &CompletionOptions{})
		if err == nil && result != "ok" {
			err = fmt.Errorf("unexpected rate-limited result %q", result)
		}
		rateDone <- err
	}()
	go func() {
		_, err := provider.Complete(context.Background(), "bad-request", &CompletionOptions{})
		badDone <- err
	}()

	<-arrived
	<-arrived
	close(release)

	if err := <-rateDone; err != nil {
		t.Errorf("rate-limited request: %v", err)
	}
	if err := <-badDone; err == nil || !strings.Contains(err.Error(), "unrecoverable error (status 400)") {
		t.Errorf("bad request returned %v, want an unrecoverable status 400 error", err)
	}
	if got := rateRequests.Load(); got != 2 {
		t.Errorf("rate-limited request made %d calls, want one retry", got)
	}
	if got := badRequests.Load(); got != 1 {
		t.Errorf("bad request made %d calls, want no retry", got)
	}
}
