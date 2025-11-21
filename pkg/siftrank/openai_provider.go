package siftrank

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
	"github.com/pkoukk/tiktoken-go"
)

// customTransport captures response headers and body for rate limit handling
type customTransport struct {
	Transport  http.RoundTripper
	Headers    http.Header
	StatusCode int
	Body       []byte
}

func (t *customTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.Transport.RoundTrip(req)
	if err != nil {
		return nil, err
	}

	t.Headers = resp.Header
	t.StatusCode = resp.StatusCode

	t.Body, err = io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	resp.Body = io.NopCloser(bytes.NewBuffer(t.Body))

	return resp, nil
}

// OpenAIProvider implements LLMProvider using OpenAI API
type OpenAIProvider struct {
	client    *openai.Client
	model     openai.ChatModel
	effort    string
	logger    *slog.Logger
	encoding  *tiktoken.Tiktoken
	transport *customTransport
}

// OpenAIConfig configures the OpenAI provider
type OpenAIConfig struct {
	APIKey   string
	Model    openai.ChatModel
	BaseURL  string // Optional: for vLLM, OpenRouter, etc.
	Encoding string // Tokenizer encoding
	Effort   string // Optional reasoning effort
	Logger   *slog.Logger
}

// NewOpenAIProvider creates a new OpenAI provider
func NewOpenAIProvider(cfg OpenAIConfig) (*OpenAIProvider, error) {
	// Create encoding
	encoding, err := tiktoken.GetEncoding(cfg.Encoding)
	if err != nil {
		return nil, fmt.Errorf("failed to get tiktoken encoding: %w", err)
	}

	// Create custom transport for rate limit handling
	transport := &customTransport{Transport: http.DefaultTransport}
	httpClient := &http.Client{Transport: transport}

	// Create OpenAI client
	clientOptions := []option.RequestOption{
		option.WithAPIKey(cfg.APIKey),
		option.WithHTTPClient(httpClient),
		option.WithMaxRetries(5),
	}

	if cfg.BaseURL != "" {
		baseURL := cfg.BaseURL
		if !strings.HasSuffix(baseURL, "/") {
			baseURL += "/"
		}
		clientOptions = append(clientOptions, option.WithBaseURL(baseURL))
	}

	client := openai.NewClient(clientOptions...)

	return &OpenAIProvider{
		client:    &client,
		model:     cfg.Model,
		effort:    cfg.Effort,
		logger:    cfg.Logger,
		encoding:  encoding,
		transport: transport,
	}, nil
}

// Complete implements LLMProvider.Complete
// Handles network-level retries only. Returns raw response without validation.
func (p *OpenAIProvider) Complete(ctx context.Context, prompt string, opts *CompletionOptions) (string, error) {
	backoff := time.Second
	maxBackoff := 30 * time.Second

	// Create default options if nil
	if opts == nil {
		opts = &CompletionOptions{}
	}

	var totalUsage Usage

	for {
		// Check if context cancelled
		if ctx.Err() != nil {
			return "", ctx.Err()
		}

		// Create timeout context for this attempt
		timeoutCtx, cancel := context.WithTimeout(ctx, 15*time.Second)

		// Build request
		params := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(prompt),
			},
			Model: p.model,
		}

		// Add structured output if schema provided
		if opts.Schema != nil {
			params.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
					JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
						Name:        "structured_response",
						Description: openai.String("Structured JSON response"),
						Schema:      opts.Schema,
						Strict:      openai.Bool(true),
					},
				},
			}
		}

		// Add temperature if provided
		if opts.Temperature != nil {
			params.Temperature = openai.Float(*opts.Temperature)
		}

		// Add max tokens if provided
		if opts.MaxTokens != nil {
			params.MaxTokens = openai.Int(int64(*opts.MaxTokens))
		}

		if p.effort != "" {
			params.ReasoningEffort = shared.ReasoningEffort(p.effort)
		}

		// Make API call
		completion, err := p.client.Chat.Completions.New(timeoutCtx, params)
		cancel() // Cancel immediately after API call to avoid resource leak

		if err == nil {
			// Success! Populate usage and metadata
			callUsage := Usage{
				InputTokens:  int(completion.Usage.PromptTokens),
				OutputTokens: int(completion.Usage.CompletionTokens),
			}

			// Check for reasoning tokens (o1/o3 models)
			// CompletionTokensDetails is a value type with zero defaults if not present
			if completion.Usage.JSON.CompletionTokensDetails.Valid() &&
				completion.Usage.CompletionTokensDetails.ReasoningTokens > 0 {
				callUsage.ReasoningTokens = int(completion.Usage.CompletionTokensDetails.ReasoningTokens)
			}

			callUsage.TotalTokens = callUsage.InputTokens + callUsage.OutputTokens + callUsage.ReasoningTokens
			totalUsage.Add(callUsage)

			// Populate output fields in opts
			opts.Usage = totalUsage
			opts.ModelUsed = string(completion.Model)
			if len(completion.Choices) > 0 {
				opts.FinishReason = string(completion.Choices[0].FinishReason)
			}
			opts.RequestID = completion.ID

			content := completion.Choices[0].Message.Content

			p.logger.Debug("OpenAI call successful",
				"input_tokens", callUsage.InputTokens,
				"output_tokens", callUsage.OutputTokens,
				"reasoning_tokens", callUsage.ReasoningTokens,
				"model", opts.ModelUsed)

			// Return raw content - no validation
			return content, nil
		}

		// Check if context cancelled
		if ctx.Err() != nil {
			return "", ctx.Err()
		}

		// Handle timeout
		if err == context.DeadlineExceeded {
			p.logger.Debug("Request timeout, retrying", "backoff", backoff)
			time.Sleep(backoff)
			backoff = minDuration(backoff*2, maxBackoff)
			continue
		}

		// Handle rate limits (429)
		if p.transport.StatusCode == http.StatusTooManyRequests {
			p.handleRateLimit(&backoff, maxBackoff)
			continue
		}

		// Handle server errors (5xx) - retry
		if p.transport.StatusCode >= 500 && p.transport.StatusCode < 600 {
			p.logger.Debug("Server error, retrying",
				"status", p.transport.StatusCode,
				"backoff", backoff)
			time.Sleep(backoff)
			backoff = minDuration(backoff*2, maxBackoff)
			continue
		}

		// Client errors (4xx except 429) are unrecoverable
		if p.transport.StatusCode >= 400 && p.transport.StatusCode < 500 {
			p.logger.Error("Unrecoverable client error",
				"status", p.transport.StatusCode,
				"error", err)
			return "", fmt.Errorf("unrecoverable error (status %d): %w",
				p.transport.StatusCode, err)
		}

		// Other errors - retry with backoff
		p.logger.Debug("Request failed, retrying", "error", err, "backoff", backoff)
		time.Sleep(backoff)
		backoff = minDuration(backoff*2, maxBackoff)
	}
}

// handleRateLimit handles rate limit errors with intelligent backoff
func (p *OpenAIProvider) handleRateLimit(backoff *time.Duration, maxBackoff time.Duration) {
	// Log rate limit headers
	for key, values := range p.transport.Headers {
		if strings.HasPrefix(key, "X-Ratelimit") || strings.HasPrefix(key, "X-RateLimit") {
			for _, value := range values {
				p.logger.Debug("Rate limit header", "key", key, "value", value)
			}
		}
	}

	if p.transport.Body != nil {
		p.logger.Debug("Rate limit response body", "body", string(p.transport.Body))
	}

	// Extract suggested wait time
	resetTokensStr := p.transport.Headers.Get("X-Ratelimit-Reset-Tokens")
	if resetTokensStr == "" {
		resetTokensStr = p.transport.Headers.Get("X-RateLimit-Reset-Tokens")
	}

	remainingTokensStr := p.transport.Headers.Get("X-Ratelimit-Remaining-Tokens")
	if remainingTokensStr == "" {
		remainingTokensStr = p.transport.Headers.Get("X-RateLimit-Remaining-Tokens")
	}

	remainingTokens, _ := strconv.Atoi(remainingTokensStr)
	resetDuration, _ := time.ParseDuration(resetTokensStr)

	p.logger.Debug("Rate limit exceeded",
		"remaining_tokens", remainingTokens,
		"reset_duration", resetDuration)

	// Use suggested wait time if available, otherwise exponential backoff
	if resetDuration > 0 {
		p.logger.Debug("Waiting for rate limit reset", "duration", resetDuration)
		time.Sleep(resetDuration)
	} else {
		p.logger.Debug("Waiting with exponential backoff", "duration", *backoff)
		time.Sleep(*backoff)
		*backoff = minDuration(*backoff*2, maxBackoff)
	}
}

// EstimateTokens implements LLMProvider.EstimateTokens
func (p *OpenAIProvider) EstimateTokens(text string) int {
	return len(p.encoding.Encode(text, nil, nil))
}

// minDuration returns the minimum of two durations
func minDuration(a, b time.Duration) time.Duration {
	if a < b {
		return a
	}
	return b
}
