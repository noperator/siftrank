package siftrank

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"sync"
	"testing"
	"time"

	"github.com/openai/openai-go"
)

func TestRankDocsConcurrentIDs(t *testing.T) {
	const workers, iterations = 16, 10
	entered := make(chan struct{}, workers*iterations)
	release := make(chan struct{})
	idsPattern := regexp.MustCompile("id: `([^`]+)`")
	p := &legacyTestProvider{complete: func(ctx context.Context, prompt string, opts *CompletionOptions) (string, error) {
		entered <- struct{}{}
		select {
		case <-release:
		case <-ctx.Done():
			return "", ctx.Err()
		}
		var ids []string
		for _, match := range idsPattern.FindAllStringSubmatch(prompt, -1) {
			ids = append(ids, match[1])
		}
		data, err := json.Marshal(rankedDocumentResponseNoRelevance{Documents: ids})
		return string(data), err
	}}
	cfg := NewConfig()
	cfg.InitialPrompt = "Rank items"
	cfg.LLMProvider = p
	cfg.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	r, err := NewRanker(cfg)
	if err != nil {
		t.Fatal(err)
	}
	r.rng = rand.New(rand.NewSource(1))
	var docs []document
	for i := 0; i < 10; i++ {
		docs = append(docs, document{ID: fmt.Sprintf("original-%d", i), Value: fmt.Sprint(i)})
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	start := make(chan struct{})
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			for i := 0; i < iterations; i++ {
				got, calls, _, err := r.rankDocs(ctx, docs, 1, i+1)
				if err != nil || calls != 1 || len(got) != len(docs) {
					t.Errorf("rankDocs: %d items, %d calls, %v", len(got), calls, err)
					return
				}
				for j, ranked := range got {
					if ranked.Document != docs[j] || ranked.Score != float64(j+1) {
						t.Errorf("candidate %d changed: %+v", j, ranked)
					}
				}
			}
		}()
	}
	close(start)
	// All providers must be in flight together before any is released.
	for i := 0; i < workers; i++ {
		select {
		case <-entered:
		case <-ctx.Done():
			t.Errorf("provider calls did not overlap: %v", ctx.Err())
		}
	}
	close(release)
	wg.Wait()
}

func TestUsageAccounting(t *testing.T) {
	providerErr := errors.New("provider failed")
	for _, tc := range []struct {
		name            string
		outcomes        []string
		zeroUsage       bool
		batches, trials int
		wantErr         error
	}{
		{name: "completed", outcomes: []string{"success"}, batches: 1, trials: 1},
		{name: "zero usage", outcomes: []string{"success"}, zeroUsage: true, batches: 1, trials: 1},
		{name: "retry then success", outcomes: []string{"invalid", "success"}, batches: 1, trials: 1},
		{name: "error", outcomes: []string{"error"}, wantErr: providerErr},
		{name: "canceled", outcomes: []string{"canceled"}},
		{name: "deadline", outcomes: []string{"deadline"}, wantErr: context.DeadlineExceeded},
		{name: "retry then error", outcomes: []string{"invalid", "error"}, wantErr: providerErr},
		{name: "dropped", outcomes: []string{"invalid", "invalid", "invalid", "invalid", "invalid", "invalid", "invalid", "invalid", "invalid"}, trials: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var logs bytes.Buffer
			var calls int
			var reported Usage
			idsPattern := regexp.MustCompile("id: `([^`]+)`")
			p := &legacyTestProvider{complete: func(_ context.Context, prompt string, opts *CompletionOptions) (string, error) {
				outcome := tc.outcomes[calls%len(tc.outcomes)]
				calls++
				if !tc.zeroUsage {
					opts.Usage = Usage{InputTokens: 10 * calls, OutputTokens: calls, ReasoningTokens: 1}
				}
				reported.Add(opts.Usage)
				opts.ModelUsed = "test-model"
				opts.FinishReason = "test-finish"
				switch outcome {
				case "invalid":
					return "not JSON", nil
				case "error":
					return `{"docs":["invalid"]}`, providerErr
				case "canceled":
					return `{"docs":["invalid"]}`, fmt.Errorf("provider: %w", context.Canceled)
				case "deadline":
					return "", fmt.Errorf("provider: %w", context.DeadlineExceeded)
				}
				var ids []string
				for _, match := range idsPattern.FindAllStringSubmatch(prompt, -1) {
					ids = append(ids, match[1])
				}
				data, err := json.Marshal(rankedDocumentResponseNoRelevance{Documents: ids})
				return string(data), err
			}}
			cfg := NewConfig()
			cfg.InitialPrompt = "Rank items"
			cfg.LLMProvider = p
			cfg.Concurrency, cfg.BatchSize, cfg.NumTrials = 1, 2, 1
			cfg.EnableConvergence = false
			cfg.Logger = slog.New(slog.NewJSONHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}))
			r, err := NewRanker(cfg)
			if err != nil {
				t.Fatal(err)
			}
			r.numBatches = 1
			r.comparedAgainst = make(map[string]map[string]bool)
			for round := 1; round <= 2; round++ {
				r.round = round
				got, err := r.shuffleBatchRank([]document{{ID: "a"}, {ID: "b"}})
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("round %d: got error %v, want %v", round, err, tc.wantErr)
				}
				if len(got) != 2*tc.batches {
					t.Fatalf("round %d: got %d ranked items, want %d", round, len(got), 2*tc.batches)
				}
			}
			if calls != 2*len(tc.outcomes) || r.totalCalls != calls || r.totalUsage != reported {
				t.Fatalf("calls=%d/%d, usage=%+v; want calls=%d, usage=%+v", r.totalCalls, calls, r.totalUsage, 2*len(tc.outcomes), reported)
			}
			if r.totalBatches != 2*tc.batches || r.totalTrials != 2*tc.trials || r.totalRounds != 2 {
				t.Fatalf("batches=%d, trials=%d, rounds=%d", r.totalBatches, r.totalTrials, r.totalRounds)
			}
			var roundCalls, roundBatches, roundTrials, inputTokens, outputTokens, callLogs, roundLogs int
			decoder := json.NewDecoder(&logs)
			for decoder.More() {
				var event map[string]interface{}
				if err := decoder.Decode(&event); err != nil {
					t.Fatal(err)
				}
				switch event["msg"] {
				case "Round completed":
					roundLogs++
					roundCalls += int(event["num_calls"].(float64))
					roundBatches += int(event["num_batches"].(float64))
					roundTrials += int(event["num_trials"].(float64))
					inputTokens += int(event["input_tokens"].(float64))
					outputTokens += int(event["output_tokens"].(float64))
				case "LLM call returned":
					wantOutcome := tc.outcomes[callLogs%len(tc.outcomes)]
					if wantOutcome == "invalid" {
						wantOutcome = "success" // Provider returned; ranking validation follows.
					} else if wantOutcome == "deadline" {
						wantOutcome = "canceled"
					}
					if event["outcome"] != wantOutcome || event["model"] != "test-model" || event["finish_reason"] != "test-finish" {
						t.Errorf("incorrect call log: %+v", event)
					}
					callLogs++
				case "LLM call completed":
					t.Error("misleading call completion log")
				}
			}
			if roundLogs != 2 || callLogs != calls || roundCalls != r.totalCalls || roundBatches != r.totalBatches || roundTrials != r.totalTrials || inputTokens != reported.InputTokens || outputTokens != reported.OutputTokens {
				t.Fatalf("round summaries do not match run totals: rounds=%d, call logs=%d, calls=%d, batches=%d, trials=%d, input=%d, output=%d", roundLogs, callLogs, roundCalls, roundBatches, roundTrials, inputTokens, outputTokens)
			}
		})
	}
}

type usageAccountingHandler struct {
	slog.Handler
	beforeHandle func(slog.Record)
}

func (h usageAccountingHandler) Handle(ctx context.Context, record slog.Record) error {
	h.beforeHandle(record)
	return h.Handler.Handle(ctx, record)
}

func TestUsageAccountingConvergence(t *testing.T) {
	var logs bytes.Buffer
	lastCallStarted := make(chan struct{})
	var calls, completedTrials int
	idsPattern := regexp.MustCompile("id: `([^`]+)`\\nvalue:\\n```\\n([^\\n]+)")
	p := &legacyTestProvider{complete: func(ctx context.Context, prompt string, opts *CompletionOptions) (string, error) {
		calls++
		opts.Usage = Usage{InputTokens: 10, OutputTokens: 2, ReasoningTokens: 1}
		if calls == 8 {
			close(lastCallStarted)
			<-ctx.Done()
			return "", ctx.Err()
		}
		matches := idsPattern.FindAllStringSubmatch(prompt, -1)
		if len(matches) != 2 {
			return "", fmt.Errorf("expected two candidates, got %d", len(matches))
		}
		if matches[0][2] > matches[1][2] {
			matches[0], matches[1] = matches[1], matches[0]
		}
		data, err := json.Marshal(rankedDocumentResponseNoRelevance{Documents: []string{matches[0][1], matches[1][1]}})
		return string(data), err
	}}
	cfg := NewConfig()
	cfg.InitialPrompt = "Rank items"
	cfg.LLMProvider = p
	cfg.Concurrency, cfg.BatchSize, cfg.NumTrials = 1, 2, 5
	cfg.MinTrials, cfg.StableTrials = 2, 2
	cfg.ElbowTolerance = 0.75
	cfg.Logger = slog.New(usageAccountingHandler{
		Handler: slog.NewJSONHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}),
		beforeHandle: func(record slog.Record) {
			if record.Message == "Trial completed" {
				completedTrials++
				if completedTrials == 3 {
					// Hold the collector until trial 4 has one successful batch
					// buffered and its second call is waiting for cancellation.
					<-lastCallStarted
				}
			}
		},
	})
	r, err := NewRanker(cfg)
	if err != nil {
		t.Fatal(err)
	}
	r.rng = rand.New(rand.NewSource(1))
	r.round, r.numBatches = 1, 2
	r.comparedAgainst = make(map[string]map[string]bool)
	got, err := r.shuffleBatchRank([]document{{ID: "a", Value: "a"}, {ID: "b", Value: "b"}, {ID: "c", Value: "c"}, {ID: "d", Value: "d"}})
	if err != nil || len(got) != 4 || !r.converged {
		t.Fatalf("got %d items, converged=%v, error=%v", len(got), r.converged, err)
	}
	if calls != 8 || r.totalCalls != 8 || r.totalUsage != (Usage{InputTokens: 80, OutputTokens: 16, ReasoningTokens: 8}) {
		t.Fatalf("calls=%d/%d, usage=%+v", calls, r.totalCalls, r.totalUsage)
	}
	if r.totalBatches != 7 || r.totalTrials != 3 || r.totalRounds != 1 {
		t.Fatalf("batches=%d, trials=%d, rounds=%d", r.totalBatches, r.totalTrials, r.totalRounds)
	}
	var rounds int
	decoder := json.NewDecoder(&logs)
	for decoder.More() {
		var event map[string]interface{}
		if err := decoder.Decode(&event); err != nil {
			t.Fatal(err)
		}
		if event["msg"] == "Round completed" {
			rounds++
			if event["num_calls"] != float64(8) || event["num_batches"] != float64(7) || event["num_trials"] != float64(3) || event["input_tokens"] != float64(80) || event["output_tokens"] != float64(16) {
				t.Errorf("incorrect round summary: %+v", event)
			}
		}
	}
	if rounds != 1 {
		t.Fatalf("got %d round summaries", rounds)
	}
}

func TestNewRanker(t *testing.T) {
	tests := []struct {
		name    string
		config  *Config
		wantErr bool
	}{
		{
			name: "valid config",
			config: &Config{
				InitialPrompt:   "test prompt",
				BatchSize:       5,
				NumTrials:       2,
				Concurrency:     20,
				OpenAIModel:     openai.ChatModelGPT4oMini,
				RefinementRatio: 0.5,
				OpenAIKey:       "test-key",
				Encoding:        "o200k_base",
				BatchTokens:     2000,
				DryRun:          true,
			},
			wantErr: false,
		},
		{
			name: "empty prompt",
			config: &Config{
				InitialPrompt: "",
				BatchSize:     5,
				NumTrials:     2,
				OpenAIModel:   openai.ChatModelGPT4oMini,
				BatchTokens:   1000,
				OpenAIKey:     "test-key",
				Encoding:      "o200k_base",
			},
			wantErr: true,
		},
		{
			name: "invalid batch size",
			config: &Config{
				InitialPrompt: "test",
				BatchSize:     1, // Less than minBatchSize (2)
				NumTrials:     2,
				OpenAIModel:   openai.ChatModelGPT4oMini,
				BatchTokens:   1000,
				OpenAIKey:     "test-key",
				Encoding:      "o200k_base",
			},
			wantErr: true,
		},
		{
			name: "missing OpenAI key",
			config: &Config{
				InitialPrompt: "test",
				BatchSize:     5,
				NumTrials:     2,
				OpenAIModel:   openai.ChatModelGPT4oMini,
				BatchTokens:   1000,
				OpenAIKey:     "", // Empty key
				Encoding:      "o200k_base",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ranker, err := NewRanker(tt.config)
			if tt.wantErr {
				if err == nil {
					t.Errorf("NewRanker() expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Errorf("NewRanker() unexpected error: %v", err)
				return
			}
			if ranker == nil {
				t.Errorf("NewRanker() returned nil ranker")
			}
		})
	}
}

func TestRankFromFile_DryRun(t *testing.T) {
	// Create a temporary test file
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.txt")
	content := "apple\nbanana\ncherry"
	if err := os.WriteFile(testFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	config := &Config{
		InitialPrompt:   "Rank by alphabetical order",
		BatchSize:       3, // Set to 3 to include all items
		NumTrials:       1,
		Concurrency:     20,
		OpenAIModel:     openai.ChatModelGPT4oMini,
		RefinementRatio: 0.0, // Set to 0.0 to disable refinement and keep all results
		OpenAIKey:       "test-key",
		Encoding:        "o200k_base",
		BatchTokens:     2000,
		DryRun:          true, // Use dry run to avoid actual API calls
	}

	ranker, err := NewRanker(config)
	if err != nil {
		t.Fatalf("NewRanker() unexpected error: %v", err)
	}

	results, err := ranker.RankFromFile(testFile, "{{.Data}}", false)
	if err != nil {
		t.Errorf("RankFromFile() unexpected error: %v", err)
		return
	}

	if len(results) == 0 {
		t.Error("RankFromFile() returned no results")
		return
	}

	// Should get at least 2 results (algorithm may batch/filter items)
	if len(results) < 2 {
		t.Errorf("RankFromFile() expected at least 2 results, got %d", len(results))
		return
	}

	// Verify all results have required fields
	for i, result := range results {
		if result.Key == "" {
			t.Errorf("Result %d missing Key", i)
		}
		if result.Value == "" {
			t.Errorf("Result %d missing Value", i)
		}
		if result.Rank == 0 {
			t.Errorf("Result %d missing Rank", i)
		}
		if result.Exposure == 0 {
			t.Errorf("Result %d missing Exposure", i)
		}
	}

	// Verify ranks are properly assigned (1-based) for the results we got
	for i, result := range results {
		expectedRank := i + 1
		if result.Rank != expectedRank {
			t.Errorf("Result %d expected rank %d, got %d", i, expectedRank, result.Rank)
		}
	}
}

func TestRankFromFile_WithSentencesData(t *testing.T) {
	sentencesFile := filepath.Join("..", "..", "testdata", "sentences.txt")

	// Check if the testdata file exists
	if _, err := os.Stat(sentencesFile); os.IsNotExist(err) {
		t.Skip("testdata/sentences.txt not found, skipping integration test")
	}

	config := &Config{
		InitialPrompt:   `Rank each of these items according to their relevancy to the concept of "time".`,
		BatchSize:       10,
		NumTrials:       3,
		Concurrency:     20,
		OpenAIModel:     openai.ChatModelGPT4oMini,
		RefinementRatio: 0.5,
		OpenAIKey:       "test-key", // This would normally come from environment
		Encoding:        "o200k_base",
		BatchTokens:     128000,
		DryRun:          true, // Use dry run to avoid actual API calls
	}

	// Allow integration testing with real OpenAI API if API key is provided
	if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey != "" {
		config.OpenAIKey = apiKey
		config.DryRun = false
		// Optionally override the base URL (defaults to OpenAI's standard URL)
		if apiBase := os.Getenv("OPENAI_API_BASE"); apiBase != "" {
			config.OpenAIAPIURL = apiBase
		}
	}

	ranker, err := NewRanker(config)
	if err != nil {
		t.Fatalf("NewRanker() unexpected error: %v", err)
	}

	results, err := ranker.RankFromFile(sentencesFile, "{{.Data}}", false)
	if err != nil {
		t.Fatalf("RankFromFile() unexpected error: %v", err)
	}

	// Basic sanity checks
	if len(results) == 0 {
		t.Error("RankFromFile() returned no results")
		return
	}

	t.Logf("Successfully ranked %d items from sentences.txt", len(results))

	// Print top 10 results
	maxResults := 10
	if len(results) < maxResults {
		maxResults = len(results)
	}

	t.Log("Top 10 results by relevance to 'time':")
	for i := 0; i < maxResults; i++ {
		t.Logf("%d. %s", i+1, results[i].Value)
	}

	// If this is not a dry run (real API call), validate that at least one time-related
	// sentence appears in the top 3 results
	if !config.DryRun {
		timeRelatedSentences := []string{
			"The train arrived exactly on time.",
			"The clock ticked steadily on the wall.",
			"The old clock chimed twelve times.",
		}

		top3Results := results
		if len(results) > 3 {
			top3Results = results[:3]
		}

		foundTimeRelated := false
		for _, result := range top3Results {
			for _, timeSentence := range timeRelatedSentences {
				if result.Value == timeSentence {
					foundTimeRelated = true
					t.Logf("Found time-related sentence in top 3: %s (rank %d)", result.Value, result.Rank)
					break
				}
			}
			if foundTimeRelated {
				break
			}
		}

		if !foundTimeRelated {
			t.Logf("Warning: None of the expected time-related sentences found in top 3")
			t.Logf("Expected one of: %v", timeRelatedSentences)
			t.Logf("Got top 3: %v", []string{top3Results[0].Value, top3Results[1].Value, top3Results[2].Value})
			// Note: This is a warning, not a failure, since AI ranking can vary
		}
	}

	// Verify structure of results
	for i, result := range results[:maxResults] {
		if result.Key == "" {
			t.Errorf("Result %d missing Key", i)
		}
		if result.Value == "" {
			t.Errorf("Result %d missing Value", i)
		}
		if result.Rank != i+1 {
			t.Errorf("Result %d expected rank %d, got %d", i, i+1, result.Rank)
		}
	}
}

func TestRankFromFile_Errors(t *testing.T) {
	config := &Config{
		InitialPrompt:   "test prompt",
		BatchSize:       5,
		NumTrials:       3,
		Concurrency:     20,
		OpenAIModel:     openai.ChatModelGPT4oMini,
		RefinementRatio: 0.5,
		OpenAIKey:       "test-key",
		Encoding:        "o200k_base",
		BatchTokens:     2000,
		DryRun:          true,
	}

	ranker, err := NewRanker(config)
	if err != nil {
		t.Fatalf("NewRanker() unexpected error: %v", err)
	}

	// Test with non-existent file
	_, err = ranker.RankFromFile("nonexistent.txt", "{{.Data}}", false)
	if err == nil {
		t.Error("RankFromFile() with non-existent file should return error")
	}

	// Test with invalid template
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.txt")
	if err := os.WriteFile(testFile, []byte("test"), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	_, err = ranker.RankFromFile(testFile, "{{.InvalidField | badFunc}}", false)
	if err == nil {
		t.Error("RankFromFile() with invalid template should return error")
	}
}
