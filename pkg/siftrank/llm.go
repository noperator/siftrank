package siftrank

import (
	"context"

	"github.com/invopop/jsonschema"
)

// TODO: Revisit provider naming and configuration in a future major version,
// including the OpenAI-specific config fields and CLI variable names.
// Preserve LLMProvider.Complete for compatibility for now; structured backends
// such as Jev use the optional ranking interface and an unsupported Complete stub.

// LLMProvider handles LLM interactions for ranking operations.
// Implementations handle network-level concerns (retries, rate limits, timeouts)
// but make no guarantees about response format.
//
// Complete may be called concurrently from multiple goroutines.
// Implementations must be safe for concurrent use.
type LLMProvider interface {
	// Complete sends a prompt and returns the raw LLM response.
	//
	// Parameters:
	//   - ctx: Context for cancellation/timeouts
	//   - prompt: The full prompt text to send
	//   - opts: Optional parameters and metadata (may be nil)
	//
	// Returns error only for unrecoverable issues (bad auth, context cancelled).
	// Transient errors (rate limits, timeouts, 5xx) should be retried internally.
	Complete(ctx context.Context, prompt string, opts *CompletionOptions) (string, error)
}

// TokenEstimator is an optional interface that LLMProviders can implement
// to provide accurate token counting for batch sizing.
//
// EstimateTokens may be called concurrently from multiple goroutines.
// Implementations must be safe for concurrent use.
//
// If an LLMProvider does not implement TokenEstimator, siftrank falls back
// to a rough approximation (~4 characters per token).
type TokenEstimator interface {
	EstimateTokens(text string) int
}

// RankingProvider is an optional SiftRank adapter capability that accepts
// structured ranking input and returns ranked-ID JSON. An adapter may convert
// backend probabilities into an ordering; the service need not rank items itself.
// Ranker uses CompleteRanking when available, otherwise LLMProvider.Complete.
// Implementations must be safe for concurrent calls. Existing LLMProvider
// implementations need no changes.
type RankingProvider interface {
	CompleteRanking(ctx context.Context, input RankingInput, opts *CompletionOptions) (string, error)
}

// RankingTokenEstimator optionally sizes structured requests instead of the chat
// prompt. Implementations must be safe for concurrent calls.
type RankingTokenEstimator interface {
	EstimateRankingTokens(input RankingInput) int
}

// RankingInput carries the ranking criteria and candidates for one batch.
type RankingInput struct {
	Prompt    string
	Documents []RankingCandidate
}

// RankingCandidate pairs a batch-local ID with the formatted item text.
type RankingCandidate struct {
	ID    string `json:"id"`
	Value string `json:"value"`
}

// rankingConfigurer handles provider-specific validation and configuration during
// ranker construction without changing Config. Context constraints are checked
// separately during batch fitting.
type rankingConfigurer interface {
	configureRanking(*Config) error
}

// Providers with multiple context constraints can check structured input during
// batch fitting. The returned estimate is the combined request token count.
type rankingBudgetChecker interface {
	checkRankingBudget(RankingInput, int) (int, error)
}

// CompletionOptions contains optional parameters for completion requests
// and receives metadata about the completion.
type CompletionOptions struct {
	// --- INPUTS (caller sets these before calling Complete) ---

	// Schema for structured output (JSON schema for constrained decoding).
	// If nil, no schema constraint is applied.
	Schema interface{}

	// Temperature for sampling (0.0 to 2.0, provider-specific).
	// Optional; if nil, provider uses its default.
	Temperature *float64

	// MaxTokens limits response length.
	// Optional; if nil, provider uses its default.
	MaxTokens *int

	// --- OUTPUTS (provider populates these during Complete) ---

	// Usage contains token consumption after the call completes.
	Usage Usage

	// ModelUsed is the actual model that generated the response.
	// May differ from requested model if provider substitutes.
	ModelUsed string

	// FinishReason indicates why generation stopped.
	// Common values: "stop" (natural end), "length" (hit max tokens),
	// "content_filter" (blocked by safety filter).
	// Optional; may be empty if provider doesn't report it.
	// Informational only; siftrank does not act on this value.
	FinishReason string

	// RequestID is the provider's identifier for this request.
	// Optional; may be empty if provider doesn't report it.
	// Useful for debugging or support requests with the provider.
	RequestID string

	// PairwiseComparisons optionally retains a validated provider comparison matrix.
	// IDs refer to the supplied RankingInput, before ordering. An empty slice means
	// no comparisons are available; these probabilities are not calibrated errors.
	PairwiseComparisons []PairwiseComparison
}

// PairwiseComparison records the probability that FirstID should rank ahead of
// SecondID under the supplied criterion. It does not change ranking scores.
type PairwiseComparison struct {
	FirstID     string  `json:"first_id"`
	SecondID    string  `json:"second_id"`
	Probability float64 `json:"probability"`
}

// Usage tracks token consumption for LLM calls
type Usage struct {
	InputTokens     int // Prompt tokens
	OutputTokens    int // Completion tokens
	ReasoningTokens int // Reasoning tokens (o1/o3 models)
}

// TotalTokens returns the sum of all token counts
func (u Usage) TotalTokens() int {
	return u.InputTokens + u.OutputTokens + u.ReasoningTokens
}

// Add adds another Usage's tokens to this Usage
func (u *Usage) Add(other Usage) {
	u.InputTokens += other.InputTokens
	u.OutputTokens += other.OutputTokens
	u.ReasoningTokens += other.ReasoningTokens
}

// generateSchema generates a JSON schema from a Go type
func generateSchema[T any]() interface{} {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	var v T
	schema := reflector.Reflect(v)
	return schema
}
