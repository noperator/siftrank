package siftrank

import (
	"context"

	"github.com/invopop/jsonschema"
)

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
