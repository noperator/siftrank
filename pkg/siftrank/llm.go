package siftrank

import (
	"context"

	"github.com/invopop/jsonschema"
)

// LLMProvider handles network-level LLM interactions.
// The provider handles network retries (rate limits, timeouts, server errors)
// but makes NO guarantees about response format. The response might be:
// - Valid JSON
// - JSON wrapped in markdown (```json ... ```)
// - Invalid JSON
// - Plain text
// The caller is responsible for all parsing and validation.
type LLMProvider interface {
	// Complete sends a prompt and returns the raw LLM response.
	// Main input: prompt string
	// Main output: response string
	// opts: optional parameters and metadata (can be nil)
	//
	// ctx: for cancellation/timeouts
	// prompt: the full prompt text
	// opts: optional completion options (schema, model params, outputs)
	//
	// Returns: raw response string, error
	// Error only for unrecoverable network issues (bad auth, context cancelled)
	Complete(ctx context.Context, prompt string, opts *CompletionOptions) (string, error)

	// EstimateTokens estimates token count for text.
	// Returns -1 if estimation not supported.
	EstimateTokens(text string) int
}

// CompletionOptions contains optional parameters for completion requests
// and receives metadata about the completion.
type CompletionOptions struct {
	// --- INPUTS (caller sets these before calling Complete) ---

	// Schema for structured output (hint for constrained decoding)
	Schema interface{}

	// Temperature for sampling (0.0 to 2.0, provider-specific)
	Temperature *float64

	// MaxTokens for response length limit
	MaxTokens *int

	// --- OUTPUTS (provider populates these during Complete) ---

	// Usage contains token consumption information
	Usage Usage

	// ModelUsed is the actual model that generated the response
	ModelUsed string

	// FinishReason indicates why generation stopped
	// Common values: "stop", "length", "content_filter"
	FinishReason string

	// RequestID from the provider for debugging/logging
	RequestID string
}

// Usage tracks token consumption for LLM calls
type Usage struct {
	InputTokens     int // Prompt tokens
	OutputTokens    int // Completion tokens
	ReasoningTokens int // Reasoning tokens (o1/o3 models)
	TotalTokens     int // Sum of all tokens
}

// Add adds another Usage's tokens to this Usage
func (u *Usage) Add(other Usage) {
	u.InputTokens += other.InputTokens
	u.OutputTokens += other.OutputTokens
	u.ReasoningTokens += other.ReasoningTokens
	u.TotalTokens += other.TotalTokens
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
