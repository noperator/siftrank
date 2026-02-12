package siftrank

import (
	"context"
)

// LLMProvider defines a common interface for interacting with various Large Language Model providers.
// Implementations must be safe for concurrent use from multiple goroutines.
//
// This interface abstracts away provider-specific details while maintaining flexibility
// for different authentication methods, API structures, and capabilities.
type LLMProvider interface {
	// Complete generates a completion based on the provided prompt and options.
	//
	// Parameters:
	//   - ctx: Context for cancellation and timeout control
	//   - prompt: The input text to generate a response from
	//   - options: Optional completion parameters (may be nil)
	//
	// Returns the raw completion text and any error encountered.
	// Transient errors (rate limits, timeouts) should be retried internally.
	// Only unrecoverable errors (auth failures, context cancelled) should be returned.
	Complete(ctx context.Context, prompt string, options *CompletionOptions) (string, error)
}

// TokenEstimator is an optional interface that LLMProviders can implement
// to provide accurate token counting for batch sizing optimization.
//
// If an LLMProvider does not implement TokenEstimator, siftrank will fall back
// to a rough approximation (~4 characters per token).
type TokenEstimator interface {
	// EstimateTokens returns the estimated token count for the given text.
	// This should match the tokenization used by the provider's model.
	EstimateTokens(text string) int
}

// ProviderCapabilities describes what features a provider supports.
// This allows siftrank to adapt its behavior based on provider capabilities.
type ProviderCapabilities struct {
	// SupportsStructuredOutput indicates if the provider supports JSON schema constraints
	SupportsStructuredOutput bool

	// SupportsStreaming indicates if the provider supports streaming completions
	SupportsStreaming bool

	// SupportedModels lists the model identifiers this provider can use
	SupportedModels []string

	// MaxTokens is the maximum context window size for this provider
	MaxTokens int
}

// CapabilityProvider is an optional interface that LLMProviders can implement
// to advertise their capabilities to the ranker.
type CapabilityProvider interface {
	// GetCapabilities returns the provider's feature set
	GetCapabilities() ProviderCapabilities
}
