package siftrank

import (
	"fmt"
	"log/slog"

	"github.com/openai/openai-go"
)

// ElbowMethod specifies the algorithm for detecting the elbow point in rankings
type ElbowMethod string

const (
	ElbowMethodCurvature     ElbowMethod = "curvature"
	ElbowMethodPerpendicular ElbowMethod = "perpendicular"
)

// Default configuration values
const (
	DefaultBatchSize         = 10
	DefaultNumTrials         = 50
	DefaultConcurrency       = 50
	DefaultBatchTokens       = 128000
	DefaultRefinementRatio   = 0.5
	DefaultEncoding          = "o200k_base"
	DefaultTemplate          = "{{.Data}}"
	DefaultElbowTolerance    = 0.05
	DefaultStableTrials      = 5
	DefaultMinTrials         = 5
	DefaultElbowMethod       = ElbowMethodCurvature
	DefaultEnableConvergence = true
	DefaultOpenAIModel       = openai.ChatModelGPT4oMini
)

// Config contains all configuration options for a Ranker.
// Use NewConfig() to get sensible defaults, then override as needed.
type Config struct {
	// InitialPrompt is the ranking instruction sent to the LLM.
	// Example: "Rank these items by relevance to machine learning"
	InitialPrompt string `json:"initial_prompt" yaml:"prompt"`

	// BatchSize is the number of documents compared per LLM call.
	// Smaller batches = more accurate, larger batches = faster.
	BatchSize int `json:"batch_size" yaml:"batch_size"`

	// NumTrials is the maximum number of ranking trials to run.
	// More trials improve stability but increase cost/time.
	NumTrials int `json:"num_trials" yaml:"max_trials"`

	// Concurrency is the maximum concurrent LLM calls across all trials.
	Concurrency int `json:"concurrency" yaml:"concurrency"`

	// RefinementRatio controls how many top documents are re-ranked (0.0-1.0).
	// 0.5 means top 50% are refined in subsequent rounds.
	RefinementRatio float64 `json:"refinement_ratio" yaml:"ratio"`

	// BatchTokens is the maximum tokens per batch (for batch sizing).
	BatchTokens int `json:"batch_tokens" yaml:"tokens"`

	// DryRun logs API calls without making them (for testing).
	DryRun bool `json:"-" yaml:"-"`

	// TracePath writes JSON Lines trace output to this file path.
	// Empty string disables tracing.
	TracePath string `json:"-" yaml:"trace"`

	// OutputFile writes JSON results to this file path in addition to stdout.
	// Empty string disables file output.
	OutputFile string `json:"-" yaml:"output"`

	// ForceJSON parses input as a JSON array regardless of file extension.
	ForceJSON bool `json:"-" yaml:"json"`

	// Template is the Go template string (or @file path) for formatting each input item.
	Template string `json:"-" yaml:"template"`

	// LogFile writes logs to this file path instead of stderr.
	// Empty string writes to stderr.
	LogFile string `json:"-" yaml:"log"`

	// Debug enables debug-level logging.
	Debug bool `json:"-" yaml:"debug"`

	// Relevance enables post-processing to generate pros/cons for each item.
	Relevance bool `json:"relevance" yaml:"relevance"`

	// LogLevel controls logging verbosity (slog.LevelInfo, slog.LevelDebug, etc).
	LogLevel slog.Level `json:"-" yaml:"-"`

	// Logger is the structured logger for output. If nil, a default is created.
	Logger *slog.Logger `json:"-" yaml:"-"`

	// EnableConvergence enables early stopping when rankings stabilize.
	EnableConvergence bool `json:"enable_convergence" yaml:"enable_convergence"`

	// ElbowTolerance is the allowed variance in elbow position (0.05 = 5%).
	ElbowTolerance float64 `json:"elbow_tolerance" yaml:"elbow_tolerance"`

	// StableTrials is how many consecutive trials must agree for convergence.
	StableTrials int `json:"stable_trials" yaml:"stable_trials"`

	// MinTrials is the minimum trials before checking convergence.
	MinTrials int `json:"min_trials" yaml:"min_trials"`

	// ElbowMethod selects the algorithm for elbow detection.
	// ElbowMethodCurvature (default) or ElbowMethodPerpendicular.
	ElbowMethod ElbowMethod `json:"elbow_method" yaml:"elbow_method"`

	// LLMProvider handles LLM calls. If nil, creates default OpenAI provider.
	LLMProvider LLMProvider `json:"-" yaml:"-"`

	// OpenAI configuration (used only if LLMProvider is nil).
	// json:"-" is intentional — prevents secrets from leaking into JSON output.
	OpenAIModel  openai.ChatModel `json:"openai_model" yaml:"model"` // Model name (e.g., "gpt-4o-mini")
	OpenAIKey    string           `json:"-" yaml:"api_key"`          // API key; may be omitted for local/unauthenticated endpoints
	OpenAIAPIURL string           `json:"-" yaml:"base_url"`         // Base URL override (e.g., for vLLM or other OpenAI-compatible APIs)

	// Encoding is the tokenizer encoding name (e.g., "o200k_base").
	// Used only by the default OpenAI provider for accurate token counting.
	// Custom LLMProvider implementations can ignore this field.
	Encoding string `json:"encoding" yaml:"encoding"`

	// Effort is the reasoning effort level: none, minimal, low, medium, high.
	Effort string `json:"effort" yaml:"effort"`

	// Watch enables live terminal visualization (CLI only).
	Watch bool `json:"-" yaml:"-"`

	// NoMinimap disables the minimap panel in watch mode (CLI only).
	NoMinimap bool `json:"-" yaml:"-"`
}

func (c *Config) Validate() error {
	if c.InitialPrompt == "" {
		return fmt.Errorf("initial prompt cannot be empty")
	}
	if c.BatchSize <= 0 {
		return fmt.Errorf("batch size must be greater than 0")
	}
	if c.NumTrials <= 0 {
		return fmt.Errorf("number of trials must be greater than 0")
	}
	if c.Concurrency <= 0 {
		return fmt.Errorf("concurrency must be greater than 0")
	}
	if c.BatchTokens <= 0 {
		return fmt.Errorf("batch tokens must be greater than 0")
	}
	if c.RefinementRatio < 0 || c.RefinementRatio >= 1 {
		return fmt.Errorf("refinement ratio must be >= 0 and < 1")
	}
	if c.BatchSize < minBatchSize {
		return fmt.Errorf("batch size must be at least %d", minBatchSize)
	}
	switch c.Effort {
	case "", "none", "minimal", "low", "medium", "high":
		// valid; empty string means "not set", use provider default
	default:
		return fmt.Errorf("effort must be one of: none, minimal, low, medium, high; got %q", c.Effort)
	}
	if c.ElbowMethod != "" && c.ElbowMethod != ElbowMethodCurvature && c.ElbowMethod != ElbowMethodPerpendicular {
		return fmt.Errorf("elbow method must be ElbowMethodCurvature or ElbowMethodPerpendicular, got '%s'", c.ElbowMethod)
	}
	// Only require OpenAI key if no provider is set
	if c.LLMProvider == nil && c.OpenAIAPIURL == "" && c.OpenAIKey == "" {
		return fmt.Errorf("openai key cannot be empty")
	}
	if c.EnableConvergence {
		if c.ElbowTolerance < 0 || c.ElbowTolerance >= 1 {
			return fmt.Errorf("elbow tolerance must be >= 0 and < 1")
		}
		if c.StableTrials < 2 {
			return fmt.Errorf("stable trials must be at least 2")
		}
		if c.MinTrials < 2 {
			return fmt.Errorf("minimum trials must be at least 2")
		}
	}
	return nil
}

// NewConfig returns a Config with sensible defaults matching the CLI.
// Callers must set at minimum: InitialPrompt and either OpenAIKey or LLMProvider.
func NewConfig() *Config {
	return &Config{
		BatchSize:         DefaultBatchSize,
		NumTrials:         DefaultNumTrials,
		Concurrency:       DefaultConcurrency,
		BatchTokens:       DefaultBatchTokens,
		RefinementRatio:   DefaultRefinementRatio,
		Encoding:          DefaultEncoding,
		Template:          DefaultTemplate,
		ElbowMethod:       DefaultElbowMethod,
		ElbowTolerance:    DefaultElbowTolerance,
		StableTrials:      DefaultStableTrials,
		MinTrials:         DefaultMinTrials,
		EnableConvergence: DefaultEnableConvergence,
		OpenAIModel:       DefaultOpenAIModel,
	}
}
