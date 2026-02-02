package siftrank

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"syscall"
	"text/template"
	"time"

	"github.com/gdamore/tcell/v2"
	"github.com/openai/openai-go"
)

const (
	idLen        = 8
	minBatchSize = 2
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
	DefaultElbowTolerance    = 0.05
	DefaultStableTrials      = 5
	DefaultMinTrials         = 5
	DefaultElbowMethod       = ElbowMethodCurvature
	DefaultEnableConvergence = true
)

// Word lists for generating memorable IDs
var (
	adjectives = []string{
		"apt", "bad", "big", "coy", "dim", "dry", "far", "fat", "fit", "fun",
		"hip", "hot", "icy", "lax", "low", "mad", "mid", "net", "new", "old",
		"pat", "raw", "red", "sad", "shy", "tan", "wet",
	}
	nouns = []string{
		"act", "age", "aid", "air", "ant", "ape", "arm", "art", "ash", "bag",
		"bar", "bat", "bay", "bed", "bet", "bid", "bin", "bit", "bog", "bow",
		"box", "boy", "bud", "bug", "bun", "bus", "can", "cap", "car", "cat",
		"cob", "cot", "cow", "cub", "cup", "cut", "dad", "dam", "day", "den",
		"dew", "dog", "dot", "ear", "elf", "elk", "elm", "emu", "end", "era",
		"eye", "fan", "fax", "fig", "fix", "flu", "fly", "fob", "fog", "fox",
		"fur", "gap", "gas", "gem", "gum", "guy", "gym", "hat", "hay", "hen",
		"hip", "hit", "hog", "hot", "hut", "ice", "ink", "jam", "jar", "jaw",
		"job", "joy", "jug", "keg", "key", "kid", "lab", "lap", "law", "leg",
		"lie", "lip", "log", "lot", "man", "map", "mat", "mix", "mom", "mop",
		"mud", "mug", "net", "nut", "oak", "oar", "oil", "one", "owl", "pad",
		"pan", "paw", "pea", "pen", "pet", "pew", "pie", "pig", "pin", "pop",
		"pot", "rag", "ram", "rat", "ray", "rim", "rip", "rod", "row", "rub",
		"rug", "rum", "run", "saw", "sea", "sir", "sky", "son", "sow", "soy",
		"spy", "sun", "tax", "tea", "tie", "tin", "tip", "toe", "tom", "ton",
		"top", "toy", "tub", "urn", "van", "wad", "war", "wax", "way", "web",
		"yak", "yam",
	}
)

// Config contains all configuration options for a Ranker.
// Use NewConfig() to get sensible defaults, then override as needed.
type Config struct {
	// InitialPrompt is the ranking instruction sent to the LLM.
	// Example: "Rank these items by relevance to machine learning"
	InitialPrompt string `json:"initial_prompt"`

	// BatchSize is the number of documents compared per LLM call.
	// Smaller batches = more accurate, larger batches = faster.
	BatchSize int `json:"batch_size"`

	// NumTrials is the maximum number of ranking trials to run.
	// More trials improve stability but increase cost/time.
	NumTrials int `json:"num_trials"`

	// Concurrency is the maximum concurrent LLM calls across all trials.
	Concurrency int `json:"concurrency"`

	// RefinementRatio controls how many top documents are re-ranked (0.0-1.0).
	// 0.5 means top 50% are refined in subsequent rounds.
	RefinementRatio float64 `json:"refinement_ratio"`

	// BatchTokens is the maximum tokens per batch (for batch sizing).
	BatchTokens int `json:"batch_tokens"`

	// DryRun logs API calls without making them (for testing).
	DryRun bool `json:"-"`

	// TracePath writes JSON Lines trace output to this file path.
	// Empty string disables tracing.
	TracePath string `json:"-"`

	// Relevance enables post-processing to generate pros/cons for each item.
	Relevance bool `json:"relevance"`

	// LogLevel controls logging verbosity (slog.LevelInfo, slog.LevelDebug, etc).
	LogLevel slog.Level `json:"-"`

	// Logger is the structured logger for output. If nil, a default is created.
	Logger *slog.Logger `json:"-"`

	// EnableConvergence enables early stopping when rankings stabilize.
	EnableConvergence bool `json:"enable_convergence"`

	// ElbowTolerance is the allowed variance in elbow position (0.05 = 5%).
	ElbowTolerance float64 `json:"elbow_tolerance"`

	// StableTrials is how many consecutive trials must agree for convergence.
	StableTrials int `json:"stable_trials"`

	// MinTrials is the minimum trials before checking convergence.
	MinTrials int `json:"min_trials"`

	// ElbowMethod selects the algorithm for elbow detection.
	// ElbowMethodCurvature (default) or ElbowMethodPerpendicular.
	ElbowMethod ElbowMethod `json:"elbow_method"`

	// LLMProvider handles LLM calls. If nil, creates default OpenAI provider.
	LLMProvider LLMProvider `json:"-"`

	// OpenAI configuration (used only if LLMProvider is nil)
	OpenAIModel  openai.ChatModel `json:"openai_model"` // Model name (e.g., "gpt-4o-mini")
	OpenAIKey    string           `json:"-"`            // API key (required if LLMProvider is nil)
	OpenAIAPIURL string           `json:"-"`            // Base URL (for compatible APIs like vLLM)

	// Encoding is the tokenizer encoding name (e.g., "o200k_base").
	// Used only by the default OpenAI provider for accurate token counting.
	// Custom LLMProvider implementations can ignore this field.
	Encoding string `json:"encoding"`

	// Effort is the reasoning effort level: none, minimal, low, medium, high.
	Effort string `json:"effort"`

	// Watch enables live terminal visualization (CLI only).
	Watch bool `json:"-"`

	// NoMinimap disables the minimap panel in watch mode (CLI only).
	NoMinimap bool `json:"-"`
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
	// Only require OpenAI key if no provider is set
	if c.LLMProvider == nil && c.OpenAIAPIURL == "" && c.OpenAIKey == "" {
		return fmt.Errorf("openai key cannot be empty")
	}
	if c.BatchSize < minBatchSize {
		return fmt.Errorf("batch size must be at least %d", minBatchSize)
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
	if c.ElbowMethod != "" && c.ElbowMethod != ElbowMethodCurvature && c.ElbowMethod != ElbowMethodPerpendicular {
		return fmt.Errorf("elbow method must be ElbowMethodCurvature or ElbowMethodPerpendicular, got '%s'", c.ElbowMethod)
	}
	return nil
}

// NewConfig returns a Config with sensible defaults matching the CLI.
// Callers should set at minimum: InitialPrompt and OpenAIKey (or LLMProvider).
func NewConfig() *Config {
	return &Config{
		BatchSize:         DefaultBatchSize,
		NumTrials:         DefaultNumTrials,
		Concurrency:       DefaultConcurrency,
		BatchTokens:       DefaultBatchTokens,
		RefinementRatio:   DefaultRefinementRatio,
		Encoding:          DefaultEncoding,
		ElbowMethod:       DefaultElbowMethod,
		ElbowTolerance:    DefaultElbowTolerance,
		StableTrials:      DefaultStableTrials,
		MinTrials:         DefaultMinTrials,
		EnableConvergence: DefaultEnableConvergence,
	}
}

type docStats struct {
	ID                string
	Value             string
	Document          interface{}
	InputIndex        int      // Index in original input (0-based)
	relevanceSnippets []string // Collected relevance from all batches/trials
}

type Ranker struct {
	cfg              *Config
	provider         LLMProvider
	rng              *rand.Rand
	numBatches       int
	round            int
	semaphore        chan struct{}              // Global concurrency limiter
	elbowPositions   []int                      // Track elbow position after each trial
	rankingOrders    [][]string                 // Track full ranking order per trial
	mu               sync.Mutex                 // Protect elbowPositions, rankingOrders, converged, comparedAgainst, and allDocStats
	converged        bool                       // Track if convergence already detected
	elbowCutoff      int                        // Cutoff position for refinement
	originalDocCount int                        // Track original dataset size for exposure calculation
	comparedAgainst  map[string]map[string]bool // Track which docs each was compared against (across ALL rounds/trials)
	allDocStats      map[string]*docStats       // Track all documents across rounds (for relevance collection)
	traceFile        *os.File                   // Keep file open across all rounds
	screen           interface{}                // tcell.Screen for terminal visualization (interface{} to avoid import cycle)

	// Token and call tracking (accumulate across all rounds)
	totalUsage   Usage
	totalCalls   int
	totalBatches int
	totalTrials  int
	totalRounds  int
}

func NewRanker(config *Config) (*Ranker, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}

	// Initialize default logger if not provided
	if config.Logger == nil {
		config.Logger = slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
			Level:     config.LogLevel,
			AddSource: false,
		})).With("component", "siftrank")
	}

	// Create provider (default to OpenAI if none specified)
	provider := config.LLMProvider
	if provider == nil {
		var err error
		provider, err = NewOpenAIProvider(OpenAIConfig{
			APIKey:   config.OpenAIKey,
			Model:    config.OpenAIModel,
			BaseURL:  config.OpenAIAPIURL,
			Encoding: config.Encoding,
			Effort:   config.Effort,
			Logger:   config.Logger,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create OpenAI provider: %w", err)
		}
	}

	return &Ranker{
		cfg:       config,
		provider:  provider,
		rng:       rand.New(rand.NewSource(time.Now().UnixNano())),
		semaphore: make(chan struct{}, config.Concurrency),
	}, nil
}

// adjustBatchSize dynamically adjusts batch size to fit within token limits
// by testing the worst case: the N largest documents
func (ranker *Ranker) adjustBatchSize(documents []document) error {
	// Calculate token size for each document
	docSizes := make([]struct {
		doc    document
		tokens int
	}, len(documents))

	for i, doc := range documents {
		docSizes[i].doc = doc
		docSizes[i].tokens = ranker.estimateTokens([]document{doc}, false)
	}

	// Sort by token size descending (largest first)
	sort.Slice(docSizes, func(i, j int) bool {
		return docSizes[i].tokens > docSizes[j].tokens
	})

	// Try to fit the N largest documents in a batch
	for {
		// Take the N largest documents (worst case)
		batchSize := min(ranker.cfg.BatchSize, len(docSizes))
		largestDocs := make([]document, batchSize)
		for i := 0; i < batchSize; i++ {
			largestDocs[i] = docSizes[i].doc
		}

		// Estimate tokens for this worst-case batch
		estBatchTokens := ranker.estimateTokens(largestDocs, true)

		if estBatchTokens <= ranker.cfg.BatchTokens {
			// Success! The largest documents fit
			ranker.cfg.Logger.Debug("Batch size validated",
				"batch_size", ranker.cfg.BatchSize,
				"worst_case_tokens", estBatchTokens,
				"max_tokens", ranker.cfg.BatchTokens,
				"utilization_pct", float64(estBatchTokens)/float64(ranker.cfg.BatchTokens)*100)
			return nil
		}

		// Batch too large - log details and decrease size
		ranker.cfg.Logger.Debug("Batch exceeds token limit",
			"batch_size", ranker.cfg.BatchSize,
			"estimated_tokens", estBatchTokens,
			"max_tokens", ranker.cfg.BatchTokens)
		ranker.logTokenSizes(largestDocs)

		if ranker.cfg.BatchSize <= minBatchSize {
			return fmt.Errorf("cannot create a valid batch within the token limit (even with batch size %d)", minBatchSize)
		}

		ranker.cfg.BatchSize--
		ranker.cfg.Logger.Debug("Decreasing batch size", "new_size", ranker.cfg.BatchSize)
	}
}

type document struct {
	ID         string      `json:"id"`
	Value      string      `json:"value"`    // to be ranked
	Document   interface{} `json:"document"` // if loading from json file
	InputIndex int         // Index in original input (0-based)
}

type rankedDocument struct {
	Document document
	Score    float64
}

type documentRelevance struct {
	ID   string `json:"id" jsonschema_description:"Document ID"`
	Text string `json:"text" jsonschema_description:"Brief explanation of qualities that make this document more or less relevant"`
}

// Used for parsing API responses (can handle both with and without relevance)
type rankedDocumentResponse struct {
	Documents []string            `json:"docs"`
	Relevance []documentRelevance `json:"relevance,omitempty"`
}

// Used for schema generation in round 1 or when relevance disabled
type rankedDocumentResponseNoRelevance struct {
	Documents []string `json:"docs" jsonschema_description:"List of ranked document IDs"`
}

// Used for schema generation in rounds 2+ when relevance enabled
type rankedDocumentResponseWithRelevance struct {
	Documents []string            `json:"docs" jsonschema_description:"List of ranked document IDs"`
	Relevance []documentRelevance `json:"relevance" jsonschema_description:"Brief explanation of qualities for each document"`
}

type RelevanceProsCons struct {
	Pros string `json:"pros"` // Qualities making item MORE relevant
	Cons string `json:"cons"` // Qualities making item LESS relevant
}

type RankedDocument struct {
	Key        string             `json:"key"`
	Value      string             `json:"value"`
	Document   interface{}        `json:"document"`    // if loading from json file
	Score      float64            `json:"score"`
	Exposure   float64            `json:"exposure"`    // percentage of dataset compared against (0.0-1.0)
	Rank       int                `json:"rank"`
	Rounds     int                `json:"rounds"`              // number of rounds participated in
	Relevance  *RelevanceProsCons `json:"relevance,omitempty"` // Only if relevance enabled
	InputIndex int                `json:"input_index"` // Index in original input (0-based)
}

type traceDocument struct {
	ID    string  `json:"id"`
	Value string  `json:"value"`
	Score float64 `json:"score"`
}

type traceLine struct {
	Round             int             `json:"round"`
	Trial             int             `json:"trial"`
	TrialsCompleted   int             `json:"trials_completed"`
	TrialsRemaining   int             `json:"trials_remaining"`
	TotalInputTokens  int             `json:"total_input_tokens"`
	TotalOutputTokens int             `json:"total_output_tokens"`
	ElbowPosition     *int            `json:"elbow_position,omitempty"`      // nil if not detected
	StableTrialsCount int             `json:"stable_trials_count,omitempty"` // only if convergence enabled
	Rankings          []traceDocument `json:"rankings"`
}

// createIDMappings generates memorable temporary IDs for a batch of documents
func createIDMappings(documents []document, rng *rand.Rand, logger *slog.Logger) (map[string]string, map[string]string, error) {
	originalToTemp := make(map[string]string)
	tempToOriginal := make(map[string]string)
	usedCombos := make(map[string]bool)

	maxAttempts := len(adjectives) * len(nouns) * 2 // Allow some randomness

	for _, doc := range documents {
		attempts := 0
		found := false

		for attempts < maxAttempts && !found {
			adj := adjectives[rng.Intn(len(adjectives))]
			noun := nouns[rng.Intn(len(nouns))]
			combination := adj + noun

			// Check for consecutively repeated characters
			hasRepeats := false
			for i := 0; i < len(combination)-1; i++ {
				if combination[i] == combination[i+1] {
					hasRepeats = true
					break
				}
			}

			// If no repeats and not used, use this combination
			if !hasRepeats && !usedCombos[combination] {
				usedCombos[combination] = true
				originalToTemp[doc.ID] = combination
				tempToOriginal[combination] = doc.ID
				found = true
			}

			attempts++
		}

		if !found {
			// Fall back to original IDs if we can't generate memorable ones
			logger.Warn("Failed to generate memorable IDs, falling back to original IDs", "error", "unable to generate unique memorable ID")
			return nil, nil, fmt.Errorf("unable to generate unique memorable ID after %d attempts", maxAttempts)
		}
	}

	return originalToTemp, tempToOriginal, nil
}

// translateIDsInResponse translates temporary IDs back to original IDs in the response
func translateIDsInResponse(response *rankedDocumentResponse, tempToOriginal map[string]string) {
	// Translate document IDs
	for i, id := range response.Documents {
		if originalID, exists := tempToOriginal[id]; exists {
			response.Documents[i] = originalID
		}
	}
	// Translate IDs in relevance array
	for i := range response.Relevance {
		if originalID, exists := tempToOriginal[response.Relevance[i].ID]; exists {
			response.Relevance[i].ID = originalID
		}
	}
}

// getResponseSchema returns the appropriate schema based on whether relevance is enabled
func (r *Ranker) getResponseSchema() interface{} {
	// Use relevance schema when relevance is enabled AND we're past round 1
	if r.cfg.Relevance && r.round > 1 {
		return generateSchema[rankedDocumentResponseWithRelevance]()
	}
	// For round 1 or when relevance is disabled, use schema without relevance field
	return generateSchema[rankedDocumentResponseNoRelevance]()
}

// ShortDeterministicID generates a deterministic ID of specified length from input string.
// It uses SHA-256 hash and Base64 encoding, keeping only alphanumeric characters.
func ShortDeterministicID(input string, length int) string {
	// Keep only A-Za-z0-9 from Base64-encoded SHA-256 hash.
	hash := sha256.Sum256([]byte(input))
	base64Encoded := base64.URLEncoding.EncodeToString(hash[:])
	var result strings.Builder
	for _, char := range base64Encoded {
		if (char >= '0' && char <= '9') || (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') {
			result.WriteRune(char)
		}
	}
	filtered := result.String()
	if length > len(filtered) {
		length = len(filtered)
	}
	return filtered[:length]
}

// RankFromFile ranks documents loaded from a file.
//
// Parameters:
//   - filePath: Path to input file (text or JSON)
//   - templateData: Go template for formatting each item. For text files,
//     use {{.Data}} to reference each line. For JSON, use field names like
//     {{.title}}. Prefix with @ to load template from file (e.g., "@template.txt").
//   - forceJSON: If true, parse as JSON regardless of file extension.
//
// For text files, each non-empty line becomes a document.
// For JSON files, expects an array of objects.
//
// Returns ranked documents sorted by score (lower = better), or error if
// ranking fails (e.g., LLM auth error, invalid input).
func (r *Ranker) RankFromFile(filePath string, templateData string, forceJSON bool) ([]*RankedDocument, error) {
	documents, err := r.loadDocumentsFromFile(filePath, templateData, forceJSON)
	if err != nil {
		return nil, err
	}

	// Open trace file if specified (only makes sense for file-based operation)
	if r.cfg.TracePath != "" {
		traceFile, err := os.Create(r.cfg.TracePath)
		if err != nil {
			return nil, fmt.Errorf("failed to create trace file: %w", err)
		}
		r.traceFile = traceFile
		defer func() {
			if err := r.traceFile.Close(); err != nil {
				r.cfg.Logger.Warn("Failed to close trace file", "error", err)
			}
		}()
	}

	return r.rankDocuments(documents)
}

// RankFromReader ranks documents read from an io.Reader.
//
// Parameters:
//   - reader: Source of input data
//   - templateData: Go template for formatting each item (see RankFromFile)
//   - isJSON: If true, parse as JSON array; if false, parse as text lines.
//
// Returns ranked documents sorted by score (lower = better), or error if
// ranking fails.
func (r *Ranker) RankFromReader(reader io.Reader, templateData string, isJSON bool) ([]*RankedDocument, error) {
	documents, err := r.loadDocumentsFromReader(reader, templateData, isJSON)
	if err != nil {
		return nil, err
	}

	return r.rankDocuments(documents)
}

// rankDocuments performs the core ranking logic on a set of documents.
func (r *Ranker) rankDocuments(documents []document) ([]*RankedDocument, error) {
	// check that no document is too large
	for _, doc := range documents {
		tokens := r.estimateTokens([]document{doc}, true)
		if tokens > r.cfg.BatchTokens {
			return nil, fmt.Errorf("document is too large with %d tokens:\n%s", tokens, doc.Value)
		}
	}

	if err := r.adjustBatchSize(documents); err != nil {
		return nil, err
	}

	// Initialize terminal visualization if enabled
	if r.cfg.Watch {
		screen, err := tcell.NewScreen()
		if err != nil {
			r.cfg.Logger.Warn("Failed to create screen for visualization", "error", err)
			r.cfg.Watch = false // Disable watch mode
		} else {
			if err := screen.Init(); err != nil {
				r.cfg.Logger.Warn("Failed to initialize screen for visualization", "error", err)
				r.cfg.Watch = false // Disable watch mode
			} else {
				r.screen = screen
				defer func() {
					if r.screen != nil {
						if s, ok := r.screen.(tcell.Screen); ok {
							s.Fini()
						}
					}
				}()

				// Setup signal and keyboard event handlers
				sigChan := make(chan os.Signal, 1)
				signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

				// Keyboard event handler
				go func() {
					for {
						ev := screen.PollEvent()
						switch ev := ev.(type) {
						case *tcell.EventKey:
							if ev.Key() == tcell.KeyCtrlC || ev.Key() == tcell.KeyEscape || ev.Rune() == 'q' {
								if s, ok := r.screen.(tcell.Screen); ok {
									s.Fini()
								}
								os.Exit(0)
							}
						case *tcell.EventResize:
							screen.Sync()
						}
					}
				}()

				// Signal handler
				go func() {
					<-sigChan
					if s, ok := r.screen.(tcell.Screen); ok {
						s.Fini()
					}
					os.Exit(0)
				}()
			}
		}
	}

	// Initialize global comparison tracking for exposure calculation
	r.comparedAgainst = make(map[string]map[string]bool)

	// Initialize relevance tracking if enabled
	if r.cfg.Relevance {
		r.allDocStats = make(map[string]*docStats)
		for _, doc := range documents {
			r.allDocStats[doc.ID] = &docStats{
				ID:                doc.ID,
				Value:             doc.Value,
				Document:          doc.Document,
				InputIndex:        doc.InputIndex,
				relevanceSnippets: []string{},
			}
		}
	}

	results, err := r.rank(documents, 1)
	if err != nil {
		return nil, err
	}

	// Summarize relevance if enabled
	if r.cfg.Relevance {
		r.cfg.Logger.Info("Summarizing relevance for all documents", "count", len(results))

		// Parallelize summarization using goroutines
		type summaryJob struct {
			index    int
			key      string
			value    string
			snippets []string
		}

		type summaryResult struct {
			index   int
			summary *RelevanceProsCons
			usage   Usage
			err     error
		}

		jobs := make(chan summaryJob, len(results))
		resultsChan := make(chan summaryResult, len(results))

		// Start worker pool (use r.cfg.Concurrency workers)
		numWorkers := r.cfg.Concurrency
		var wg sync.WaitGroup
		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				for job := range jobs {
					r.cfg.Logger.Info("Summarizing relevance",
						"document", job.index+1,
						"of", len(results),
						"snippets", len(job.snippets),
						"worker", workerID)
					summary, usage, err := r.summarizeRelevance(job.key, job.value, job.snippets)
					resultsChan <- summaryResult{index: job.index, summary: summary, usage: usage, err: err}
				}
			}(w)
		}

		// Queue all jobs (only for documents with collected relevance)
		itemsToSummarize := 0
		for i := range results {
			if stats, exists := r.allDocStats[results[i].Key]; exists && len(stats.relevanceSnippets) > 0 {
				jobs <- summaryJob{
					index:    i,
					key:      results[i].Key,
					value:    results[i].Value,
					snippets: stats.relevanceSnippets,
				}
				itemsToSummarize++
			} else {
				// No relevance collected (e.g., eliminated in round 1)
				results[i].Relevance = nil
			}
		}
		close(jobs)

		// Wait for workers to complete and collect results
		go func() {
			wg.Wait()
			close(resultsChan)
		}()

		// Collect results and track usage
		for result := range resultsChan {
			// Track usage (even on error, some tokens may have been used)
			r.mu.Lock()
			r.totalUsage.Add(result.usage)
			if result.usage.InputTokens > 0 || result.usage.OutputTokens > 0 {
				r.totalCalls++
			}
			r.mu.Unlock()

			if result.err != nil {
				r.cfg.Logger.Warn("Failed to summarize relevance", "document", result.index, "error", result.err)
				results[result.index].Relevance = nil
			} else {
				results[result.index].Relevance = result.summary
			}
		}
	}

	// Calculate final exposure percentages across all rounds/trials
	for i := range results {
		// Add rank
		results[i].Rank = i + 1

		// Calculate exposure as percentage of original dataset
		uniqueComparisons := float64(len(r.comparedAgainst[results[i].Key]))
		totalPossible := float64(r.originalDocCount - 1) // Exclude self
		if totalPossible > 0 {
			results[i].Exposure = uniqueComparisons / totalPossible
		} else {
			results[i].Exposure = 1.0 // Edge case: single document
		}
	}

	// Log final totals
	r.cfg.Logger.Info("Ranking completed",
		"num_rounds", r.totalRounds,
		"num_trials", r.totalTrials,
		"num_batches", r.totalBatches,
		"num_calls", r.totalCalls,
		"input_tokens", r.totalUsage.InputTokens,
		"output_tokens", r.totalUsage.OutputTokens)

	return results, nil
}

func (r *Ranker) loadDocumentsFromFile(filePath string, templateData string, forceJSON bool) ([]document, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open input file %s: %w", filePath, err)
	}
	defer file.Close()

	ext := strings.ToLower(filepath.Ext(filePath))
	isJSON := ext == ".json" || forceJSON

	return r.loadDocumentsFromReader(file, templateData, isJSON)
}

func (r *Ranker) loadDocumentsFromReader(reader io.Reader, templateData string, isJSON bool) ([]document, error) {
	// Template parsing
	var tmpl *template.Template
	if templateData != "" {
		if templateData[0] == '@' {
			content, err := os.ReadFile(templateData[1:])
			if err != nil {
				return nil, fmt.Errorf("failed to read template file %s: %w", templateData[1:], err)
			}
			templateData = string(content)
		}
		var err error
		if tmpl, err = template.New("siftrank-item-template").Parse(templateData); err != nil {
			return nil, fmt.Errorf("failed to parse template: %w", err)
		}
	}

	if isJSON {
		return r.loadJSONDocuments(reader, tmpl)
	}
	return r.loadTextDocuments(reader, tmpl)
}

func (r *Ranker) loadTextDocuments(reader io.Reader, tmpl *template.Template) ([]document, error) {
	content, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to read content: %w", err)
	}

	var documents []document
	lines := strings.Split(string(content), "\n")

	for i, line := range lines {
		line = strings.TrimRight(line, "\r") // Handle Windows line endings
		if line == "" {
			continue // Skip empty lines
		}

		if tmpl != nil {
			var tmplData bytes.Buffer
			if err := tmpl.Execute(&tmplData, map[string]string{"Data": line}); err != nil {
				return nil, fmt.Errorf("failed to execute template on line: %w", err)
			}
			line = tmplData.String()
		}

		id := ShortDeterministicID(line, idLen)
		documents = append(documents, document{
			ID:         id,
			Document:   nil,
			Value:      line,
			InputIndex: i,
		})
	}

	return documents, nil
}

func (r *Ranker) loadJSONDocuments(reader io.Reader, tmpl *template.Template) ([]document, error) {
	var data []interface{}
	if err := json.NewDecoder(reader).Decode(&data); err != nil {
		return nil, fmt.Errorf("failed to decode JSON: %w", err)
	}

	var documents []document
	for i, value := range data {
		var valueStr string
		if tmpl != nil {
			var tmplData bytes.Buffer
			if err := tmpl.Execute(&tmplData, value); err != nil {
				return nil, fmt.Errorf("failed to execute template: %w", err)
			}
			valueStr = tmplData.String()
		} else {
			r.cfg.Logger.Warn("using json input without a template, using JSON document as-is")
			jsonValue, err := json.Marshal(value)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal JSON value: %w", err)
			}
			valueStr = string(jsonValue)
		}

		id := ShortDeterministicID(valueStr, idLen)
		documents = append(documents, document{
			ID:         id,
			Document:   value,
			Value:      valueStr,
			InputIndex: i,
		})
	}

	return documents, nil
}

// perform the ranking algorithm on the given documents
func (r *Ranker) rank(documents []document, round int) ([]*RankedDocument, error) {
	r.round = round

	// Track original document count for exposure calculation
	if round == 1 {
		r.originalDocCount = len(documents)
	}

	r.cfg.Logger.Info("Ranking documents", "round", r.round, "count", len(documents))

	// If we've narrowed down to a single document, we're done.
	if len(documents) == 1 {
		return []*RankedDocument{
			{
				Key:        documents[0].ID,
				Value:      documents[0].Value,
				Document:   documents[0].Document,
				Score:      0,   // 0 is guaranteed to be the "highest" score.
				Exposure:   1.0, // 100% exposure (single document)
				Rounds:     round,
				InputIndex: documents[0].InputIndex,
			},
		}, nil
	}

	// Downstream ranking gets unhappy if we try to rank more documents than we
	// have.
	if r.cfg.BatchSize > len(documents) {
		r.cfg.BatchSize = len(documents)
	}

	r.numBatches = len(documents) / r.cfg.BatchSize

	// Process the documents and get the sorted results.
	results, err := r.shuffleBatchRank(documents)
	if err != nil {
		return nil, err
	}

	// Determine cutoff for refinement
	var mid int

	if r.cfg.EnableConvergence {
		// Convergence mode: use elbow cutoff
		if r.elbowCutoff > 0 && r.elbowCutoff < len(results) {
			mid = r.elbowCutoff
			r.cfg.Logger.Debug("Using elbow cutoff for refinement",
				"round", r.round,
				"cutoff", mid,
				"total_docs", len(results))
		} else {
			// No valid elbow - don't refine
			r.cfg.Logger.Debug("No valid elbow cutoff, stopping refinement",
				"round", r.round,
				"total_docs", len(results),
				"elbow_cutoff", r.elbowCutoff)
			return results, nil
		}
	} else {
		// Non-convergence mode: use ratio
		if r.cfg.RefinementRatio == 0 {
			return results, nil
		}
		mid = int(float64(len(results)) * r.cfg.RefinementRatio)
		r.cfg.Logger.Debug("Using ratio cutoff for refinement",
			"round", r.round,
			"cutoff", mid,
			"ratio", r.cfg.RefinementRatio,
			"total_docs", len(results))
	}

	// Ensure we have at least 2 documents for meaningful ranking
	if mid < 2 {
		return results, nil
	}

	topPortion := results[:mid]
	bottomPortion := results[mid:]

	// If we haven't reduced the number of documents (as may eventually happen
	// for a ratio above 0.5), we're done.
	if len(topPortion) == len(documents) {
		return results, nil
	}

	r.cfg.Logger.Debug("Top items being sent back into recursion:")
	for i, doc := range topPortion {
		r.cfg.Logger.Debug("Recursive item", "rank", i+1, "id", doc.Key, "score", doc.Score, "value", doc.Value)
	}

	var topPortionDocs []document
	for _, result := range topPortion {
		topPortionDocs = append(topPortionDocs, document{ID: result.Key, Value: result.Value, Document: result.Document, InputIndex: result.InputIndex})
	}

	refinedTopPortion, err := r.rank(topPortionDocs, round+1)
	if err != nil {
		return nil, err
	}

	// Adjust scores by recursion depth; this serves as an inverted weight so
	// that later rounds are guaranteed to sit higher in the final list.
	for i := range refinedTopPortion {
		refinedTopPortion[i].Score /= float64(2 * round)
	}

	// Combine the refined top portion with the unrefined bottom portion.
	finalResults := append(refinedTopPortion, bottomPortion...)

	return finalResults, nil
}

func (r *Ranker) summarizeRelevance(docID string, docValue string, snippets []string) (*RelevanceProsCons, Usage, error) {
	// Skip if no snippets or dry run mode
	if len(snippets) == 0 || r.cfg.DryRun {
		return nil, Usage{}, nil
	}

	// Build prompt with numbered snippets
	snippetsList := ""
	for i, snippet := range snippets {
		snippetsList += fmt.Sprintf("%d. %s\n", i+1, snippet)
	}

	prompt := fmt.Sprintf(`Below are %d relevance snippets from different comparisons of the document "%s". These snippets come from various trials where this document was compared against different sets of documents.

Your task: Analyze these snippets and extract two things:
1. PROS: Qualities of this document that make it RELEVANT to the user's ranking criteria/prompt. Focus strictly on relevance, not whether these qualities are inherently "good" or "bad". (2-4 sentences)
2. CONS: Qualities of this document that make it NOT RELEVANT to the user's ranking criteria/prompt. Focus strictly on lack of relevance, not whether these qualities are inherently "good" or "bad". (2-4 sentences)

CRITICAL: Do not confuse "positive/negative qualities" with "relevant/not relevant". For example, if the prompt asks to "find security vulnerabilities", then vulnerabilities are RELEVANT (pros) even though they are bad, and secure code practices are NOT RELEVANT (cons) even though they are good.

Either pros or cons can be empty if there's nothing to say. For top-performing documents, cons might be empty. For lower-performing documents, pros might be minimal.

Critical style rules:
- DO NOT start sentences with "The document emphasizes", "The document focuses on", "This document", "It emphasizes", "It focuses", "It highlights"
- DO NOT end with "Overall," or "Overall, it" - just stop when you're done
- DO NOT use "While it", "Although it", "However," at the start of every other sentence
- DO NOT use formal academic language - write like you're taking quick notes
- DO NOT use absolute position language like "ranked highest", "ranked lowest", "ranked second"
- DO vary sentence structure - mix short and longer sentences, start sentences differently
- Just capture what's notable: what does it reference? what concepts appear? what stands out?

Write naturally and directly. Imagine explaining to a colleague in conversation.

Relevance snippets:
%s

Provide your response in JSON format with two keys: 'pros' and 'cons'. Each value should be a string (or empty string if nothing to say).
Example: {"pros": "Strong connection to X. References Y explicitly.", "cons": "Lacks specificity compared to documents with Z."}`, len(snippets), docValue, snippetsList)

	// Generate schema for RelevanceProsCons
	schema := generateSchema[RelevanceProsCons]()

	// Call provider with options
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	opts := &CompletionOptions{
		Schema: schema,
	}

	rawResponse, err := r.provider.Complete(ctx, prompt, opts)
	if err != nil {
		return nil, opts.Usage, fmt.Errorf("provider call failed: %w", err)
	}

	// Extract JSON from response
	jsonResponse, err := extractJSON(rawResponse)
	if err != nil {
		return nil, opts.Usage, fmt.Errorf("failed to extract JSON: %w", err)
	}

	// Parse response
	var result RelevanceProsCons
	if err := json.Unmarshal([]byte(jsonResponse), &result); err != nil {
		return nil, opts.Usage, fmt.Errorf("failed to parse relevance response: %w", err)
	}

	return &result, opts.Usage, nil
}

func (r *Ranker) recordTrialState(trialNum int, trialsCompleted int, scores map[string][]float64, documents []document) error {
	// Early exit if neither feature is enabled
	if r.traceFile == nil && !r.cfg.Watch {
		return nil
	}

	// Calculate current rankings from accumulated scores
	var rankings []traceDocument
	for id, scoreList := range scores {
		var sum float64
		for _, score := range scoreList {
			sum += score
		}
		avgScore := sum / float64(len(scoreList))

		// Find the document value
		var value string
		for _, doc := range documents {
			if doc.ID == id {
				value = doc.Value
				break
			}
		}

		rankings = append(rankings, traceDocument{
			ID:    id,
			Value: value,
			Score: avgScore,
		})
	}

	// Sort by score (lower is better)
	sort.Slice(rankings, func(i, j int) bool {
		return rankings[i].Score < rankings[j].Score
	})

	// Build trace line
	trace := traceLine{
		Round:             r.round,
		Trial:             trialNum,
		TrialsCompleted:   trialsCompleted,
		TrialsRemaining:   r.cfg.NumTrials - trialsCompleted,
		TotalInputTokens:  r.totalUsage.InputTokens,
		TotalOutputTokens: r.totalUsage.OutputTokens,
		Rankings:          rankings,
	}

	// Add convergence info if enabled
	if r.cfg.EnableConvergence {
		r.mu.Lock()
		if len(r.elbowPositions) > 0 {
			lastElbow := r.elbowPositions[len(r.elbowPositions)-1]
			if lastElbow >= 0 {
				trace.ElbowPosition = &lastElbow
			}

			// Calculate stability count using shared logic
			stableCount, _ := r.countStableElbows(len(rankings))
			if stableCount > 0 {
				trace.StableTrialsCount = stableCount
			}
		}
		r.mu.Unlock()
	}

	// File writing - only if trace enabled
	if r.traceFile != nil {
		data, err := json.Marshal(trace)
		if err != nil {
			return fmt.Errorf("failed to marshal trace line: %w", err)
		}

		if _, err := r.traceFile.Write(append(data, '\n')); err != nil {
			return fmt.Errorf("failed to write trace line: %w", err)
		}

		// Flush to disk immediately
		if err := r.traceFile.Sync(); err != nil {
			return fmt.Errorf("failed to sync trace file: %w", err)
		}
	}

	// Visualization - only if watch enabled
	if r.cfg.Watch && r.screen != nil {
		r.renderVisualization(rankings, r.round, trialNum)
	}

	return nil
}

// writeString writes a string to the screen at the given position with the given style
func (r *Ranker) writeString(screen tcell.Screen, x, y int, s string, style tcell.Style) {
	for i, ch := range s {
		screen.SetContent(x+i, y, ch, nil, style)
	}
}

// renderVisualization routes to the appropriate rendering function based on configuration
func (r *Ranker) renderVisualization(rankings []traceDocument, round, trial int) {
	if r.screen == nil {
		return
	}

	screen, ok := r.screen.(tcell.Screen)
	if !ok {
		return
	}

	// Route to appropriate rendering function
	if r.cfg.NoMinimap {
		r.renderFullWidth(screen, rankings, round, trial)
	} else {
		r.renderWithMinimap(screen, rankings, round, trial)
	}
}

// renderFullWidth renders the visualization using the full terminal width
func (r *Ranker) renderFullWidth(screen tcell.Screen, rankings []traceDocument, round, trial int) {
	screen.Clear()
	width, height := screen.Size()

	// Render using full width
	r.renderMainDisplay(screen, rankings, round, trial, 0, width, height)

	screen.Show()
}

// renderWithMinimap renders a split-screen view with main display and minimap
func (r *Ranker) renderWithMinimap(screen tcell.Screen, rankings []traceDocument, round, trial int) {
	screen.Clear()
	width, height := screen.Size()

	// Check if terminal is too narrow for minimap
	if width < 30 {
		// Fall back to full-width if too narrow
		r.renderMainDisplay(screen, rankings, round, trial, 0, width, height)
		screen.Show()
		return
	}

	// Calculate layout: 80% main, 20% minimap
	mainWidth := int(float64(width) * 0.8)
	minimapStart := mainWidth + 1
	minimapWidth := width - minimapStart

	// Draw main display (left side)
	r.renderMainDisplay(screen, rankings, round, trial, 0, mainWidth, height)

	// Draw vertical separator
	separatorStyle := tcell.StyleDefault.Foreground(tcell.ColorGray)
	for y := 0; y < height; y++ {
		screen.SetContent(mainWidth, y, '│', nil, separatorStyle)
	}

	// Draw minimap (right side)
	r.renderMinimap(screen, rankings, round, minimapStart, minimapWidth, height)

	screen.Show()
}

// renderMinimap renders a condensed overview of all rankings
func (r *Ranker) renderMinimap(screen tcell.Screen, rankings []traceDocument, round, startX, width, height int) {
	if width < 5 {
		return // Not enough space
	}

	// Header
	header := fmt.Sprintf("Map:%d", len(rankings))
	r.writeString(screen, startX, 0, header, tcell.StyleDefault.Bold(true))

	// Available height for items (reserve header + margins)
	displayHeight := height - 3
	if displayHeight < 1 {
		return
	}

	totalItems := len(rankings)
	if totalItems == 0 {
		return
	}

	// Calculate max score for normalization
	maxScore := 0.0
	if len(rankings) > 0 {
		maxScore = rankings[len(rankings)-1].Score
	}

	// Determine compression ratio
	itemsPerRow := 1.0
	if totalItems > displayHeight {
		itemsPerRow = float64(totalItems) / float64(displayHeight)
	}

	// Find elbow position in the rankings
	elbowIndex := -1
	r.mu.Lock()
	if r.cfg.EnableConvergence && len(r.elbowPositions) > 0 {
		elbowIndex = r.elbowPositions[len(r.elbowPositions)-1]
	}
	r.mu.Unlock()

	// Render each row
	for row := 0; row < displayHeight; row++ {
		y := row + 3 // Align with main display (which starts data at row 3)
		if y >= height {
			break
		}

		// Calculate which items belong to this row
		startIdx := int(float64(row) * itemsPerRow)
		endIdx := int(float64(row+1) * itemsPerRow)
		if endIdx > totalItems {
			endIdx = totalItems
		}
		if startIdx >= totalItems {
			break
		}

		// Calculate average score for this bucket
		var sumScore float64
		for i := startIdx; i < endIdx; i++ {
			sumScore += rankings[i].Score
		}
		avgScore := sumScore / float64(endIdx-startIdx)

		// Calculate bar length (inverse of normalized score)
		barLength := width - 1
		if maxScore > 0 {
			barLength = int((1.0 - avgScore/maxScore) * float64(width-1))
		}
		if barLength < 0 {
			barLength = 0
		}
		if barLength > width-1 {
			barLength = width - 1
		}

		// Check if this row contains the elbow
		containsElbow := false
		if elbowIndex >= startIdx && elbowIndex < endIdx {
			containsElbow = true
		}

		// Determine color
		style := tcell.StyleDefault.Foreground(tcell.ColorWhite)
		if containsElbow {
			style = tcell.StyleDefault.Foreground(tcell.ColorRed)
		}

		// Draw bar
		for x := 0; x < barLength && x < width-1; x++ {
			screen.SetContent(startX+x, y, '█', nil, style)
		}
	}
}

// renderMainDisplay renders the main detailed ranking display
func (r *Ranker) renderMainDisplay(screen tcell.Screen, rankings []traceDocument, round, trial int, startX, maxWidth, maxHeight int) {
	// Help text
	help := "Press Ctrl+C, Esc, or 'q' to quit"
	r.writeString(screen, startX, 0, help, tcell.StyleDefault.Foreground(tcell.ColorDarkGray))

	// Header
	header := fmt.Sprintf("Round %d | Trial %d | Items: %d", round, trial, len(rankings))
	r.writeString(screen, startX, 1, header, tcell.StyleDefault.Bold(true))

	// Calculate max score for normalization
	maxScore := 0.0
	if len(rankings) > 0 {
		maxScore = rankings[len(rankings)-1].Score
	}

	// Rankings (one per line, starting at row 3)
	for i, doc := range rankings {
		row := i + 3
		if row >= maxHeight {
			break // Don't render beyond screen height
		}

		// Calculate bar length proportional to score (invert so lower score = longer bar)
		barLength := maxWidth - 10 // Reserve space for score display
		if maxScore > 0 {
			barLength = int((1.0 - doc.Score/maxScore) * float64(maxWidth-10))
		}
		if barLength < 0 {
			barLength = 0
		}

		// Styles
		whiteStyle := tcell.StyleDefault.Foreground(tcell.ColorWhite)
		grayStyle := tcell.StyleDefault.Foreground(tcell.ColorGray)

		// Draw document text as the bar
		x := startX
		for _, ch := range doc.Value {
			if x-startX >= maxWidth-10 {
				break // Leave room for score
			}

			// White for bar portion, gray for overflow
			style := whiteStyle
			if x-startX >= barLength {
				style = grayStyle
			}
			screen.SetContent(x, row, ch, nil, style)
			x++
		}

		// Fill remaining bar space with '+' if text is shorter than bar
		for x-startX < barLength && x-startX < maxWidth-10 {
			screen.SetContent(x, row, '+', nil, whiteStyle)
			x++
		}

		// Show score at the end
		scoreLabel := fmt.Sprintf(" %.2f", doc.Score)
		r.writeString(screen, startX+maxWidth-9, row, scoreLabel, tcell.StyleDefault.Foreground(tcell.ColorYellow))
	}
}

func (r *Ranker) logFromApiCall(trialNum, batchNum int, message string, args ...interface{}) {
	formattedMessage := fmt.Sprintf(message, args...)
	r.cfg.Logger.Debug(formattedMessage, "round", r.round, "trial", trialNum, "total_trials", r.cfg.NumTrials, "batch", batchNum, "total_batches", r.numBatches)
}

func (r *Ranker) shuffleBatchRank(documents []document) ([]*RankedDocument, error) {
	// Reset convergence state for this recursion level (round)
	r.mu.Lock()
	r.converged = false
	r.elbowPositions = nil // Also clear elbow history
	r.rankingOrders = nil  // Clear ranking order history
	r.elbowCutoff = -1     // Reset cutoff
	r.mu.Unlock()

	// Shared scores for convergence detection and final ranking (all trials combined)
	scores := make(map[string][]float64)
	var scoresMutex sync.Mutex

	// Per-trial scores for building cumulative trace snapshots
	// Structure: trialScores[trialNumber][documentID] = []float64{scores from that trial}
	trialScores := make(map[int]map[string][]float64)
	var trialScoresMutex sync.Mutex

	type workItem struct {
		trialNum int
		batchNum int
		batch    []document
	}

	type batchResult struct {
		rankedDocs  []rankedDocument
		usage       Usage // Tokens for this batch (sum of all calls/retries)
		numCalls    int   // Number of LLM calls made for this batch
		err         error
		trialNumber int
		batchNumber int
	}

	// Create cancellable context for early stopping
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel() // Ensure cleanup

	// Channel for work items (sized for all batches across all trials)
	workQueue := make(chan workItem, r.numBatches*r.cfg.NumTrials)

	// Channel for results
	resultsChan := make(chan batchResult, r.cfg.Concurrency)

	var firstTrialRemainderItems []document

	// Load work queue depth-first (all of trial 1, then all of trial 2, etc.)
	for trialNum := 1; trialNum <= r.cfg.NumTrials; trialNum++ {
		// Shuffle documents for this trial
		shuffledDocs := make([]document, len(documents))
		copy(shuffledDocs, documents)
		r.rng.Shuffle(len(shuffledDocs), func(i, j int) {
			shuffledDocs[i], shuffledDocs[j] = shuffledDocs[j], shuffledDocs[i]
		})

		// Ensure remainder items from the first trial are not in the remainder
		// range in the second trial
		if trialNum == 2 && len(firstTrialRemainderItems) > 0 {
			for {
				remainderStart := r.numBatches * r.cfg.BatchSize
				remainderItems := shuffledDocs[remainderStart:]
				conflictFound := false
				for _, item := range remainderItems {
					for _, firstTrialItem := range firstTrialRemainderItems {
						if item.ID == firstTrialItem.ID {
							r.cfg.Logger.Debug("Conflicting remainder item found", "current", item, "first_trial", firstTrialItem)
							conflictFound = true
							break
						}
					}
					if conflictFound {
						break
					}
				}
				if !conflictFound {
					break
				}
				// Re-shuffle if conflict found
				r.rng.Shuffle(len(shuffledDocs), func(i, j int) {
					shuffledDocs[i], shuffledDocs[j] = shuffledDocs[j], shuffledDocs[i]
				})
			}
		}

		// Save remainder items from the first trial
		if trialNum == 1 {
			remainderStart := r.numBatches * r.cfg.BatchSize
			if remainderStart < len(shuffledDocs) {
				firstTrialRemainderItems = make([]document, len(shuffledDocs[remainderStart:]))
				copy(firstTrialRemainderItems, shuffledDocs[remainderStart:])
				r.cfg.Logger.Debug("First trial remainder items", "items", firstTrialRemainderItems)
			}
		}

		// Queue all batches for this trial
		for batchNum := 0; batchNum < r.numBatches; batchNum++ {
			batch := shuffledDocs[batchNum*r.cfg.BatchSize : (batchNum+1)*r.cfg.BatchSize]
			workQueue <- workItem{
				trialNum: trialNum,
				batchNum: batchNum + 1, // 1-indexed for logging
				batch:    batch,
			}
		}
	}
	close(workQueue) // No more work will be added

	// Launch worker pool
	var workersWg sync.WaitGroup
	for i := 0; i < r.cfg.Concurrency; i++ {
		workersWg.Add(1)
		go func() {
			defer workersWg.Done()

			for work := range workQueue {
				// Check if we should stop (context cancelled due to convergence)
				select {
				case <-ctx.Done():
					r.cfg.Logger.Debug("Worker stopping due to convergence",
						"trial", work.trialNum, "batch", work.batchNum)
					continue // Skip this work item
				default:
					// Continue processing
				}

				// Acquire semaphore
				r.semaphore <- struct{}{}

				// Process batch
				rankedBatch, numCalls, usage, err := r.rankDocs(ctx, work.batch, work.trialNum, work.batchNum)

				// Release semaphore
				<-r.semaphore

				// Send result
				resultsChan <- batchResult{
					rankedDocs:  rankedBatch,
					usage:       usage,
					numCalls:    numCalls,
					err:         err,
					trialNumber: work.trialNum,
					batchNumber: work.batchNum,
				}
			}
		}()
	}

	// Close results channel when all workers are done
	go func() {
		workersWg.Wait()
		close(resultsChan)
	}()

	// Track trial completion and stats
	completedBatches := make(map[int]int) // trialNum -> count of completed batches
	completedTrials := make(map[int]bool) // trialNum -> true if all batches completed

	// Track per-trial stats
	type trialStats struct {
		numBatches int
		numCalls   int
		usage      Usage
	}
	trialStatsMap := make(map[int]*trialStats) // trial number -> stats

	// Track fatal errors that should propagate to callers
	var fatalErr error

	// Collect results
	for result := range resultsChan {
		if result.err != nil {
			// Skip logging if context was cancelled (intentional due to convergence)
			if errors.Is(result.err, context.Canceled) {
				continue
			}
			r.cfg.Logger.Error("Error in batch processing", "error", result.err)
			// Store the first fatal error to return to caller
			if fatalErr == nil {
				fatalErr = result.err
			}
			continue
		}

		// Thread-safe update of shared scores (for convergence detection and final ranking)
		scoresMutex.Lock()
		for _, rankedDoc := range result.rankedDocs {
			scores[rankedDoc.Document.ID] = append(scores[rankedDoc.Document.ID], rankedDoc.Score)
		}
		scoresMutex.Unlock()

		// Track scores per trial for cumulative trace snapshots
		trialScoresMutex.Lock()
		if trialScores[result.trialNumber] == nil {
			trialScores[result.trialNumber] = make(map[string][]float64)
		}
		for _, rankedDoc := range result.rankedDocs {
			trialScores[result.trialNumber][rankedDoc.Document.ID] = append(
				trialScores[result.trialNumber][rankedDoc.Document.ID],
				rankedDoc.Score,
			)
		}
		trialScoresMutex.Unlock()

		// Track comparisons globally (across all rounds/trials)
		r.mu.Lock()
		for _, rankedDoc := range result.rankedDocs {
			// Track which documents this was compared against
			if r.comparedAgainst[rankedDoc.Document.ID] == nil {
				r.comparedAgainst[rankedDoc.Document.ID] = make(map[string]bool)
			}
			// Add all OTHER documents from this batch to the compared-against set
			for _, otherDoc := range result.rankedDocs {
				if otherDoc.Document.ID != rankedDoc.Document.ID {
					r.comparedAgainst[rankedDoc.Document.ID][otherDoc.Document.ID] = true
				}
			}
		}
		r.mu.Unlock()

		// Log batch completion at debug level
		r.cfg.Logger.Debug("Batch completed",
			"round", r.round,
			"trial", result.trialNumber,
			"batch", result.batchNumber,
			"num_calls", result.numCalls,
			"input_tokens", result.usage.InputTokens,
			"output_tokens", result.usage.OutputTokens)

		// Track stats per trial
		if trialStatsMap[result.trialNumber] == nil {
			trialStatsMap[result.trialNumber] = &trialStats{}
		}
		stats := trialStatsMap[result.trialNumber]
		stats.numBatches++
		stats.numCalls += result.numCalls
		stats.usage.Add(result.usage)

		// Track trial completion
		completedBatches[result.trialNumber]++

		// Check if this trial just completed
		if completedBatches[result.trialNumber] == r.numBatches {
			// Mark this trial as fully completed
			completedTrials[result.trialNumber] = true
			completedTrialsCount := len(completedTrials)

			r.cfg.Logger.Info("Trial completed",
				"round", r.round,
				"trial", completedTrialsCount,
				"num_batches", stats.numBatches,
				"num_calls", stats.numCalls,
				"input_tokens", stats.usage.InputTokens,
				"output_tokens", stats.usage.OutputTokens)

			// Update running totals immediately after trial completion
			r.mu.Lock()
			r.totalUsage.Add(stats.usage)
			r.totalCalls += stats.numCalls
			r.totalBatches += stats.numBatches
			r.mu.Unlock()

			// Check for convergence (this adds the current trial's elbow to the array)
			// Note: hasConverged uses the shared 'scores' map (all trials) which is correct
			converged := r.hasConverged(scores, result.trialNumber)

			// Build cumulative scores from ALL trials that have fully completed
			// (regardless of their trial numbers - we care about completion order, not launch order)
			trialScoresMutex.Lock()
			cumulativeScores := make(map[string][]float64)
			for trialNum := range completedTrials {
				if trialData, exists := trialScores[trialNum]; exists {
					for docID, scoreList := range trialData {
						cumulativeScores[docID] = append(cumulativeScores[docID], scoreList...)
					}
				}
			}
			trialScoresMutex.Unlock()

			// Record trial state with cumulative scores from trials 1..N only
			if err := r.recordTrialState(completedTrialsCount, completedTrialsCount, cumulativeScores, documents); err != nil {
				r.cfg.Logger.Error("Failed to record trial state", "error", err)
			}

			// Signal workers to stop if converged
			if converged {
				// No need to log here - hasConverged() already logged if it's the first detection
				cancel() // Signal all workers to stop processing new work
			}
		}
	}

	// Accumulate round totals from all trials
	var roundUsage Usage
	var roundCalls int
	var roundBatches int
	completedTrialsCount := len(completedTrials) // Only trials that actually completed all batches

	for _, stats := range trialStatsMap {
		roundUsage.Add(stats.usage)
		roundCalls += stats.numCalls
		roundBatches += stats.numBatches
	}

	r.cfg.Logger.Info("Round completed",
		"round", r.round,
		"num_trials", completedTrialsCount,
		"num_batches", roundBatches,
		"num_calls", roundCalls,
		"input_tokens", roundUsage.InputTokens,
		"output_tokens", roundUsage.OutputTokens)

	// Update round-level totals (usage/calls/batches already updated per-trial)
	r.mu.Lock()
	r.totalTrials += completedTrialsCount
	r.totalRounds++ // Increment on each round
	r.mu.Unlock()

	// Calculate average scores
	finalScores := make(map[string]float64)

	for id, scoreList := range scores {
		var sum float64
		for _, score := range scoreList {
			sum += score
		}
		finalScores[id] = sum / float64(len(scoreList))
	}

	var results []*RankedDocument
	for id, score := range finalScores {
		for _, doc := range documents {
			if doc.ID == id {
				results = append(results, &RankedDocument{
					Key:        id,
					Value:      doc.Value,
					Document:   doc.Document,
					Score:      score,
					Exposure:   0.0, // Will be calculated at the end in RankFromFile
					Rounds:     r.round,
					InputIndex: doc.InputIndex,
				})
				break
			}
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score < results[j].Score
	})

	// Set elbow cutoff for refinement
	r.setElbowCutoff(len(results))

	// Return fatal error if any batch failed with unrecoverable error
	if fatalErr != nil {
		return nil, fatalErr
	}

	return results, nil
}

// perpendicularDistance calculates perpendicular distance from point to line
func perpendicularDistance(x0, y0, x1, y1, x2, y2 float64) float64 {
	// Line from (x1, y1) to (x2, y2)
	// Point at (x0, y0)
	// Formula: |ax + by + c| / sqrt(a^2 + b^2)
	// where line is ax + by + c = 0

	numerator := math.Abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
	denominator := math.Sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1))

	if denominator == 0 {
		return 0
	}

	return numerator / denominator
}

// findElbowPerpendicular returns the index of the elbow using perpendicular distance method.
// Returns -1 if elbow cannot be determined (e.g., too few documents).
func findElbowPerpendicular(rankedDocs []RankedDocument) int {
	n := len(rankedDocs)

	// Need at least 3 documents to find an elbow
	if n < 3 {
		return -1
	}

	firstScore := rankedDocs[0].Score
	lastScore := rankedDocs[n-1].Score

	// If scores are identical (flat line), no elbow exists
	if firstScore == lastScore {
		return -1
	}

	maxDist := 0.0
	elbowIdx := -1

	// Check each document except first and last
	for i := 1; i < n-1; i++ {
		dist := perpendicularDistance(
			float64(i),
			rankedDocs[i].Score,
			0.0,
			firstScore,
			float64(n-1),
			lastScore,
		)

		if dist > maxDist {
			maxDist = dist
			elbowIdx = i
		}
	}

	return elbowIdx
}

// findElbow dispatches to the appropriate elbow detection method based on the method name.
// Valid methods: ElbowMethodCurvature (default), ElbowMethodPerpendicular.
func findElbow(rankedDocs []RankedDocument, method ElbowMethod) int {
	switch method {
	case ElbowMethodPerpendicular:
		return findElbowPerpendicular(rankedDocs)
	case ElbowMethodCurvature, "":
		return findElbowCurvature(rankedDocs)
	default:
		// Shouldn't happen if config validation is correct, but default to curvature
		return findElbowCurvature(rankedDocs)
	}
}

// gaussianSmooth applies 1D Gaussian smoothing to a slice of values.
// sigma controls the width of the Gaussian kernel.
func gaussianSmooth(data []float64, sigma float64) []float64 {
	n := len(data)
	if n == 0 {
		return data
	}

	// Kernel radius: typically 3*sigma is sufficient to capture 99.7% of the Gaussian
	radius := int(math.Ceil(3 * sigma))
	if radius < 1 {
		radius = 1
	}

	// Build Gaussian kernel
	kernelSize := 2*radius + 1
	kernel := make([]float64, kernelSize)
	var kernelSum float64
	for i := 0; i < kernelSize; i++ {
		x := float64(i - radius)
		kernel[i] = math.Exp(-(x * x) / (2 * sigma * sigma))
		kernelSum += kernel[i]
	}
	// Normalize kernel
	for i := range kernel {
		kernel[i] /= kernelSum
	}

	// Apply convolution with boundary handling (extend edge values)
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		var sum float64
		for k := 0; k < kernelSize; k++ {
			// Map kernel index to data index with boundary clamping
			dataIdx := i + (k - radius)
			if dataIdx < 0 {
				dataIdx = 0
			} else if dataIdx >= n {
				dataIdx = n - 1
			}
			sum += data[dataIdx] * kernel[k]
		}
		result[i] = sum
	}

	return result
}

// gradient calculates the numerical gradient (first derivative) of a slice.
// Uses central differences for interior points and forward/backward differences at edges.
func gradient(data []float64) []float64 {
	n := len(data)
	if n < 2 {
		return make([]float64, n)
	}

	result := make([]float64, n)

	// Forward difference at start
	result[0] = data[1] - data[0]

	// Central differences for interior
	for i := 1; i < n-1; i++ {
		result[i] = (data[i+1] - data[i-1]) / 2.0
	}

	// Backward difference at end
	result[n-1] = data[n-1] - data[n-2]

	return result
}

// findElbowCurvature returns the index of the elbow in a sorted list of ranked documents
// using curvature-based detection. It finds the point of maximum curvature
// (global minimum of 2nd derivative) which represents the transition from
// the steep "interesting" section to the flatter "tail" section.
// Returns -1 if elbow cannot be determined (e.g., too few documents or flat scores).
func findElbowCurvature(rankedDocs []RankedDocument) int {
	n := len(rankedDocs)

	// Need at least 4 documents to find an elbow meaningfully
	if n < 4 {
		return -1
	}

	// Extract scores
	scores := make([]float64, n)
	for i, doc := range rankedDocs {
		scores[i] = doc.Score
	}

	// Check if scores are flat (all identical within epsilon)
	const epsilon = 1e-9
	firstScore := scores[0]
	allFlat := true
	for _, score := range scores {
		if math.Abs(score-firstScore) > epsilon {
			allFlat = false
			break
		}
	}
	if allFlat {
		return -1
	}

	// Calculate sigma: 3% of dataset size, minimum 1.0
	sigma := math.Max(1.0, float64(n)*0.03)

	// Step 1: Smooth the scores
	smoothedScores := gaussianSmooth(scores, sigma)

	// Step 2: Calculate derivatives (cascade approach)
	// 1st derivative from smoothed scores
	firstDeriv := gradient(smoothedScores)
	// Smooth the 1st derivative
	smoothedFirstDeriv := gaussianSmooth(firstDeriv, sigma)
	// 2nd derivative from smoothed 1st derivative
	secondDeriv := gradient(smoothedFirstDeriv)
	// Smooth the 2nd derivative
	smoothedSecondDeriv := gaussianSmooth(secondDeriv, sigma)

	// Step 3: Find the global minimum of the 2nd derivative
	// This is the point of maximum curvature
	minVal := smoothedSecondDeriv[0]
	minIdx := 0
	for i := 1; i < n; i++ {
		if smoothedSecondDeriv[i] < minVal {
			minVal = smoothedSecondDeriv[i]
			minIdx = i
		}
	}

	return minIdx
}

// countStableElbows returns how many consecutive recent elbow positions are within tolerance
// Returns the count (1 to StableTrials) and the tolerance used
func (r *Ranker) countStableElbows(numDocuments int) (int, int) {
	n := len(r.elbowPositions)

	if n < 2 {
		return 0, 0
	}

	// Calculate tolerance in absolute terms (number of positions)
	tolerance := int(r.cfg.ElbowTolerance * float64(numDocuments))
	if tolerance < 1 {
		tolerance = 1 // At minimum, allow 1 position variance
	}

	// Check windows of increasing size to find largest that fits within tolerance
	stableCount := 1 // Current position always counts
	for windowSize := 2; windowSize <= r.cfg.StableTrials && windowSize <= n; windowSize++ {
		// Check if last 'windowSize' positions are all within tolerance
		start := n - windowSize
		minPos := r.elbowPositions[start]
		maxPos := r.elbowPositions[start]
		for i := start + 1; i < n; i++ {
			if r.elbowPositions[i] < minPos {
				minPos = r.elbowPositions[i]
			}
			if r.elbowPositions[i] > maxPos {
				maxPos = r.elbowPositions[i]
			}
		}

		if maxPos-minPos <= tolerance {
			stableCount = windowSize
		}
	}

	return stableCount, tolerance
}

// isElbowStable checks if recent elbow positions are within tolerance
// Returns (isStable, actualTolerance)
func (r *Ranker) isElbowStable(numDocuments int) (bool, int) {
	stableCount, tolerance := r.countStableElbows(numDocuments)
	return stableCount >= r.cfg.StableTrials, tolerance
}

// isRankingStable checks if the full ranking order has stabilized
// Returns (isStable, actualTrialsChecked)
func (r *Ranker) isRankingStable() (bool, int) {
	n := len(r.rankingOrders)

	// Need at least StableTrials to check
	if n < r.cfg.StableTrials {
		return false, 0
	}

	// Get the most recent ranking orders
	recentOrders := r.rankingOrders[n-r.cfg.StableTrials:]

	// Check if all recent orders are identical
	firstOrder := recentOrders[0]
	for i := 1; i < len(recentOrders); i++ {
		if len(recentOrders[i]) != len(firstOrder) {
			return false, len(recentOrders)
		}
		for j := range recentOrders[i] {
			if recentOrders[i][j] != firstOrder[j] {
				return false, len(recentOrders)
			}
		}
	}

	return true, len(recentOrders)
}

// hasConverged checks if the ranking has stabilized across trials
// Returns true if early stopping should occur
func (r *Ranker) hasConverged(scores map[string][]float64, completedTrialNum int) bool {
	// Early stopping disabled
	if !r.cfg.EnableConvergence {
		return false
	}

	// Check if we already detected convergence (quick exit)
	r.mu.Lock()
	alreadyConverged := r.converged
	r.mu.Unlock()

	if alreadyConverged {
		// Convergence already detected by another trial
		// Still return true to respect the cancellation, but don't log again
		return true
	}

	// Not enough trials yet
	if completedTrialNum < r.cfg.MinTrials {
		return false
	}

	// Calculate current rankings based on scores so far
	var currentRankings []RankedDocument
	for id, scoreList := range scores {
		var sum float64
		for _, score := range scoreList {
			sum += score
		}
		avgScore := sum / float64(len(scoreList))

		currentRankings = append(currentRankings, RankedDocument{
			Key:   id,
			Score: avgScore,
		})
	}

	// Sort by score (lower is better)
	sort.Slice(currentRankings, func(i, j int) bool {
		return currentRankings[i].Score < currentRankings[j].Score
	})

	// Store the ranking order for this trial
	rankingOrder := make([]string, len(currentRankings))
	for i, doc := range currentRankings {
		rankingOrder[i] = doc.Key
	}

	r.mu.Lock()
	r.rankingOrders = append(r.rankingOrders, rankingOrder)
	r.mu.Unlock()

	// Check criterion 1: Elbow stabilization
	var elbowStable bool
	var actualTolerance int

	elbowPos := findElbow(currentRankings, r.cfg.ElbowMethod)
	if elbowPos != -1 {
		// Store elbow position
		r.mu.Lock()
		r.elbowPositions = append(r.elbowPositions, elbowPos)
		r.mu.Unlock()

		r.cfg.Logger.Debug("Elbow detected",
			"trial", completedTrialNum,
			"position", elbowPos,
			"total_docs", len(currentRankings))

		elbowStable, actualTolerance = r.isElbowStable(len(currentRankings))
	} else {
		r.cfg.Logger.Debug("No elbow found in trial", "trial", completedTrialNum)
	}

	// Check criterion 2: Ranking order stabilization
	rankingStable, trialsChecked := r.isRankingStable()

	if rankingStable {
		r.cfg.Logger.Debug("Ranking order stable",
			"trial", completedTrialNum,
			"trials_checked", trialsChecked,
			"total_docs", len(currentRankings))
	}

	// Converge if EITHER criterion is met
	stable := elbowStable || rankingStable

	if stable {
		// Acquire lock to set convergence flag
		r.mu.Lock()
		// Double-check we haven't converged in the meantime (race condition)
		if !r.converged {
			r.converged = true

			// Use len(r.rankingOrders) for all convergence types
			// This shows how many trials were evaluated for convergence
			trialsEvaluated := len(r.rankingOrders)

			// Log which criterion triggered
			if elbowStable && rankingStable {
				r.cfg.Logger.Info("Convergence: elbow and ranking both stabilized",
					"round", r.round,
					"trials_evaluated", trialsEvaluated,
					"recent_elbow_positions", r.elbowPositions[len(r.elbowPositions)-r.cfg.StableTrials:],
					"elbow_tolerance", actualTolerance)
			} else if elbowStable {
				r.cfg.Logger.Info("Convergence: elbow position stabilized",
					"round", r.round,
					"trials_evaluated", trialsEvaluated,
					"recent_positions", r.elbowPositions[len(r.elbowPositions)-r.cfg.StableTrials:],
					"tolerance", actualTolerance)
			} else {
				r.cfg.Logger.Info("Convergence: ranking order stabilized",
					"round", r.round,
					"trials_evaluated", trialsEvaluated,
					"trials_checked", trialsChecked,
					"total_docs", len(currentRankings))
			}
		}
		r.mu.Unlock()
	}

	return stable
}

// setElbowCutoff determines the cutoff position for refinement based on elbow detection
func (r *Ranker) setElbowCutoff(numDocuments int) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !r.cfg.EnableConvergence {
		// Convergence disabled - cutoff won't be used
		r.elbowCutoff = -1
		return
	}

	// Check if we converged via ranking stability only (no valid elbows found)
	hasValidElbow := false
	for _, pos := range r.elbowPositions {
		if pos > 0 {
			hasValidElbow = true
			break
		}
	}

	if !hasValidElbow {
		// Either no elbows tracked OR all were -1
		// This means we converged via ranking stability (or didn't converge)
		// Don't refine - the ranking is stable as-is
		r.elbowCutoff = -1
		if r.converged {
			r.cfg.Logger.Debug("Ranking stabilized without elbow, skipping refinement",
				"round", r.round,
				"total_docs", numDocuments)
		}
		return
	}

	if r.converged {
		// Converged via elbow stability - use the last recorded elbow position
		r.elbowCutoff = r.elbowPositions[len(r.elbowPositions)-1]
		r.cfg.Logger.Debug("Using converged elbow position for refinement",
			"round", r.round,
			"cutoff", r.elbowCutoff,
			"total_docs", numDocuments)
	} else {
		// Max trials reached without convergence - use median of recent positions
		recentCount := r.cfg.StableTrials
		if recentCount > len(r.elbowPositions) {
			recentCount = len(r.elbowPositions)
		}
		recentPositions := r.elbowPositions[len(r.elbowPositions)-recentCount:]

		// Calculate median
		sortedPositions := make([]int, len(recentPositions))
		copy(sortedPositions, recentPositions)
		sort.Ints(sortedPositions)

		median := sortedPositions[len(sortedPositions)/2]
		r.elbowCutoff = median

		r.cfg.Logger.Debug("Using median elbow position for refinement",
			"round", r.round,
			"cutoff", r.elbowCutoff,
			"recent_positions", recentPositions,
			"total_docs", numDocuments)
	}
}

func (r *Ranker) logTokenSizes(group []document) {
	r.cfg.Logger.Debug("Logging token sizes for each document in the batch:")
	for _, doc := range group {
		tokenSize := r.estimateTokens([]document{doc}, false)
		valuePreview := doc.Value
		if len(valuePreview) > 100 {
			valuePreview = valuePreview[:100]
		}
		r.cfg.Logger.Debug("Document token size", "id", doc.ID, "token_size", tokenSize, "value_preview", valuePreview)
	}
}

const promptFmt = "id: `%s`\nvalue:\n```\n%s\n```\n\n"

var promptDisclaimer = "\n\nREMEMBER to:\n" +
	"- ALWAYS respond with the short 6-8 character ID of each item found above the value " +
	"(i.e., I'll provide you with `id: <ID>` above the value, and you should respond with that same ID in your response)\n" +
	"— NEVER respond with the actual value!\n" +
	"— NEVER include backticks around IDs in your response!\n" +
	"— NEVER include scores or a written reason/justification in your response!\n" +
	"- Respond in RANKED DESCENDING order, where the FIRST item in your response is the MOST RELEVANT\n" +
	"- Respond in JSON format, with the following schema:\n  {\"docs\": [\"<ID1>\", \"<ID2>\", ...]}\n\n" +
	"Here are the documents to be ranked:\n\n"

const missingIDsStr = "Your last response was missing the following IDs: [%s]. " +
	"Try again—and make ABSOLUTELY SURE to remember to:\n" +
	"- ALWAYS return the IDs and NOT THE VALUES! " +
	"- ALWAYS respond in JSON format as specified! " +
	"- ALWAYS return ALL of the IDs in the list!" +
	"- NEVER include backticks around IDs in your response!" +
	"— NEVER include scores or a written reason/justification in your response!"

const invalidJSONStr = "Your last response was not valid JSON. Try again!"

func (r *Ranker) estimateTokens(group []document, includePrompt bool) int {
	text := ""
	if includePrompt {
		text += r.cfg.InitialPrompt + promptDisclaimer
	}
	for _, doc := range group {
		text += fmt.Sprintf(promptFmt, doc.ID, doc.Value)
	}

	// Check if provider supports token estimation
	if estimator, ok := r.provider.(TokenEstimator); ok {
		return estimator.EstimateTokens(text)
	}

	// Fallback: rough approximation (~4 chars per token)
	return len(text) / 4
}

// extractJSON attempts to extract JSON from various response formats.
// Handles:
// - Plain JSON
// - Markdown code blocks (```json ... ``` or ``` ... ```)
// Returns the extracted JSON string or error if no valid JSON found.
func extractJSON(response string) (string, error) {
	response = strings.TrimSpace(response)

	if response == "" {
		return "", fmt.Errorf("empty response")
	}

	// Try as-is first (most common with structured output)
	var js json.RawMessage
	if err := json.Unmarshal([]byte(response), &js); err == nil {
		return response, nil
	}

	// Try extracting from markdown code blocks
	patterns := []string{
		"```json\\s*\\n([\\s\\S]*?)\\n```",
		"```\\s*\\n([\\s\\S]*?)\\n```",
	}

	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		if matches := re.FindStringSubmatch(response); len(matches) > 1 {
			jsonStr := strings.TrimSpace(matches[1])
			if err := json.Unmarshal([]byte(jsonStr), &js); err == nil {
				return jsonStr, nil
			}
		}
	}

	return "", fmt.Errorf("no valid JSON found in response")
}

func (r *Ranker) rankDocs(ctx context.Context, group []document, trialNumber int, batchNumber int) ([]rankedDocument, int, Usage, error) {
	if r.cfg.DryRun {
		r.cfg.Logger.Debug("Dry run API call")
		// Simulate a ranked response for dry run
		var rankedDocs []rankedDocument
		for i, doc := range group {
			rankedDocs = append(rankedDocs, rankedDocument{
				Document: doc,
				Score:    float64(i + 1), // Simulate scores based on position
			})
		}
		return rankedDocs, 0, Usage{}, nil // Zero calls, zero tokens in dry-run
	}

	var totalUsage Usage
	var numCalls int

	// Get schema once
	schema := r.getResponseSchema()

	// Track previous attempt for feedback
	type previousAttempt struct {
		response string
		problem  string // What went wrong
	}
	var lastAttempt *previousAttempt

	maxRetries := 10
	for attempt := 0; attempt < maxRetries; attempt++ {
		// Check if context cancelled
		if ctx.Err() != nil {
			return nil, numCalls, totalUsage, ctx.Err()
		}

		// Try to create memorable ID mappings for each attempt
		originalToTemp, tempToOriginal, err := createIDMappings(group, r.rng, r.cfg.Logger)
		useMemorableIDs := err == nil && originalToTemp != nil && tempToOriginal != nil

		// Build prompt (business logic)
		prompt := r.cfg.InitialPrompt + promptDisclaimer

		// Track input IDs for validation
		inputIDs := make(map[string]bool)

		if useMemorableIDs {
			// Use memorable IDs in the prompt
			for _, doc := range group {
				tempID := originalToTemp[doc.ID]
				prompt += fmt.Sprintf(promptFmt, tempID, doc.Value)
				inputIDs[tempID] = true
			}
		} else {
			// Fall back to original IDs
			for _, doc := range group {
				prompt += fmt.Sprintf(promptFmt, doc.ID, doc.Value)
				inputIDs[doc.ID] = true
			}
		}

		// Add relevance instructions when enabled and past round 1
		if r.cfg.Relevance && r.round > 1 {
			prompt += "\n\nIMPORTANT: In addition to ranking, you must also provide relevance explanations. For each document, write a brief 1-2 sentence explanation focusing on the specific qualities that make it MORE or LESS relevant to the ranking criteria/prompt. Do not confuse 'good qualities' with 'relevant to prompt' - for example, if ranking by 'find vulnerabilities', vulnerabilities are relevant even though they are bad.\n\n"
			prompt += "Your response must include both:\n"
			prompt += "1. A 'docs' array with the ranked IDs\n"
			prompt += "2. A 'relevance' array with an entry for each document\n\n"
			prompt += "Example format:\n"
			prompt += "{\n"
			prompt += "  \"docs\": [\"id1\", \"id2\", \"id3\"],\n"
			prompt += "  \"relevance\": [\n"
			prompt += "    {\"id\": \"id1\", \"text\": \"This document ranked highest because...\"},\n"
			prompt += "    {\"id\": \"id2\", \"text\": \"This document ranked second because...\"},\n"
			prompt += "    {\"id\": \"id3\", \"text\": \"This document ranked lowest because...\"}\n"
			prompt += "  ]\n"
			prompt += "}\n"
		}

		// Add feedback from previous attempt (SiftRank's business logic - prompt-based feedback)
		if lastAttempt != nil {
			prompt += "\n\n--- PREVIOUS ATTEMPT ---\n"
			prompt += fmt.Sprintf("You previously returned: %s\n", lastAttempt.response)
			prompt += fmt.Sprintf("PROBLEM: %s\n", lastAttempt.problem)
			prompt += "Please provide a corrected response.\n"
			prompt += "--- END PREVIOUS ATTEMPT ---\n"
		}

		// Call provider with options
		opts := &CompletionOptions{
			Schema: schema,
		}

		rawResponse, err := r.provider.Complete(ctx, prompt, opts)

		// Accumulate usage from opts
		numCalls++
		totalUsage.Add(opts.Usage)

		// Log the call
		r.cfg.Logger.Debug("LLM call completed",
			"round", r.round,
			"trial", trialNumber,
			"batch", batchNumber,
			"attempt", attempt+1,
			"input_tokens", opts.Usage.InputTokens,
			"output_tokens", opts.Usage.OutputTokens,
			"reasoning_tokens", opts.Usage.ReasoningTokens,
			"model", opts.ModelUsed,
			"finish_reason", opts.FinishReason)

		if err != nil {
			if attempt == maxRetries-1 {
				return nil, numCalls, totalUsage, err
			}
			r.logFromApiCall(trialNumber, batchNumber,
				"Provider call failed, retrying (attempt %d): %v", attempt+1, err)
			continue
		}

		// Extract JSON from response (handles markdown, etc.)
		jsonResponse, err := extractJSON(rawResponse)
		if err != nil {
			lastAttempt = &previousAttempt{
				response: rawResponse,
				problem:  "Your response was not valid JSON or was not formatted correctly. Please respond with ONLY valid JSON matching the schema, with no markdown formatting or extra text.",
			}

			if attempt == maxRetries-1 {
				return nil, numCalls, totalUsage,
					fmt.Errorf("failed to extract JSON after %d attempts: %w", maxRetries, err)
			}

			r.logFromApiCall(trialNumber, batchNumber,
				"JSON extraction failed, retrying (attempt %d): %v", attempt+1, err)
			continue
		}

		// Parse JSON (business logic)
		var rankedResponse rankedDocumentResponse
		if err := json.Unmarshal([]byte(jsonResponse), &rankedResponse); err != nil {
			lastAttempt = &previousAttempt{
				response: jsonResponse,
				problem:  fmt.Sprintf("Your JSON had a syntax error: %v", err),
			}

			if attempt == maxRetries-1 {
				return nil, numCalls, totalUsage, fmt.Errorf("invalid JSON: %w", err)
			}

			r.logFromApiCall(trialNumber, batchNumber,
				"JSON parse failed, retrying (attempt %d): %v", attempt+1, err)
			continue
		}

		// Validate IDs (business logic)
		// This also fixes case-insensitive matches in place
		missingIDs, err := validateIDs(&rankedResponse, inputIDs)
		if err != nil {
			lastAttempt = &previousAttempt{
				response: jsonResponse,
				problem:  fmt.Sprintf("You're missing these IDs: [%s]. Please include ALL IDs.", strings.Join(missingIDs, ", ")),
			}

			if attempt == maxRetries-1 {
				return nil, numCalls, totalUsage,
					fmt.Errorf("missing IDs after %d attempts: %v", maxRetries, missingIDs)
			}

			r.logFromApiCall(trialNumber, batchNumber,
				"Missing IDs, retrying (attempt %d): %v", attempt+1, missingIDs)
			continue
		}

		// Translate memorable IDs back (business logic)
		if useMemorableIDs {
			translateIDsInResponse(&rankedResponse, tempToOriginal)
		}

		// Success! Build ranked documents (business logic)
		var rankedDocs []rankedDocument
		for i, id := range rankedResponse.Documents {
			for _, doc := range group {
				if doc.ID == id {
					rankedDocs = append(rankedDocs, rankedDocument{
						Document: doc,
						Score:    float64(i + 1), // Score based on position (1 for first, 2 for second, etc.)
					})
					break
				}
			}
		}

		// Store relevance snippets if collected (business logic)
		if r.cfg.Relevance && r.round > 1 && len(rankedResponse.Relevance) > 0 {
			r.mu.Lock()
			for _, docRelevance := range rankedResponse.Relevance {
				if stats, exists := r.allDocStats[docRelevance.ID]; exists {
					stats.relevanceSnippets = append(stats.relevanceSnippets, docRelevance.Text)
				}
			}
			r.mu.Unlock()
		}

		return rankedDocs, numCalls, totalUsage, nil
	}

	return nil, numCalls, totalUsage, fmt.Errorf("failed after %d attempts", maxRetries)
}

// validateIDs updates the rankedResponse in place to fix case-insensitive ID mismatches.
// If any IDs are missing, returns the missing IDs along with an error.
func validateIDs(rankedResponse *rankedDocumentResponse, inputIDs map[string]bool) ([]string, error) {
	// Create a map for case-insensitive ID matching
	inputIDsLower := make(map[string]string)
	for id := range inputIDs {
		inputIDsLower[strings.ToLower(id)] = id
	}

	missingIDs := make(map[string]bool)
	for id := range inputIDs {
		missingIDs[id] = true
	}

	for i, id := range rankedResponse.Documents {
		id = strings.ReplaceAll(id, "`", "")
		lowerID := strings.ToLower(id)
		if correctID, found := inputIDsLower[lowerID]; found {
			if correctID != id {
				// Replace the case-wrong match with the correct ID
				rankedResponse.Documents[i] = correctID
			}
			delete(missingIDs, correctID)
		}
	}

	if len(missingIDs) == 0 {
		return nil, nil
	} else {
		missingIDsKeys := make([]string, 0, len(missingIDs))
		for id := range missingIDs {
			missingIDsKeys = append(missingIDsKeys, id)
		}
		return missingIDsKeys, fmt.Errorf("missing IDs: %s", strings.Join(missingIDsKeys, ", "))
	}
}
