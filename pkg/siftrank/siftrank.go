package siftrank

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"os"
	"os/signal"
	"sort"
	"sync"
	"syscall"
	"time"

	"github.com/gdamore/tcell/v2"
)

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
	Document   interface{}        `json:"document"` // if loading from json file
	Score      float64            `json:"score"`
	Exposure   float64            `json:"exposure"` // percentage of dataset compared against (0.0-1.0)
	Rank       int                `json:"rank"`
	Rounds     int                `json:"rounds"`              // number of rounds participated in
	Relevance  *RelevanceProsCons `json:"relevance,omitempty"` // Only if relevance enabled
	InputIndex int                `json:"input_index"`         // Index in original input (0-based)
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
