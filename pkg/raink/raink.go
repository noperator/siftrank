package raink

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"text/template"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
	"github.com/pkoukk/tiktoken-go"
)

const (
	idLen        = 8
	minBatchSize = 2
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

/*
When deciding whether a value belongs in Config or Ranker structs, consider the following:
- Does this value change during operation? → Ranker if yes, Config if no
- Should users be able to configure this directly? → Config if yes, Ranker if no
- Is this derived from other configuration? → Usually Ranker
- Does this require initialization or cleanup? → Usually Ranker
- Is this part of the public API? → Config if yes, Ranker if no
*/

type Config struct {
	InitialPrompt   string           `json:"initial_prompt"`
	BatchSize       int              `json:"batch_size"`
	NumTrials       int              `json:"num_trials"`
	Concurrency     int              `json:"concurrency"`
	OpenAIModel     openai.ChatModel `json:"openai_model"`
	TokenLimit      int              `json:"token_limit"`
	RefinementRatio float64          `json:"refinement_ratio"`
	OpenAIKey       string           `json:"-"`
	OpenAIAPIURL    string           `json:"-"`
	Encoding        string           `json:"encoding"`
	BatchTokens     int              `json:"batch_tokens"`
	DryRun          bool             `json:"-"`
	Logger          *slog.Logger     `json:"-"`
	LogLevel        slog.Level       `json:"-"` // Defaults to 0 (slog.LevelInfo)

	// Convergence detection
	EnableConvergence bool    `json:"enable_convergence"`
	ElbowTolerance    float64 `json:"elbow_tolerance"`
	StableTrials      int     `json:"stable_trials"`
	MinTrials         int     `json:"min_trials"`
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
	if c.TokenLimit <= 0 {
		return fmt.Errorf("token limit must be greater than 0")
	}
	if c.OpenAIAPIURL == "" && c.OpenAIKey == "" {
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
	return nil
}

type Ranker struct {
	cfg            *Config
	encoding       *tiktoken.Tiktoken
	rng            *rand.Rand
	numBatches     int
	round          int
	semaphore      chan struct{} // Global concurrency limiter
	elbowPositions []int         // Track elbow position after each trial
	mu             sync.Mutex    // Protect elbowPositions and converged
	converged      bool          // Track if convergence already detected
	elbowCutoff    int           // Cutoff position for refinement
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
		})).With("component", "raink")
	}

	encoding, err := tiktoken.GetEncoding(config.Encoding)
	if err != nil {
		return nil, fmt.Errorf("failed to get tiktoken encoding: %w", err)
	}

	return &Ranker{
		cfg:       config,
		encoding:  encoding,
		rng:       rand.New(rand.NewSource(time.Now().UnixNano())),
		semaphore: make(chan struct{}, config.Concurrency),
	}, nil
}

// dynamically adjust batch size to fit within token limits
func (ranker *Ranker) adjustBatchSize(documents []document, samples int) error {
	// Dynamically adjust batch size upfront.
	for {
		valid := true
		var estTotalTokens int
		var numBatches int

		for i := 0; i < samples; i++ {
			ranker.rng.Shuffle(len(documents), func(i, j int) {
				documents[i], documents[j] = documents[j], documents[i]
			})
			numBatches = max(1, len(documents)/ranker.cfg.BatchSize) // Need at least one batch.
			for j := 0; j < numBatches; j++ {
				batch := documents[j*ranker.cfg.BatchSize : (j+1)*min(len(documents), ranker.cfg.BatchSize)] // Don't index more documents than we have.
				estBatchTokens := ranker.estimateTokens(batch, true)
				estTotalTokens += estBatchTokens
				if estBatchTokens > ranker.cfg.TokenLimit {
					ranker.cfg.Logger.Debug("Sample exceeded token threshold - estimated tokens > max limit", "sample", i, "estimated_tokens", estBatchTokens, "max_threshold", ranker.cfg.TokenLimit)
					ranker.logTokenSizes(batch)
					valid = false
					break
				}
			}
			if !valid {
				break
			}
		}

		if valid {
			avgEstTokens := estTotalTokens / (samples * numBatches)
			avgEstPct := float64(avgEstTokens) / float64(ranker.cfg.TokenLimit) * 100
			ranker.cfg.Logger.Debug("Average estimated tokens calculated", "tokens", avgEstTokens, "percentage_of_max", avgEstPct, "max_tokens", ranker.cfg.TokenLimit)
			break
		}
		if ranker.cfg.BatchSize <= minBatchSize {
			return fmt.Errorf("cannot create a valid batch within the token limit")
		}
		ranker.cfg.BatchSize--
		ranker.cfg.Logger.Debug("Decreasing batch size to fit within token limits", "new_size", ranker.cfg.BatchSize)
	}
	return nil
}

type document struct {
	ID       string      `json:"id"`
	Value    string      `json:"value"`    // to be ranked
	Document interface{} `json:"document"` // if loading from json file
}

type rankedDocument struct {
	Document document
	Score    float64
}

type rankedDocumentResponse struct {
	Documents []string `json:"docs" jsonschema_description:"List of ranked document IDs"`
}

type RankedDocument struct {
	Key      string      `json:"key"`
	Value    string      `json:"value"`
	Document interface{} `json:"document"` // if loading from json file
	Score    float64     `json:"score"`
	Exposure int         `json:"exposure"`
	Rank     int         `json:"rank"`
}

func generateSchema[T any]() interface{} {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	var v T
	schema := reflector.Reflect(v)
	return schema
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
	for i, id := range response.Documents {
		if originalID, exists := tempToOriginal[id]; exists {
			response.Documents[i] = originalID
		}
	}
}

var rankedDocumentResponseSchema = generateSchema[rankedDocumentResponse]()

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

// ranks documents loaded from a file with optional template
func (r *Ranker) RankFromFile(filePath string, templateData string, forceJSON bool) ([]RankedDocument, error) {
	documents, err := r.loadDocumentsFromFile(filePath, templateData, forceJSON)
	if err != nil {
		return nil, err
	}

	// check that no document is too large
	for _, doc := range documents {
		tokens := r.estimateTokens([]document{doc}, true)
		if tokens > r.cfg.BatchTokens {
			return nil, fmt.Errorf("document is too large with %d tokens:\n%s", tokens, doc.Value)
		}
	}

	if err := r.adjustBatchSize(documents, 10); err != nil {
		return nil, err
	}

	results := r.rank(documents, 1)

	// Add the rank key to each final result based on its position in the list
	for i := range results {
		results[i].Rank = i + 1
	}

	return results, nil
}

func (r *Ranker) loadDocumentsFromFile(filePath string, templateData string, forceJSON bool) (documents []document, err error) {
	var tmpl *template.Template
	if templateData != "" {
		if templateData[0] == '@' {
			content, err := os.ReadFile(templateData[1:])
			if err != nil {
				return nil, fmt.Errorf("failed to read template file %s: %w", templateData[1:], err)
			}
			templateData = string(content)
		}
		if tmpl, err = template.New("raink-item-template").Parse(templateData); err != nil {
			return nil, fmt.Errorf("failed to parse template: %w", err)
		}
	}

	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open input file %s: %w", filePath, err)
	}
	defer file.Close()

	ext := strings.ToLower(filepath.Ext(filePath))
	if ext == ".json" || forceJSON {
		// parse the file in an opaque array
		var data []interface{}
		if err := json.NewDecoder(file).Decode(&data); err != nil {
			return nil, fmt.Errorf("failed to decode JSON from %s: %w", filePath, err)
		}

		// iterate over the map and create documents
		for _, value := range data {
			var valueStr string
			if tmpl != nil {
				var tmplData bytes.Buffer
				if err := tmpl.Execute(&tmplData, value); err != nil {
					return nil, fmt.Errorf("failed to execute template: %w", err)
				}
				valueStr = tmplData.String()
			} else {
				r.cfg.Logger.Warn("using json input without a template, using JSON document as it is")
				jsonValue, err := json.Marshal(value)
				if err != nil {
					return nil, fmt.Errorf("failed to marshal JSON value: %w", err)
				}
				valueStr = string(jsonValue)
			}

			id := ShortDeterministicID(valueStr, idLen)
			documents = append(documents, document{ID: id, Document: value, Value: valueStr})
		}
	} else {
		// read and interpolate the file line by line
		reader := bufio.NewReader(file)
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if err == io.EOF {
					break
				}
				return nil, fmt.Errorf("failed to read line from %s: %w", filePath, err)
			}
			line = strings.TrimSpace(line)

			if tmpl != nil {
				var tmplData bytes.Buffer
				if err := tmpl.Execute(&tmplData, map[string]string{"Data": line}); err != nil {
					return nil, fmt.Errorf("failed to execute template on line: %w", err)
				}
				line = tmplData.String()
			}

			id := ShortDeterministicID(line, idLen)
			documents = append(documents, document{ID: id, Document: nil, Value: line})
		}
	}

	return documents, nil
}

// perform the ranking algorithm on the given documents
func (r *Ranker) rank(documents []document, round int) []RankedDocument {
	r.round = round

	r.cfg.Logger.Info("Ranking documents", "round", r.round, "count", len(documents))

	// If we've narrowed down to a single document, we're done.
	if len(documents) == 1 {
		return []RankedDocument{
			{
				Key:      documents[0].ID,
				Value:    documents[0].Value,
				Document: documents[0].Document,
				Score:    0, // 0 is guaranteed to be the "highest" score.
				Exposure: 1,
			},
		}
	}

	// Downstream ranking gets unhappy if we try to rank more documents than we
	// have.
	if r.cfg.BatchSize > len(documents) {
		r.cfg.BatchSize = len(documents)
	}

	r.numBatches = len(documents) / r.cfg.BatchSize

	// Process the documents and get the sorted results.
	results := r.shuffleBatchRank(documents)

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
			return results
		}
	} else {
		// Non-convergence mode: use ratio
		if r.cfg.RefinementRatio == 0 {
			return results
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
		return results
	}

	topPortion := results[:mid]
	bottomPortion := results[mid:]

	// If we haven't reduced the number of documents (as may eventually happen
	// for a ratio above 0.5), we're done.
	if len(topPortion) == len(documents) {
		return results
	}

	r.cfg.Logger.Debug("Top items being sent back into recursion:")
	for i, doc := range topPortion {
		r.cfg.Logger.Debug("Recursive item", "rank", i+1, "id", doc.Key, "score", doc.Score, "value", doc.Value)
	}

	var topPortionDocs []document
	for _, result := range topPortion {
		topPortionDocs = append(topPortionDocs, document{ID: result.Key, Value: result.Value, Document: result.Document})
	}

	refinedTopPortion := r.rank(topPortionDocs, round+1)

	// Adjust scores by recursion depth; this serves as an inverted weight so
	// that later rounds are guaranteed to sit higher in the final list.
	for i := range refinedTopPortion {
		refinedTopPortion[i].Score /= float64(2 * round)
	}

	// Combine the refined top portion with the unrefined bottom portion.
	finalResults := append(refinedTopPortion, bottomPortion...)

	return finalResults
}

func (r *Ranker) logFromApiCall(trialNum, batchNum int, message string, args ...interface{}) {
	formattedMessage := fmt.Sprintf(message, args...)
	r.cfg.Logger.Debug(formattedMessage, "round", r.round, "trial", trialNum, "total_trials", r.cfg.NumTrials, "batch", batchNum, "total_batches", r.numBatches)
}

func (r *Ranker) shuffleBatchRank(documents []document) []RankedDocument {
	// Reset convergence state for this recursion level (round)
	r.mu.Lock()
	r.converged = false
	r.elbowPositions = nil // Also clear elbow history
	r.elbowCutoff = -1     // Reset cutoff
	r.mu.Unlock()

	scores := make(map[string][]float64)
	exposureCounts := make(map[string]int)
	var scoresMutex sync.Mutex

	type workItem struct {
		trialNum int
		batchNum int
		batch    []document
	}

	type batchResult struct {
		rankedDocs  []rankedDocument
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
				rankedBatch, err := r.rankDocs(work.batch, work.trialNum, work.batchNum)

				// Release semaphore
				<-r.semaphore

				// Send result
				resultsChan <- batchResult{
					rankedDocs:  rankedBatch,
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

	// Track trial completion
	completedBatches := make(map[int]int) // trialNum -> count of completed batches

	// Collect results
	for result := range resultsChan {
		if result.err != nil {
			r.cfg.Logger.Error("Error in batch processing", "error", result.err)
			continue
		}

		// Thread-safe update of shared maps
		scoresMutex.Lock()
		for _, rankedDoc := range result.rankedDocs {
			scores[rankedDoc.Document.ID] = append(scores[rankedDoc.Document.ID], rankedDoc.Score)
			exposureCounts[rankedDoc.Document.ID]++
		}
		scoresMutex.Unlock()

		// Track trial completion
		completedBatches[result.trialNumber]++

		// Check if this trial just completed
		if completedBatches[result.trialNumber] == r.numBatches {
			r.cfg.Logger.Info("Trial completed", "round", r.round, "trial", result.trialNumber)

			// Check for convergence
			if r.hasConverged(scores, exposureCounts, result.trialNumber) {
				// No need to log here - hasConverged() already logged if it's the first detection
				cancel() // Signal all workers to stop processing new work
			}
		}
	}

	// Calculate average scores
	finalScores := make(map[string]float64)
	for id, scoreList := range scores {
		var sum float64
		for _, score := range scoreList {
			sum += score
		}
		finalScores[id] = sum / float64(len(scoreList))
	}

	var results []RankedDocument
	for id, score := range finalScores {
		for _, doc := range documents {
			if doc.ID == id {
				results = append(results, RankedDocument{
					Key:      id,
					Value:    doc.Value,
					Document: doc.Document,
					Score:    score,
					Exposure: exposureCounts[id],
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

	return results
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

// findElbow returns the index of the elbow in a sorted list of ranked documents
// Returns -1 if elbow cannot be determined (e.g., too few documents)
func findElbow(rankedDocs []RankedDocument) int {
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

// isElbowStable checks if recent elbow positions are within tolerance
func (r *Ranker) isElbowStable(numDocuments int) bool {
	n := len(r.elbowPositions)

	// Need at least StableTrials to check
	if n < r.cfg.StableTrials {
		return false
	}

	// Get the most recent positions
	recentPositions := r.elbowPositions[n-r.cfg.StableTrials:]

	// Calculate tolerance in absolute terms (number of positions)
	tolerance := int(r.cfg.ElbowTolerance * float64(numDocuments))
	if tolerance < 1 {
		tolerance = 1 // At minimum, allow 1 position variance
	}

	// Find min and max of recent positions
	minPos := recentPositions[0]
	maxPos := recentPositions[0]
	for _, pos := range recentPositions[1:] {
		if pos < minPos {
			minPos = pos
		}
		if pos > maxPos {
			maxPos = pos
		}
	}

	// Check if range is within tolerance
	return (maxPos - minPos) <= tolerance
}

// hasConverged checks if the ranking has stabilized across trials
// Returns true if early stopping should occur
func (r *Ranker) hasConverged(scores map[string][]float64, exposureCounts map[string]int, completedTrialNum int) bool {
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

	// Find elbow in current rankings
	elbowPos := findElbow(currentRankings)

	// If no elbow found, cannot converge
	if elbowPos == -1 {
		r.cfg.Logger.Debug("No elbow found in trial", "trial", completedTrialNum)
		return false
	}

	// Store elbow position for this trial
	r.mu.Lock()
	r.elbowPositions = append(r.elbowPositions, elbowPos)
	r.mu.Unlock()

	r.cfg.Logger.Debug("Elbow detected",
		"trial", completedTrialNum,
		"position", elbowPos,
		"total_docs", len(currentRankings))

	// Check if elbow has stabilized
	stable := r.isElbowStable(len(currentRankings))

	if stable {
		// Acquire lock to set convergence flag
		r.mu.Lock()
		// Double-check we haven't converged in the meantime (race condition)
		if !r.converged {
			r.converged = true
			r.cfg.Logger.Info("Elbow position stabilized",
				"round", r.round,
				"trial", completedTrialNum,
				"recent_positions", r.elbowPositions[len(r.elbowPositions)-r.cfg.StableTrials:],
				"tolerance", int(r.cfg.ElbowTolerance*float64(len(currentRankings))))
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
		// Convergence disabled - will use ratio-based approach in rank()
		r.elbowCutoff = -1
		return
	}

	if len(r.elbowPositions) == 0 {
		// No elbow positions tracked (shouldn't happen, but be defensive)
		r.elbowCutoff = -1
		return
	}

	if r.converged {
		// Convergence detected - use the last recorded elbow position
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

	return len(r.encoding.Encode(text, nil, nil))
}

func (r *Ranker) rankDocs(group []document, trialNumber int, batchNumber int) ([]rankedDocument, error) {
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
		return rankedDocs, nil
	}

	maxRetries := 10
	for attempt := 0; attempt < maxRetries; attempt++ {
		// Try to create memorable ID mappings for each attempt
		originalToTemp, tempToOriginal, err := createIDMappings(group, r.rng, r.cfg.Logger)
		useMemorableIDs := err == nil && originalToTemp != nil && tempToOriginal != nil

		prompt := r.cfg.InitialPrompt + promptDisclaimer
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

		var rankedResponse rankedDocumentResponse
		rankedResponse, err = r.callOpenAI(prompt, trialNumber, batchNumber, inputIDs)
		if err != nil {
			if attempt == maxRetries-1 {
				return nil, err
			}
			r.logFromApiCall(trialNumber, batchNumber, "API call failed, retrying with new memorable IDs (attempt %d): %v", attempt+1, err)
			continue
		}

		// Translate temporary IDs back to original IDs if using memorable IDs
		if useMemorableIDs {
			translateIDsInResponse(&rankedResponse, tempToOriginal)
		}

		// Check if we got all expected IDs
		expectedIDs := make(map[string]bool)
		for _, doc := range group {
			expectedIDs[doc.ID] = true
		}
		for _, id := range rankedResponse.Documents {
			delete(expectedIDs, id)
		}

		if len(expectedIDs) > 0 {
			var missingIDs []string
			for id := range expectedIDs {
				missingIDs = append(missingIDs, id)
			}
			if attempt == maxRetries-1 {
				return nil, fmt.Errorf("missing IDs after %d attempts: %v", maxRetries, missingIDs)
			}
			r.logFromApiCall(trialNumber, batchNumber, "Missing IDs, retrying with new memorable IDs (attempt %d): %v", attempt+1, missingIDs)
			continue
		}

		// Success! Assign scores based on position in the ranked list
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

		return rankedDocs, nil
	}

	return nil, fmt.Errorf("failed after %d attempts", maxRetries)
}

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

func (r *Ranker) callOpenAI(prompt string, trialNum int, batchNum int, inputIDs map[string]bool) (rankedDocumentResponse, error) {

	customTransport := &customTransport{Transport: http.DefaultTransport}
	customClient := &http.Client{Transport: customTransport}

	clientOptions := []option.RequestOption{
		option.WithAPIKey(r.cfg.OpenAIKey),
		option.WithHTTPClient(customClient),
		option.WithMaxRetries(5),
	}

	// Add base URL option if specified
	if r.cfg.OpenAIAPIURL != "" {
		// Ensure the URL ends with a trailing slash
		baseURL := r.cfg.OpenAIAPIURL
		if !strings.HasSuffix(baseURL, "/") {
			baseURL += "/"
		}
		clientOptions = append(clientOptions, option.WithBaseURL(baseURL))
	}

	client := openai.NewClient(clientOptions...)

	backoff := time.Second

	conversationHistory := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage(prompt),
	}

	var rankedResponse rankedDocumentResponse
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()

		completion, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Messages: conversationHistory,
			ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
					JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
						Name:        "ranked_document_response",
						Description: openai.String("List of ranked document IDs"),
						Schema:      rankedDocumentResponseSchema,
						Strict:      openai.Bool(true),
					},
				},
			},
			Model: r.cfg.OpenAIModel,
		})
		if err == nil {

			conversationHistory = append(conversationHistory,
				openai.AssistantMessage(completion.Choices[0].Message.Content),
			)

			err = json.Unmarshal([]byte(completion.Choices[0].Message.Content), &rankedResponse)
			if err != nil {
				r.logFromApiCall(trialNum, batchNum, fmt.Sprintf("Error unmarshalling response: %v\n", err))
				conversationHistory = append(conversationHistory,
					openai.UserMessage(invalidJSONStr),
				)
				trimmedContent := strings.TrimSpace(completion.Choices[0].Message.Content)
				r.cfg.Logger.Debug("OpenAI API response", "content", trimmedContent)
				continue
			}

			missingIDs, err := validateIDs(&rankedResponse, inputIDs)
			if err != nil {
				r.logFromApiCall(trialNum, batchNum, fmt.Sprintf("Missing IDs: [%s]", strings.Join(missingIDs, ", ")))
				conversationHistory = append(conversationHistory,
					openai.UserMessage(fmt.Sprintf(missingIDsStr, strings.Join(missingIDs, ", "))),
				)
				trimmedContent := strings.TrimSpace(completion.Choices[0].Message.Content)
				r.cfg.Logger.Debug("OpenAI API response", "content", trimmedContent)
				continue
			}

			return rankedResponse, nil
		}

		if err == context.DeadlineExceeded {
			r.logFromApiCall(trialNum, batchNum, "Context deadline exceeded, retrying...")
			time.Sleep(backoff)
			backoff *= 2
			continue
		}

		if customTransport.StatusCode == http.StatusTooManyRequests {
			for key, values := range customTransport.Headers {
				if strings.HasPrefix(key, "X-Ratelimit") {
					for _, value := range values {
						r.logFromApiCall(trialNum, batchNum, fmt.Sprintf("Rate limit header: %s: %s", key, value))
					}
				}
			}

			respBody := customTransport.Body
			if respBody == nil {
				r.logFromApiCall(trialNum, batchNum, "Error reading response body: %v", "response body is nil")
			} else {
				r.logFromApiCall(trialNum, batchNum, "Response body: %s", string(respBody))
			}

			remainingTokensStr := customTransport.Headers.Get("X-Ratelimit-Remaining-Tokens")
			resetTokensStr := customTransport.Headers.Get("X-Ratelimit-Reset-Tokens")

			remainingTokens, _ := strconv.Atoi(remainingTokensStr)
			resetDuration, _ := time.ParseDuration(strings.Replace(resetTokensStr, "s", "s", 1))

			r.logFromApiCall(trialNum, batchNum, fmt.Sprintf("Rate limit exceeded. Suggested wait time: %v. Remaining tokens: %d", resetDuration, remainingTokens))

			if resetDuration > 0 {
				r.logFromApiCall(trialNum, batchNum, fmt.Sprintf("Waiting for %v before retrying...", resetDuration))
				time.Sleep(resetDuration)
			} else {
				r.logFromApiCall(trialNum, batchNum, fmt.Sprintf("Waiting for %v before retrying...", backoff))
				time.Sleep(backoff)
				backoff *= 2
			}
		} else {
			return rankedDocumentResponse{}, fmt.Errorf("trial %*d/%d, batch %*d/%d: unexpected error: %w", len(strconv.Itoa(r.cfg.NumTrials)), trialNum, r.cfg.NumTrials, len(strconv.Itoa(r.numBatches)), batchNum, r.numBatches, err)
		}
	}
}
