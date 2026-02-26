package siftrank

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"math/rand"
	"regexp"
	"strings"
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

func (r *Ranker) logFromApiCall(trialNum, batchNum int, message string, args ...interface{}) {
	formattedMessage := fmt.Sprintf(message, args...)
	r.cfg.Logger.Debug(formattedMessage, "round", r.round, "trial", trialNum, "total_trials", r.cfg.NumTrials, "batch", batchNum, "total_batches", r.numBatches)
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
