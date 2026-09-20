// closecalls captures one fixed candidate batch across input-order shuffles.
// It does not run SiftRank's recursive ranking engine or read evaluation labels.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/noperator/siftrank/pkg/siftrank"
)

type batch struct {
	Criterion  string                      `json:"criterion"`
	Candidates []siftrank.RankingCandidate `json:"candidates"`
}

type capture struct {
	RunID      string                        `json:"run_id"`
	CaseID     string                        `json:"case_id"`
	Criterion  string                        `json:"criterion"`
	Candidates []siftrank.RankingCandidate   `json:"candidates"`
	Pairs      []siftrank.PairwiseComparison `json:"pairs"`
	Ordering   []string                      `json:"ordering"`
	Model      string                        `json:"model"`
	RequestID  string                        `json:"request_id"`
	Usage      tokenUsage                    `json:"usage"`
	Status     string                        `json:"status"`
	Error      string                        `json:"error"`
}

type tokenUsage struct {
	Input     int `json:"input_tokens"`
	Output    int `json:"output_tokens"`
	Reasoning int `json:"reasoning_tokens"`
}

type providerFactory func(string, string) (siftrank.RankingProvider, error)

func main() {
	err := run(os.Args[1:], os.Stdout, os.Getenv, func(key, model string) (siftrank.RankingProvider, error) {
		return siftrank.NewJevProvider(siftrank.JevConfig{APIKey: key, Model: model})
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "closecalls:", err)
		os.Exit(1)
	}
}

func run(args []string, out io.Writer, getenv func(string) string, factory providerFactory) error {
	flags := flag.NewFlagSet("closecalls", flag.ContinueOnError)
	flags.SetOutput(out)
	inputPath := flags.String("input", "", "label-free JSON candidate batch (2–16 candidates)")
	outputPath := flags.String("output", "", "new JSONL capture file (never overwritten)")
	caseID := flags.String("case-id", "demo", "stable identifier for this batch")
	model := flags.String("model", "jev-1.13.0", "Jev model")
	seedText := flags.String("seeds", "1,2,3", "1–16 unique comma-separated shuffle seeds")
	timeout := flags.Duration("timeout", 30*time.Second, "per-call timeout, at most 5m (includes retries)")
	live := flags.Bool("live", false, "allow API calls using TYPESAFE_API_KEY; default is dry run")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || *inputPath == "" || strings.TrimSpace(*caseID) == "" || strings.TrimSpace(*model) == "" {
		return errors.New("provide -input, a nonempty -case-id and -model, and no positional arguments")
	}
	if *timeout <= 0 || *timeout > 5*time.Minute {
		return errors.New("-timeout must be positive and at most 5m")
	}
	seeds, err := parseSeeds(*seedText)
	if err != nil {
		return err
	}
	input, err := readBatch(*inputPath)
	if err != nil {
		return err
	}
	if !*live {
		_, err := fmt.Fprintf(out, "dry run: %d fixed-batch requests, %d candidates and %d pairs per request; pass -live and -output to capture\n", len(seeds), len(input.Candidates), len(input.Candidates)*(len(input.Candidates)-1)/2)
		return err
	}
	if *outputPath == "" {
		return errors.New("-live requires -output")
	}
	key := getenv("TYPESAFE_API_KEY")
	if key == "" {
		return errors.New("-live requires TYPESAFE_API_KEY")
	}
	provider, err := factory(key, *model)
	if err != nil {
		return err
	}
	file, err := os.OpenFile(*outputPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	defer file.Close()
	encoder := json.NewEncoder(file)
	failures := 0
	for _, seed := range seeds {
		candidates := append([]siftrank.RankingCandidate(nil), input.Candidates...)
		rand.New(rand.NewSource(seed)).Shuffle(len(candidates), func(i, j int) { candidates[i], candidates[j] = candidates[j], candidates[i] })
		record := capture{RunID: fmt.Sprintf("%s/seed-%d", *caseID, seed), CaseID: *caseID, Criterion: input.Criterion,
			Candidates: candidates, Pairs: []siftrank.PairwiseComparison{}, Ordering: []string{}, Model: *model, Status: "ok"}
		ctx, cancel := context.WithTimeout(context.Background(), *timeout)
		opts := &siftrank.CompletionOptions{}
		result, callErr := provider.CompleteRanking(ctx, siftrank.RankingInput{Prompt: input.Criterion, Documents: candidates}, opts)
		cancel()
		record.RequestID = opts.RequestID
		record.Usage = tokenUsage{opts.Usage.InputTokens, opts.Usage.OutputTokens, opts.Usage.ReasoningTokens}
		if opts.ModelUsed != "" {
			record.Model = opts.ModelUsed
		}
		if callErr == nil {
			var response struct {
				Docs []string `json:"docs"`
			}
			callErr = json.Unmarshal([]byte(result), &response)
			if callErr == nil {
				callErr = validateCapture(candidates, response.Docs, opts.PairwiseComparisons)
			}
			if callErr == nil {
				record.Ordering, record.Pairs = response.Docs, opts.PairwiseComparisons
			}
		}
		if callErr != nil {
			failures++
			record.Status, record.Error = "error", callErr.Error()
		}
		if err := encoder.Encode(record); err != nil {
			return err
		}
		if err := file.Sync(); err != nil {
			return err
		}
	}
	if err := file.Close(); err != nil {
		return err
	}
	if failures > 0 {
		return fmt.Errorf("%d of %d calls failed; errors retained in %s", failures, len(seeds), *outputPath)
	}
	_, err = fmt.Fprintf(out, "captured %d fixed-batch requests in %s\n", len(seeds), *outputPath)
	return err
}

func parseSeeds(value string) ([]int64, error) {
	parts := strings.Split(value, ",")
	if len(parts) > 16 {
		return nil, errors.New("provide at most 16 seeds")
	}
	seen := make(map[int64]bool)
	seeds := make([]int64, 0, len(parts))
	for _, part := range parts {
		seed, err := strconv.ParseInt(strings.TrimSpace(part), 10, 64)
		if err != nil || seen[seed] {
			return nil, errors.New("seeds must be unique integers")
		}
		seen[seed] = true
		seeds = append(seeds, seed)
	}
	return seeds, nil
}

func readBatch(path string) (batch, error) {
	var input batch
	file, err := os.Open(path)
	if err != nil {
		return input, err
	}
	defer file.Close()
	const limit = 1 << 20
	data, err := io.ReadAll(io.LimitReader(file, limit+1))
	if err != nil || len(data) > limit {
		return input, errors.New("input cannot be read or exceeds 1 MiB")
	}
	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.DisallowUnknownFields() // Reject accidental label fields.
	if err := decoder.Decode(&input); err != nil {
		return input, err
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return input, errors.New("input must contain exactly one JSON object")
	}
	if strings.TrimSpace(input.Criterion) == "" || len(input.Candidates) < 2 || len(input.Candidates) > 16 {
		return input, errors.New("input requires a criterion and 2–16 candidates")
	}
	seen := make(map[string]bool)
	for _, candidate := range input.Candidates {
		if strings.TrimSpace(candidate.ID) == "" || seen[candidate.ID] {
			return input, errors.New("candidate IDs must be unique and nonempty")
		}
		seen[candidate.ID] = true
	}
	return input, nil
}

func validateCapture(candidates []siftrank.RankingCandidate, ordering []string, pairs []siftrank.PairwiseComparison) error {
	if len(ordering) != len(candidates) || len(pairs) != len(candidates)*(len(candidates)-1)/2 {
		return errors.New("provider returned an incomplete ordering or comparison matrix")
	}
	seen := make(map[string]bool)
	for _, candidate := range candidates {
		seen[candidate.ID] = true
	}
	for _, id := range ordering {
		if !seen[id] {
			return errors.New("provider ordering contains a duplicate or unknown ID")
		}
		delete(seen, id)
	}
	return nil
}
