package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/noperator/siftrank/pkg/siftrank"
	"github.com/openai/openai-go"
)

func main() {
	inputFile := flag.String("f", "", "Input file")
	forceJSON := flag.Bool("json", false, "Force JSON parsing regardless of file extension")
	inputTemplate := flag.String("template", "{{.Data}}", "Template for each object in the input file (prefix with @ to use a file)")
	batchSize := flag.Int("s", 10, "Number of items per batch")
	numTrials := flag.Int("r", 50, "Number of trials")
	concurrency := flag.Int("c", 20, "Max concurrent LLM calls across all trials")
	batchTokens := flag.Int("t", 128000, "Max tokens per batch")
	initialPrompt := flag.String("p", "", "Initial prompt (prefix with @ to use a file)")
	outputFile := flag.String("o", "", "JSON output file")

	oaiModel := flag.String("openai-model", openai.ChatModelGPT4oMini, "OpenAI model name")
	oaiURL := flag.String("openai-url", "", "OpenAI API base URL (e.g., for OpenAI-compatible API like vLLM)")
	encoding := flag.String("encoding", "o200k_base", "Tokenizer encoding")

	dryRun := flag.Bool("dry-run", false, "Enable dry run mode (log API calls without making them)")
	refinementRatio := flag.Float64("ratio", 0.5, "Refinement ratio as a decimal (e.g., 0.5 for 50%)")
	debug := flag.Bool("debug", false, "Enable debug logging")

	noConverge := flag.Bool("no-converge", false, "Disable early stopping based on convergence")
	elbowTolerance := flag.Float64("elbow-tolerance", 0.05, "Elbow position tolerance (0.05 = 5%)")
	stableTrials := flag.Int("stable-trials", 5, "Stable trials required for convergence")
	minTrials := flag.Int("min-trials", 5, "Minimum trials before checking convergence")

	flag.Parse()

	// Set up structured logging with level based on debug flag
	logLevel := slog.LevelInfo
	if *debug {
		logLevel = slog.LevelDebug
	}
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: logLevel,
	})).With("component", "siftrank-cli")

	// This "threshold" is a way to add some padding to our estimation of
	// average token usage per batch. We're effectively leaving 5% of
	// wiggle room.
	var tokenLimitThreshold = int(0.95 * float64(*batchTokens))

	if *inputFile == "" {
		logger.Error("Usage: raink -f <input_file> [-s <batch_size>] [-r <num_trials>] [-p <initial_prompt>] [-t <batch_tokens>] [-openai-model <model_name>] [-openai-url <base_url>] [-ratio <refinement_ratio>]")
		return
	}

	if *refinementRatio < 0 || *refinementRatio >= 1 {
		fmt.Println("refinement ratio must be >= 0 and < 1")
		os.Exit(1)
	}

	userPrompt := *initialPrompt
	if strings.HasPrefix(userPrompt, "@") {
		filePath := strings.TrimPrefix(userPrompt, "@")
		content, err := os.ReadFile(filePath)
		if err != nil {
			logger.Error("could not read initial prompt file", "error", err)
			os.Exit(1)
		}
		userPrompt = string(content)
	}

	config := &siftrank.Config{
		InitialPrompt:   userPrompt,
		BatchSize:       *batchSize,
		NumTrials:       *numTrials,
		Concurrency:     *concurrency,
		OpenAIModel:     *oaiModel,
		TokenLimit:      tokenLimitThreshold,
		RefinementRatio: *refinementRatio,
		OpenAIKey:       os.Getenv("OPENAI_API_KEY"),
		OpenAIAPIURL:    *oaiURL,
		Encoding:        *encoding,
		BatchTokens:     *batchTokens,
		DryRun:          *dryRun,
		LogLevel:        logLevel,

		EnableConvergence: !*noConverge,
		ElbowTolerance:    *elbowTolerance,
		StableTrials:      *stableTrials,
		MinTrials:         *minTrials,
	}

	ranker, err := siftrank.NewRanker(config)
	if err != nil {
		logger.Error("failed to create ranker", "error", err)
		os.Exit(1)
	}

	finalResults, err := ranker.RankFromFile(*inputFile, *inputTemplate, *forceJSON)
	if err != nil {
		logger.Error("failed to rank from file", "error", err)
		os.Exit(1)
	}

	jsonResults, err := json.MarshalIndent(finalResults, "", "  ")
	if err != nil {
		logger.Error("could not marshal results to JSON", "error", err)
		os.Exit(1)
	}

	if !config.DryRun {
		fmt.Println(string(jsonResults))
	}

	if *outputFile != "" {
		os.WriteFile(*outputFile, jsonResults, 0644)
		logger.Info("results written to file", "file", *outputFile)
	}
}
