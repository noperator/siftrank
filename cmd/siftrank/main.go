package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"

	"github.com/noperator/siftrank/pkg/siftrank"
	"github.com/openai/openai-go"
	"github.com/spf13/cobra"
)

var (
	// Input/Output
	inputFile  string
	forceJSON  bool
	outputFile string

	// Prompt/Template
	initialPrompt string
	inputTemplate string

	// Algorithm params
	batchSize       int
	maxTrials       int
	concurrency     int
	batchTokens     int
	refinementRatio float64

	// Model params
	oaiModel string
	oaiURL   string
	encoding string
	effort   string

	// Convergence params
	noConverge     bool
	elbowTolerance float64
	stableTrials   int
	minTrials      int
	elbowMethod    string

	// Execution params
	dryRun    bool
	debug     bool
	relevance bool
	traceFile string
	watch     bool
	noMinimap bool
	logFile   string
)

var rootCmd = &cobra.Command{
	Use:   "siftrank",
	Short: "Use LLMs for document ranking via the SiftRank algorithm",
	Long: `siftrank uses the SiftRank algorithm to rank documents using large language models.

SiftRank employs multiple randomized trials with pairwise comparisons to create
stable, reliable rankings even with non-deterministic LLM outputs.

Examples:
  # Rank sentences by relevance to "time"
  siftrank -f sentences.txt -p "Rank by relevance to time"

  # Rank JSON objects using a template
  siftrank -f data.json --template "{{.title}}: {{.description}}"

  # Use more trials for higher confidence
  siftrank -f items.txt -p "Best to worst" --max-trials 100`,
	RunE: run,
}

func init() {
	// Input/Output flags
	rootCmd.Flags().StringVarP(&inputFile, "file", "f", "", "input file (required)")
	rootCmd.Flags().BoolVar(&forceJSON, "json", false, "force JSON parsing regardless of file extension")
	rootCmd.Flags().StringVarP(&outputFile, "output", "o", "", "JSON output file")
	rootCmd.MarkFlagRequired("file")

	// Prompt/Template flags
	rootCmd.Flags().StringVarP(&initialPrompt, "prompt", "p", "", "initial prompt (prefix with @ to use a file)")
	rootCmd.Flags().StringVar(&inputTemplate, "template", "{{.Data}}", "template for each object (prefix with @ to use a file)")

	// Algorithm parameter flags
	rootCmd.Flags().IntVarP(&batchSize, "batch-size", "b", 10, "number of items per batch")
	rootCmd.Flags().IntVar(&maxTrials, "max-trials", 50, "maximum number of ranking trials")
	rootCmd.Flags().IntVarP(&concurrency, "concurrency", "c", 50, "max concurrent LLM calls across all trials")
	rootCmd.Flags().IntVar(&batchTokens, "tokens", 128000, "max tokens per batch")
	rootCmd.Flags().Float64Var(&refinementRatio, "ratio", 0.5, "refinement ratio (0.0-1.0, e.g. 0.5 = top 50%)")

	// Model parameter flags
	rootCmd.Flags().StringVarP(&oaiModel, "model", "m", openai.ChatModelGPT4oMini, "OpenAI model name")
	rootCmd.Flags().StringVarP(&oaiURL, "base-url", "u", "", "OpenAI API base URL (for compatible APIs like vLLM)")
	rootCmd.Flags().StringVar(&encoding, "encoding", "o200k_base", "tokenizer encoding")
	rootCmd.Flags().StringVarP(&effort, "effort", "e", "", "reasoning effort level: none, minimal, low, medium, high")

	// Convergence parameter flags
	rootCmd.Flags().BoolVar(&noConverge, "no-converge", false, "disable early stopping based on convergence")
	rootCmd.Flags().Float64Var(&elbowTolerance, "elbow-tolerance", 0.05, "elbow position tolerance (0.05 = 5%)")
	rootCmd.Flags().IntVar(&stableTrials, "stable-trials", 5, "stable trials required for convergence")
	rootCmd.Flags().IntVar(&minTrials, "min-trials", 5, "minimum trials before checking convergence")
	rootCmd.Flags().StringVar(&elbowMethod, "elbow-method", "curvature", "elbow detection method: curvature (default), perpendicular")

	// Execution flags
	rootCmd.Flags().BoolVar(&dryRun, "dry-run", false, "log API calls without making them")
	rootCmd.Flags().BoolVarP(&debug, "debug", "d", false, "enable debug logging")
	rootCmd.Flags().BoolVarP(&relevance, "relevance", "r", false, "post-process each item by providing relevance justification (skips round 1)")
	rootCmd.Flags().StringVar(&traceFile, "trace", "", "trace file path for streaming trial execution state (JSON Lines format)")
	rootCmd.Flags().BoolVar(&watch, "watch", false, "enable live terminal visualization (logs suppressed unless --log is specified)")
	rootCmd.Flags().BoolVar(&noMinimap, "no-minimap", false, "disable minimap panel in watch mode")
	rootCmd.Flags().StringVar(&logFile, "log", "", "write logs to file instead of stderr")
}

func run(cmd *cobra.Command, args []string) error {
	// Set up logging
	logLevel := slog.LevelInfo
	if debug {
		logLevel = slog.LevelDebug
	}

	var logWriter *os.File
	var logOutput io.Writer = os.Stderr
	if logFile != "" {
		var err error
		logWriter, err = os.OpenFile(logFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			return fmt.Errorf("failed to open log file: %w", err)
		}
		defer logWriter.Close()
		logOutput = logWriter
	} else if watch {
		// Suppress logs when --watch is used without --log
		logOutput = io.Discard
	}

	logger := slog.New(slog.NewTextHandler(logOutput, &slog.HandlerOptions{
		Level: logLevel,
	})).With("component", "siftrank-cli")

	// Validate refinement ratio
	if refinementRatio < 0 || refinementRatio >= 1 {
		return fmt.Errorf("refinement ratio must be >= 0 and < 1")
	}

	// Load prompt from file if needed
	userPrompt := initialPrompt
	if strings.HasPrefix(userPrompt, "@") {
		filePath := strings.TrimPrefix(userPrompt, "@")
		content, err := os.ReadFile(filePath)
		if err != nil {
			return fmt.Errorf("could not read initial prompt file: %w", err)
		}
		userPrompt = string(content)
	}

	// Create config
	config := &siftrank.Config{
		InitialPrompt:   userPrompt,
		BatchSize:       batchSize,
		NumTrials:       maxTrials,
		Concurrency:     concurrency,
		OpenAIModel:     oaiModel,
		RefinementRatio: refinementRatio,
		OpenAIKey:       os.Getenv("OPENAI_API_KEY"),
		OpenAIAPIURL:    oaiURL,
		Encoding:        encoding,
		BatchTokens:     batchTokens,
		DryRun:          dryRun,
		TracePath:       traceFile,
		Relevance:       relevance,
		Effort:          effort,
		LogLevel:  logLevel,
		Logger:    logger,
		Watch:     watch,
		NoMinimap: noMinimap,

		EnableConvergence: !noConverge,
		ElbowTolerance:    elbowTolerance,
		StableTrials:      stableTrials,
		MinTrials:         minTrials,
		ElbowMethod:       elbowMethod,
	}

	// Create ranker
	ranker, err := siftrank.NewRanker(config)
	if err != nil {
		return fmt.Errorf("failed to create ranker: %w", err)
	}

	// Rank documents
	finalResults, err := ranker.RankFromFile(inputFile, inputTemplate, forceJSON)
	if err != nil {
		return fmt.Errorf("failed to rank from file: %w", err)
	}

	// Marshal results to JSON
	jsonResults, err := json.MarshalIndent(finalResults, "", "  ")
	if err != nil {
		return fmt.Errorf("could not marshal results to JSON: %w", err)
	}

	// Print results to stdout (unless dry run)
	if !config.DryRun {
		fmt.Println(string(jsonResults))
	}

	// Write to output file if specified
	if outputFile != "" {
		if err := os.WriteFile(outputFile, jsonResults, 0644); err != nil {
			return fmt.Errorf("failed to write output file: %w", err)
		}
		logger.Info("results written to file", "file", outputFile)
	}

	return nil
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}
