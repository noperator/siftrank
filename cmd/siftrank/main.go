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
	"github.com/spf13/pflag"
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

// setFlagGroup annotates flags with a group name for organized help output.
func setFlagGroup(cmd *cobra.Command, group string, names ...string) {
	for _, name := range names {
		f := cmd.Flags().Lookup(name)
		if f != nil {
			if f.Annotations == nil {
				f.Annotations = make(map[string][]string)
			}
			f.Annotations["group"] = []string{group}
		}
	}
}

// FlagsInGroup returns a FlagSet containing only flags that belong to the specified group.
func FlagsInGroup(cmd *cobra.Command, group string) *pflag.FlagSet {
	result := pflag.NewFlagSet("grouped", pflag.ContinueOnError)
	cmd.Flags().VisitAll(func(f *pflag.Flag) {
		if g, ok := f.Annotations["group"]; ok && len(g) > 0 && g[0] == group {
			result.AddFlag(f)
		}
	})
	return result
}

// FilterFlags returns a FlagSet containing only flags that don't belong to any group (plus help).
func FilterFlags(cmd *cobra.Command) *pflag.FlagSet {
	result := pflag.NewFlagSet("ungrouped", pflag.ContinueOnError)
	cmd.Flags().VisitAll(func(f *pflag.Flag) {
		if _, ok := f.Annotations["group"]; !ok {
			result.AddFlag(f)
		}
	})
	return result
}

const usageTemplate = `Usage:
  {{.UseLine}}

Options:
{{FlagsInGroup . "options" | FlagUsages | trimTrailingWhitespaces}}

Visualization:
{{FlagsInGroup . "visualization" | FlagUsages | trimTrailingWhitespaces}}

Debug:
{{FlagsInGroup . "debug" | FlagUsages | trimTrailingWhitespaces}}

Advanced:
{{FlagsInGroup . "advanced" | FlagUsages | trimTrailingWhitespaces}}

Flags:
{{FilterFlags . | FlagUsages | trimTrailingWhitespaces}}
`

var rootCmd = &cobra.Command{
	Use:   "siftrank",
	Short: "Use LLMs for document ranking via the SiftRank algorithm",
	RunE:  run,
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
	rootCmd.Flags().IntVarP(&batchSize, "batch-size", "b", siftrank.DefaultBatchSize, "number of items per batch")
	rootCmd.Flags().IntVar(&maxTrials, "max-trials", siftrank.DefaultNumTrials, "maximum number of ranking trials")
	rootCmd.Flags().IntVarP(&concurrency, "concurrency", "c", siftrank.DefaultConcurrency, "max concurrent LLM calls across all trials")
	rootCmd.Flags().IntVar(&batchTokens, "tokens", siftrank.DefaultBatchTokens, "max tokens per batch")
	rootCmd.Flags().Float64Var(&refinementRatio, "ratio", siftrank.DefaultRefinementRatio, "refinement ratio (0.0-1.0, e.g. 0.5 = top 50%)")

	// Model parameter flags
	rootCmd.Flags().StringVarP(&oaiModel, "model", "m", openai.ChatModelGPT4oMini, "OpenAI model name")
	rootCmd.Flags().StringVarP(&oaiURL, "base-url", "u", "", "OpenAI API base URL (for compatible APIs like vLLM)")
	rootCmd.Flags().StringVar(&encoding, "encoding", siftrank.DefaultEncoding, "tokenizer encoding")
	rootCmd.Flags().StringVarP(&effort, "effort", "e", "", "reasoning effort level: none, minimal, low, medium, high")

	// Convergence parameter flags
	rootCmd.Flags().BoolVar(&noConverge, "no-converge", false, "disable early stopping based on convergence")
	rootCmd.Flags().Float64Var(&elbowTolerance, "elbow-tolerance", siftrank.DefaultElbowTolerance, "elbow position tolerance (0.05 = 5%)")
	rootCmd.Flags().IntVar(&stableTrials, "stable-trials", siftrank.DefaultStableTrials, "stable trials required for convergence")
	rootCmd.Flags().IntVar(&minTrials, "min-trials", siftrank.DefaultMinTrials, "minimum trials before checking convergence")
	rootCmd.Flags().StringVar(&elbowMethod, "elbow-method", string(siftrank.DefaultElbowMethod), "elbow detection method: curvature (default), perpendicular")

	// Execution flags
	rootCmd.Flags().BoolVar(&dryRun, "dry-run", false, "log API calls without making them")
	rootCmd.Flags().BoolVarP(&debug, "debug", "d", false, "enable debug logging")
	rootCmd.Flags().BoolVarP(&relevance, "relevance", "r", false, "post-process each item by providing relevance justification (skips round 1)")
	rootCmd.Flags().StringVar(&traceFile, "trace", "", "trace file path for streaming trial execution state (JSON Lines format)")
	rootCmd.Flags().BoolVar(&watch, "watch", false, "enable live terminal visualization (logs suppressed unless --log is specified)")
	rootCmd.Flags().BoolVar(&noMinimap, "no-minimap", false, "disable minimap panel in watch mode")
	rootCmd.Flags().StringVar(&logFile, "log", "", "write logs to file instead of stderr")

	// Register template functions for flag grouping
	cobra.AddTemplateFunc("FlagsInGroup", FlagsInGroup)
	cobra.AddTemplateFunc("FilterFlags", FilterFlags)
	cobra.AddTemplateFunc("FlagUsages", func(fs *pflag.FlagSet) string {
		return fs.FlagUsages()
	})

	// Set custom usage template
	rootCmd.SetUsageTemplate(usageTemplate)

	// Organize flags into groups
	setFlagGroup(rootCmd, "options", "file", "prompt", "output", "model", "relevance")
	setFlagGroup(rootCmd, "visualization", "watch", "no-minimap")
	setFlagGroup(rootCmd, "debug", "trace", "debug", "dry-run", "log")
	setFlagGroup(rootCmd, "advanced", "template", "json", "base-url", "encoding", "effort", "tokens", "batch-size", "max-trials", "concurrency", "ratio", "no-converge", "elbow-tolerance", "stable-trials", "min-trials", "elbow-method")
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
		ElbowMethod:       siftrank.ElbowMethod(elbowMethod),
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
