package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/noperator/siftrank/pkg/siftrank"
)

type fakeProvider func(context.Context, siftrank.RankingInput, *siftrank.CompletionOptions) (string, error)

func (f fakeProvider) CompleteRanking(ctx context.Context, input siftrank.RankingInput, opts *siftrank.CompletionOptions) (string, error) {
	return f(ctx, input, opts)
}

func writeInput(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "input.json")
	if err := os.WriteFile(path, []byte(content), 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

const testInput = `{"criterion":"find useful evidence","candidates":[{"id":"source-a","value":"first"},{"id":"source-b","value":"second"},{"id":"source-c","value":"third"}]}`

func TestDryRunNeedsNoCredentialsOrProvider(t *testing.T) {
	output := filepath.Join(t.TempDir(), "capture.jsonl")
	var log bytes.Buffer
	err := run([]string{"-input", writeInput(t, testInput), "-output", output}, &log,
		func(string) string { t.Fatal("dry run read credentials"); return "" },
		func(string, string) (siftrank.RankingProvider, error) {
			t.Fatal("dry run created provider")
			return nil, nil
		})
	if err != nil || !strings.Contains(log.String(), "dry run: 3 fixed-batch requests") {
		t.Fatalf("dry run: %v, %s", err, log.String())
	}
	if _, err := os.Stat(output); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("dry run created output: %v", err)
	}
}

func TestRejectLabelsAndInvalidInputsBeforeCredentials(t *testing.T) {
	for _, input := range []string{
		strings.Replace(testInput, `"criterion":`, `"grades":{"source-a":1},"criterion":`, 1),
		strings.Replace(testInput, `"value":"first"`, `"value":"first","relevance":1`, 1),
		strings.Replace(testInput, "source-b", "source-a", 1),
		testInput + `{}`,
		`{"criterion":"find evidence","candidates":[]}`,
	} {
		err := run([]string{"-input", writeInput(t, input), "-live", "-output", filepath.Join(t.TempDir(), "capture.jsonl")}, io.Discard,
			func(string) string { t.Fatal("invalid input read credentials"); return "" }, nil)
		if err == nil {
			t.Fatalf("accepted invalid or labeled input: %s", input)
		}
	}
	for _, seeds := range []string{"", "1,1", "bad", strings.Repeat("1,", 16) + "2"} {
		if _, err := parseSeeds(seeds); err == nil {
			t.Fatalf("accepted seed list %q", seeds)
		}
	}
}

func TestLiveRetainsErrorsAndOriginalCandidateIdentities(t *testing.T) {
	inputPath := writeInput(t, testInput)
	output := filepath.Join(t.TempDir(), "capture.jsonl")
	calls := 0
	provider := fakeProvider(func(ctx context.Context, input siftrank.RankingInput, opts *siftrank.CompletionOptions) (string, error) {
		calls++
		if deadline, ok := ctx.Deadline(); !ok || time.Until(deadline) > time.Second {
			t.Error("per-call timeout not enforced")
		}
		if input.Prompt != "find useful evidence" {
			t.Error("criterion changed")
		}
		want := map[string]string{"source-a": "first", "source-b": "second", "source-c": "third"}
		for _, candidate := range input.Documents {
			if want[candidate.ID] != candidate.Value {
				t.Errorf("candidate changed: %+v", candidate)
			}
			delete(want, candidate.ID)
		}
		if len(want) != 0 {
			t.Error("lost candidates")
		}
		opts.RequestID, opts.ModelUsed = "request", "resolved-model"
		opts.Usage.InputTokens = 10
		var ids []string
		for i, first := range input.Documents {
			ids = append(ids, first.ID)
			for _, second := range input.Documents[i+1:] {
				opts.PairwiseComparisons = append(opts.PairwiseComparisons, siftrank.PairwiseComparison{FirstID: first.ID, SecondID: second.ID, Probability: 0.5})
			}
		}
		if calls == 1 {
			return "", errors.New("record this failure")
		}
		data, _ := json.Marshal(map[string]any{"docs": ids})
		return string(data), nil
	})
	factory := func(key, model string) (siftrank.RankingProvider, error) {
		if key != "test-key" || model != "jev-1.13.0" {
			t.Errorf("unexpected config: %s", model)
		}
		return provider, nil
	}
	err := run([]string{"-input", inputPath, "-output", output, "-seeds", "1,2", "-timeout", "1s", "-live"}, io.Discard,
		func(name string) string {
			if name != "TYPESAFE_API_KEY" {
				t.Fatal(name)
			}
			return "test-key"
		}, factory)
	if err == nil || !strings.Contains(err.Error(), "1 of 2 calls failed") || calls != 2 {
		t.Fatalf("failure was lost: calls=%d, %v", calls, err)
	}
	info, err := os.Stat(output)
	if err != nil || info.Mode().Perm() != 0600 {
		t.Fatalf("output permissions: %v, %v", info, err)
	}
	data, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	var records []capture
	decoder := json.NewDecoder(bytes.NewReader(data))
	for {
		var record capture
		if err := decoder.Decode(&record); err == io.EOF {
			break
		} else if err != nil {
			t.Fatal(err)
		}
		records = append(records, record)
	}
	if len(records) != 2 || records[0].Status != "error" || records[0].Error != "record this failure" || len(records[0].Pairs) != 0 || len(records[0].Ordering) != 0 {
		t.Fatalf("invalid failure capture: %+v", records)
	}
	if records[1].Status != "ok" || len(records[1].Pairs) != 3 || records[1].RequestID != "request" || records[1].Model != "resolved-model" || records[1].Usage.Input != 10 || records[1].RunID != "demo/seed-2" {
		t.Fatalf("invalid success capture: %+v", records[1])
	}
	// A reused path must not make another call or overwrite the first capture.
	err = run([]string{"-input", inputPath, "-output", output, "-live"}, io.Discard, func(string) string { return "test-key" }, factory)
	after, _ := os.ReadFile(output)
	if err == nil || calls != 2 || !bytes.Equal(data, after) {
		t.Fatal("existing capture was overwritten or another call was made")
	}
	original, err := readBatch(inputPath)
	if err != nil || !reflect.DeepEqual(original.Candidates, []siftrank.RankingCandidate{{ID: "source-a", Value: "first"}, {ID: "source-b", Value: "second"}, {ID: "source-c", Value: "third"}}) {
		t.Fatal("input file changed")
	}
}

func TestLiveRequiresCredentialsAndCompleteMetadata(t *testing.T) {
	output := filepath.Join(t.TempDir(), "capture.jsonl")
	args := []string{"-input", writeInput(t, testInput), "-output", output, "-seeds", "1", "-live"}
	if err := run(args, io.Discard, func(string) string { return "" }, nil); err == nil {
		t.Fatal("accepted missing credentials")
	}
	err := run(args, io.Discard, func(string) string { return "test-key" }, func(string, string) (siftrank.RankingProvider, error) {
		return fakeProvider(func(context.Context, siftrank.RankingInput, *siftrank.CompletionOptions) (string, error) {
			return `{"docs":["source-a","source-b","source-c"]}`, nil
		}), nil
	})
	if err == nil {
		t.Fatal("accepted ordering without comparisons")
	}
	data, _ := os.ReadFile(output)
	if !bytes.Contains(data, []byte(`"status":"error"`)) {
		t.Fatal("missing-metadata error not retained")
	}
}
