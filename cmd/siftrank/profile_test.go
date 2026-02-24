package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/noperator/siftrank/pkg/siftrank"
	"github.com/openai/openai-go"
)

func TestFindConfigFile(t *testing.T) {
	t.Run("explicit path returns path", func(t *testing.T) {
		path, found := findConfigFile("/some/path/config.yaml")
		if !found {
			t.Error("expected found=true for explicit path")
		}
		if path != "/some/path/config.yaml" {
			t.Errorf("expected path=/some/path/config.yaml, got %s", path)
		}
	})

	t.Run("cwd config.yaml is found", func(t *testing.T) {
		// Create temp dir and config file
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.yaml")
		if err := os.WriteFile(configPath, []byte("default: test\n"), 0644); err != nil {
			t.Fatal(err)
		}

		// Change to temp dir
		origDir, _ := os.Getwd()
		if err := os.Chdir(tmpDir); err != nil {
			t.Fatal(err)
		}
		defer os.Chdir(origDir)

		path, found := findConfigFile("")
		if !found {
			t.Error("expected to find config.yaml in cwd")
		}
		if path != "config.yaml" {
			t.Errorf("expected path=config.yaml, got %s", path)
		}
	})

	t.Run("no config file returns empty", func(t *testing.T) {
		// Create temp dir without config file
		tmpDir := t.TempDir()
		origDir, _ := os.Getwd()
		if err := os.Chdir(tmpDir); err != nil {
			t.Fatal(err)
		}
		defer os.Chdir(origDir)

		path, found := findConfigFile("")
		if found {
			t.Errorf("expected found=false, got path=%s", path)
		}
	})
}

func TestLoadConfigFile(t *testing.T) {
	t.Run("valid config file", func(t *testing.T) {
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.yaml")
		content := `
default: work
profiles:
  work:
    model: o3-mini
    effort: high
  local:
    model: llama-3
`
		if err := os.WriteFile(configPath, []byte(content), 0644); err != nil {
			t.Fatal(err)
		}

		cfg, err := loadConfigFile(configPath)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if cfg.Default != "work" {
			t.Errorf("expected default=work, got %s", cfg.Default)
		}
		if len(cfg.Profiles) != 2 {
			t.Errorf("expected 2 profiles, got %d", len(cfg.Profiles))
		}
		if cfg.Profiles["work"]["model"] != "o3-mini" {
			t.Errorf("expected work profile model=o3-mini, got %v", cfg.Profiles["work"]["model"])
		}
	})

	t.Run("invalid yaml", func(t *testing.T) {
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.yaml")
		content := `
default: [invalid yaml
`
		if err := os.WriteFile(configPath, []byte(content), 0644); err != nil {
			t.Fatal(err)
		}

		_, err := loadConfigFile(configPath)
		if err == nil {
			t.Error("expected error for invalid yaml")
		}
	})

	t.Run("file not found", func(t *testing.T) {
		_, err := loadConfigFile("/nonexistent/config.yaml")
		if err == nil {
			t.Error("expected error for nonexistent file")
		}
	})
}

func TestResolveProfile(t *testing.T) {
	cfg := &ConfigFile{
		Profiles: map[string]map[string]interface{}{
			"work": {"model": "o3-mini"},
		},
	}

	t.Run("existing profile", func(t *testing.T) {
		profile, err := resolveProfile(cfg, "work")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if profile["model"] != "o3-mini" {
			t.Errorf("expected model=o3-mini, got %v", profile["model"])
		}
	})

	t.Run("nonexistent profile", func(t *testing.T) {
		_, err := resolveProfile(cfg, "nonexistent")
		if err == nil {
			t.Error("expected error for nonexistent profile")
		}
	})
}

func TestResolveCMDFields(t *testing.T) {
	t.Run("simple command", func(t *testing.T) {
		profile := map[string]interface{}{
			"api_key_cmd": "echo test-key",
		}
		err := resolveCMDFields(profile)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if profile["api_key"] != "test-key" {
			t.Errorf("expected api_key=test-key, got %v", profile["api_key"])
		}
		if _, exists := profile["api_key_cmd"]; exists {
			t.Error("expected api_key_cmd to be deleted")
		}
	})

	t.Run("trims whitespace", func(t *testing.T) {
		profile := map[string]interface{}{
			"api_key_cmd": "echo '  spaced  '",
		}
		err := resolveCMDFields(profile)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if profile["api_key"] != "spaced" {
			t.Errorf("expected api_key=spaced, got %v", profile["api_key"])
		}
	})

	t.Run("conflict error", func(t *testing.T) {
		profile := map[string]interface{}{
			"api_key":     "literal-key",
			"api_key_cmd": "echo cmd-key",
		}
		err := resolveCMDFields(profile)
		if err == nil {
			t.Error("expected error for conflicting keys")
		}
	})

	t.Run("command failure", func(t *testing.T) {
		profile := map[string]interface{}{
			"api_key_cmd": "exit 1",
		}
		err := resolveCMDFields(profile)
		if err == nil {
			t.Error("expected error for failed command")
		}
	})

	t.Run("non-string value", func(t *testing.T) {
		profile := map[string]interface{}{
			"api_key_cmd": 123,
		}
		err := resolveCMDFields(profile)
		if err == nil {
			t.Error("expected error for non-string _cmd value")
		}
	})
}

func TestApplyProfile(t *testing.T) {
	t.Run("applies string field", func(t *testing.T) {
		profile := map[string]interface{}{
			"api_key": "profile-key",
		}
		config := &siftrank.Config{}
		applyProfile(profile, config, func(name string) bool { return false })

		if config.OpenAIKey != "profile-key" {
			t.Errorf("expected OpenAIKey=profile-key, got %s", config.OpenAIKey)
		}
	})

	t.Run("applies int field", func(t *testing.T) {
		profile := map[string]interface{}{
			"batch_size": 20,
		}
		config := &siftrank.Config{}
		applyProfile(profile, config, func(name string) bool { return false })

		if config.BatchSize != 20 {
			t.Errorf("expected BatchSize=20, got %d", config.BatchSize)
		}
	})

	t.Run("applies float64 field", func(t *testing.T) {
		profile := map[string]interface{}{
			"ratio": 0.75,
		}
		config := &siftrank.Config{}
		applyProfile(profile, config, func(name string) bool { return false })

		if config.RefinementRatio != 0.75 {
			t.Errorf("expected RefinementRatio=0.75, got %f", config.RefinementRatio)
		}
	})

	t.Run("applies bool field", func(t *testing.T) {
		profile := map[string]interface{}{
			"relevance": true,
		}
		config := &siftrank.Config{}
		applyProfile(profile, config, func(name string) bool { return false })

		if !config.Relevance {
			t.Error("expected Relevance=true")
		}
	})

	t.Run("applies model field", func(t *testing.T) {
		profile := map[string]interface{}{
			"model": "gpt-4",
		}
		config := &siftrank.Config{}
		applyProfile(profile, config, func(name string) bool { return false })

		if config.OpenAIModel != openai.ChatModel("gpt-4") {
			t.Errorf("expected OpenAIModel=gpt-4, got %s", config.OpenAIModel)
		}
	})

	t.Run("respects CLI flag precedence", func(t *testing.T) {
		profile := map[string]interface{}{
			"model":      "profile-model",
			"batch_size": 50,
		}
		config := &siftrank.Config{
			OpenAIModel: openai.ChatModel("cli-model"),
			BatchSize:   10,
		}

		// Simulate that --model was changed but --batch-size was not
		changed := func(name string) bool {
			return name == "model"
		}
		applyProfile(profile, config, changed)

		// Model should NOT be overwritten (CLI flag was set)
		if config.OpenAIModel != openai.ChatModel("cli-model") {
			t.Errorf("expected OpenAIModel=cli-model (CLI precedence), got %s", config.OpenAIModel)
		}

		// BatchSize SHOULD be overwritten (CLI flag was not set)
		if config.BatchSize != 50 {
			t.Errorf("expected BatchSize=50 (from profile), got %d", config.BatchSize)
		}
	})

	t.Run("env var takes precedence over profile for api_key", func(t *testing.T) {
		profile := map[string]interface{}{
			"api_key": "profile-key",
		}
		config := &siftrank.Config{
			OpenAIKey: "env-key", // Simulates value set from OPENAI_API_KEY env var
		}

		// Even though api_key has no CLI flag, env var should take precedence
		applyProfile(profile, config, func(name string) bool { return false })

		if config.OpenAIKey != "env-key" {
			t.Errorf("expected OpenAIKey=env-key (env var precedence), got %s", config.OpenAIKey)
		}
	})

	t.Run("profile api_key applies when env var is empty", func(t *testing.T) {
		profile := map[string]interface{}{
			"api_key": "profile-key",
		}
		config := &siftrank.Config{
			OpenAIKey: "", // No env var set
		}

		applyProfile(profile, config, func(name string) bool { return false })

		if config.OpenAIKey != "profile-key" {
			t.Errorf("expected OpenAIKey=profile-key (from profile), got %s", config.OpenAIKey)
		}
	})

	t.Run("unknown keys are ignored", func(t *testing.T) {
		profile := map[string]interface{}{
			"unknown_field": "value",
			"api_key":       "valid-key",
		}
		config := &siftrank.Config{}
		// Should not panic
		applyProfile(profile, config, func(name string) bool { return false })

		if config.OpenAIKey != "valid-key" {
			t.Errorf("expected OpenAIKey=valid-key, got %s", config.OpenAIKey)
		}
	})

	t.Run("handles yaml int as float64", func(t *testing.T) {
		// YAML parser may return numbers as float64
		profile := map[string]interface{}{
			"batch_size": float64(25),
		}
		config := &siftrank.Config{}
		applyProfile(profile, config, func(name string) bool { return false })

		if config.BatchSize != 25 {
			t.Errorf("expected BatchSize=25, got %d", config.BatchSize)
		}
	})
}
