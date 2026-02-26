package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/noperator/siftrank/pkg/siftrank"
	"github.com/openai/openai-go"
	"gopkg.in/yaml.v3"
)

// ConfigFile is the top-level structure of the YAML config file.
type ConfigFile struct {
	Default  string                            `yaml:"default"`
	Profiles map[string]map[string]interface{} `yaml:"profiles"`
}

// findConfigFile implements the config file discovery order.
// If flagPath is non-empty, returns it directly (caller should verify existence).
// Otherwise checks ./config.yaml then ~/.config/siftrank/config.yaml.
// Returns the path and whether a file was found.
func findConfigFile(flagPath string) (string, bool) {
	if flagPath != "" {
		return flagPath, true
	}

	// Check current working directory
	if _, err := os.Stat("config.yaml"); err == nil {
		return "config.yaml", true
	}

	// Check ~/.config/siftrank/config.yaml
	home, err := os.UserHomeDir()
	if err == nil {
		configPath := filepath.Join(home, ".config", "siftrank", "config.yaml")
		if _, err := os.Stat(configPath); err == nil {
			return configPath, true
		}
	}

	return "", false
}

// loadConfigFile reads and parses the YAML config file.
func loadConfigFile(path string) (*ConfigFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var cfg ConfigFile
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	return &cfg, nil
}

// resolveProfile looks up a named profile in the config file.
func resolveProfile(cfg *ConfigFile, name string) (map[string]interface{}, error) {
	profile, ok := cfg.Profiles[name]
	if !ok {
		return nil, fmt.Errorf("profile %q not found in config file", name)
	}
	return profile, nil
}

// resolveCMDFields processes _cmd keys in the profile map.
// For any key ending in "_cmd", it runs the value as a shell command
// and replaces it with the base key containing the command's stdout.
func resolveCMDFields(profile map[string]interface{}) error {
	cmdKeys := make([]string, 0)
	for key := range profile {
		if strings.HasSuffix(key, "_cmd") {
			cmdKeys = append(cmdKeys, key)
		}
	}

	for _, cmdKey := range cmdKeys {
		baseKey := strings.TrimSuffix(cmdKey, "_cmd")

		// Check for conflict
		if _, hasBase := profile[baseKey]; hasBase {
			return fmt.Errorf("profile contains both %q and %q; only one is allowed", baseKey, cmdKey)
		}

		cmdValue, ok := profile[cmdKey].(string)
		if !ok {
			return fmt.Errorf("%q value must be a string", cmdKey)
		}

		// Run the command
		cmd := exec.Command("sh", "-c", cmdValue)
		output, err := cmd.Output()
		if err != nil {
			if exitErr, ok := err.(*exec.ExitError); ok {
				return fmt.Errorf("%q command failed: %s", cmdKey, string(exitErr.Stderr))
			}
			return fmt.Errorf("%q command failed: %w", cmdKey, err)
		}

		// Set the base key with trimmed output
		profile[baseKey] = strings.TrimSpace(string(output))
		delete(profile, cmdKey)
	}

	return nil
}

// applyProfile applies profile values to the config, respecting CLI flag precedence.
// Only values present in the profile and not explicitly set via CLI flags are applied.
func applyProfile(profile map[string]interface{}, config *siftrank.Config, changed func(string) bool) {
	// exceptions maps yaml tag names to their CLI flag names, only for fields
	// where underscore→hyphen translation alone doesn't produce the right flag name.
	// Empty string means no CLI flag — always apply.
	exceptions := map[string]string{
		"api_key":            "",            // no CLI flag, always apply
		"enable_convergence": "no-converge", // inverted/renamed CLI flag
	}

	configVal := reflect.ValueOf(config).Elem()
	configType := configVal.Type()

	for i := 0; i < configType.NumField(); i++ {
		field := configType.Field(i)
		yamlTag := field.Tag.Get("yaml")

		// Skip fields without yaml tags or with "-"
		if yamlTag == "" || yamlTag == "-" {
			continue
		}

		// Check if the yaml key exists in the profile
		profileValue, ok := profile[yamlTag]
		if !ok {
			continue
		}

		// Determine the CLI flag name
		flagName, isException := exceptions[yamlTag]
		if !isException {
			// Default: translate underscores to hyphens to get CLI flag name
			flagName = strings.ReplaceAll(yamlTag, "_", "-")
		}

		// Empty flagName means no CLI flag exists — always apply
		if flagName != "" && changed(flagName) {
			continue
		}

		// For api_key specifically: env var takes precedence over profile
		if yamlTag == "api_key" && config.OpenAIKey != "" {
			continue
		}

		// Apply the profile value to the config field
		fieldVal := configVal.Field(i)
		if !fieldVal.CanSet() {
			continue
		}

		setFieldValue(fieldVal, profileValue)
	}
}

// setFieldValue sets a reflect.Value from an interface{} value.
func setFieldValue(field reflect.Value, value interface{}) {
	if value == nil {
		return
	}

	switch field.Kind() {
	case reflect.String:
		if s, ok := value.(string); ok {
			field.SetString(s)
		}
	case reflect.Int, reflect.Int64:
		switch v := value.(type) {
		case int:
			field.SetInt(int64(v))
		case int64:
			field.SetInt(v)
		case float64:
			field.SetInt(int64(v))
		}
	case reflect.Float64:
		switch v := value.(type) {
		case float64:
			field.SetFloat(v)
		case int:
			field.SetFloat(float64(v))
		}
	case reflect.Bool:
		if b, ok := value.(bool); ok {
			field.SetBool(b)
		}
	default:
		// Handle openai.ChatModel which is a string type alias
		if field.Type() == reflect.TypeOf(openai.ChatModel("")) {
			if s, ok := value.(string); ok {
				field.Set(reflect.ValueOf(openai.ChatModel(s)))
			}
		}
		// Handle siftrank.ElbowMethod which is a string type alias
		if field.Type() == reflect.TypeOf(siftrank.ElbowMethod("")) {
			if s, ok := value.(string); ok {
				field.Set(reflect.ValueOf(siftrank.ElbowMethod(s)))
			}
		}
	}
}

