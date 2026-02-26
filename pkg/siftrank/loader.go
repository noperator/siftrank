package siftrank

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"text/template"
)

func (r *Ranker) loadDocumentsFromFile(filePath string, templateData string, forceJSON bool) ([]document, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open input file %s: %w", filePath, err)
	}
	defer file.Close()

	ext := strings.ToLower(filepath.Ext(filePath))
	isJSON := ext == ".json" || forceJSON

	return r.loadDocumentsFromReader(file, templateData, isJSON)
}

func (r *Ranker) loadDocumentsFromReader(reader io.Reader, templateData string, isJSON bool) ([]document, error) {
	// Template parsing
	var tmpl *template.Template
	if templateData != "" {
		if templateData[0] == '@' {
			content, err := os.ReadFile(templateData[1:])
			if err != nil {
				return nil, fmt.Errorf("failed to read template file %s: %w", templateData[1:], err)
			}
			templateData = string(content)
		}
		var err error
		if tmpl, err = template.New("siftrank-item-template").Parse(templateData); err != nil {
			return nil, fmt.Errorf("failed to parse template: %w", err)
		}
	}

	if isJSON {
		return r.loadJSONDocuments(reader, tmpl)
	}
	return r.loadTextDocuments(reader, tmpl)
}

func (r *Ranker) loadTextDocuments(reader io.Reader, tmpl *template.Template) ([]document, error) {
	content, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to read content: %w", err)
	}

	var documents []document
	lines := strings.Split(string(content), "\n")

	for i, line := range lines {
		line = strings.TrimRight(line, "\r") // Handle Windows line endings
		if line == "" {
			continue // Skip empty lines
		}

		if tmpl != nil {
			var tmplData bytes.Buffer
			if err := tmpl.Execute(&tmplData, map[string]string{"Data": line}); err != nil {
				return nil, fmt.Errorf("failed to execute template on line: %w", err)
			}
			line = tmplData.String()
		}

		id := ShortDeterministicID(line, idLen)
		documents = append(documents, document{
			ID:         id,
			Document:   nil,
			Value:      line,
			InputIndex: i,
		})
	}

	return documents, nil
}

func (r *Ranker) loadJSONDocuments(reader io.Reader, tmpl *template.Template) ([]document, error) {
	var data []interface{}
	if err := json.NewDecoder(reader).Decode(&data); err != nil {
		return nil, fmt.Errorf("failed to decode JSON: %w", err)
	}

	var documents []document
	for i, value := range data {
		var valueStr string
		if tmpl != nil {
			var tmplData bytes.Buffer
			if err := tmpl.Execute(&tmplData, value); err != nil {
				return nil, fmt.Errorf("failed to execute template: %w", err)
			}
			valueStr = tmplData.String()
		} else {
			r.cfg.Logger.Warn("using json input without a template, using JSON document as-is")
			jsonValue, err := json.Marshal(value)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal JSON value: %w", err)
			}
			valueStr = string(jsonValue)
		}

		id := ShortDeterministicID(valueStr, idLen)
		documents = append(documents, document{
			ID:         id,
			Document:   value,
			Value:      valueStr,
			InputIndex: i,
		})
	}

	return documents, nil
}
