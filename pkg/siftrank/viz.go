package siftrank

import (
	"encoding/json"
	"fmt"
	"sort"

	"github.com/gdamore/tcell/v2"
)

// writeString writes a string to the screen at the given position with the given style
func (r *Ranker) writeString(screen tcell.Screen, x, y int, s string, style tcell.Style) {
	for i, ch := range s {
		screen.SetContent(x+i, y, ch, nil, style)
	}
}

// renderVisualization routes to the appropriate rendering function based on configuration
func (r *Ranker) renderVisualization(rankings []traceDocument, round, trial int) {
	if r.screen == nil {
		return
	}

	screen, ok := r.screen.(tcell.Screen)
	if !ok {
		return
	}

	// Route to appropriate rendering function
	if r.cfg.NoMinimap {
		r.renderFullWidth(screen, rankings, round, trial)
	} else {
		r.renderWithMinimap(screen, rankings, round, trial)
	}
}

// renderFullWidth renders the visualization using the full terminal width
func (r *Ranker) renderFullWidth(screen tcell.Screen, rankings []traceDocument, round, trial int) {
	screen.Clear()
	width, height := screen.Size()

	// Render using full width
	r.renderMainDisplay(screen, rankings, round, trial, 0, width, height)

	screen.Show()
}

// renderWithMinimap renders a split-screen view with main display and minimap
func (r *Ranker) renderWithMinimap(screen tcell.Screen, rankings []traceDocument, round, trial int) {
	screen.Clear()
	width, height := screen.Size()

	// Check if terminal is too narrow for minimap
	if width < 30 {
		// Fall back to full-width if too narrow
		r.renderMainDisplay(screen, rankings, round, trial, 0, width, height)
		screen.Show()
		return
	}

	// Calculate layout: 80% main, 20% minimap
	mainWidth := int(float64(width) * 0.8)
	minimapStart := mainWidth + 1
	minimapWidth := width - minimapStart

	// Draw main display (left side)
	r.renderMainDisplay(screen, rankings, round, trial, 0, mainWidth, height)

	// Draw vertical separator
	separatorStyle := tcell.StyleDefault.Foreground(tcell.ColorGray)
	for y := 0; y < height; y++ {
		screen.SetContent(mainWidth, y, '│', nil, separatorStyle)
	}

	// Draw minimap (right side)
	r.renderMinimap(screen, rankings, round, minimapStart, minimapWidth, height)

	screen.Show()
}

// renderMinimap renders a condensed overview of all rankings
func (r *Ranker) renderMinimap(screen tcell.Screen, rankings []traceDocument, round, startX, width, height int) {
	if width < 5 {
		return // Not enough space
	}

	// Header
	header := fmt.Sprintf("Map:%d", len(rankings))
	r.writeString(screen, startX, 0, header, tcell.StyleDefault.Bold(true))

	// Available height for items (reserve header + margins)
	displayHeight := height - 3
	if displayHeight < 1 {
		return
	}

	totalItems := len(rankings)
	if totalItems == 0 {
		return
	}

	// Calculate max score for normalization
	maxScore := 0.0
	if len(rankings) > 0 {
		maxScore = rankings[len(rankings)-1].Score
	}

	// Determine compression ratio
	itemsPerRow := 1.0
	if totalItems > displayHeight {
		itemsPerRow = float64(totalItems) / float64(displayHeight)
	}

	// Find elbow position in the rankings
	elbowIndex := -1
	r.mu.Lock()
	if r.cfg.EnableConvergence && len(r.elbowPositions) > 0 {
		elbowIndex = r.elbowPositions[len(r.elbowPositions)-1]
	}
	r.mu.Unlock()

	// Render each row
	for row := 0; row < displayHeight; row++ {
		y := row + 3 // Align with main display (which starts data at row 3)
		if y >= height {
			break
		}

		// Calculate which items belong to this row
		startIdx := int(float64(row) * itemsPerRow)
		endIdx := int(float64(row+1) * itemsPerRow)
		if endIdx > totalItems {
			endIdx = totalItems
		}
		if startIdx >= totalItems {
			break
		}

		// Calculate average score for this bucket
		var sumScore float64
		for i := startIdx; i < endIdx; i++ {
			sumScore += rankings[i].Score
		}
		avgScore := sumScore / float64(endIdx-startIdx)

		// Calculate bar length (inverse of normalized score)
		barLength := width - 1
		if maxScore > 0 {
			barLength = int((1.0 - avgScore/maxScore) * float64(width-1))
		}
		if barLength < 0 {
			barLength = 0
		}
		if barLength > width-1 {
			barLength = width - 1
		}

		// Check if this row contains the elbow
		containsElbow := false
		if elbowIndex >= startIdx && elbowIndex < endIdx {
			containsElbow = true
		}

		// Determine color
		style := tcell.StyleDefault.Foreground(tcell.ColorWhite)
		if containsElbow {
			style = tcell.StyleDefault.Foreground(tcell.ColorRed)
		}

		// Draw bar
		for x := 0; x < barLength && x < width-1; x++ {
			screen.SetContent(startX+x, y, '█', nil, style)
		}
	}
}

// renderMainDisplay renders the main detailed ranking display
func (r *Ranker) renderMainDisplay(screen tcell.Screen, rankings []traceDocument, round, trial int, startX, maxWidth, maxHeight int) {
	// Help text
	help := "Press Ctrl+C, Esc, or 'q' to quit"
	r.writeString(screen, startX, 0, help, tcell.StyleDefault.Foreground(tcell.ColorDarkGray))

	// Header
	header := fmt.Sprintf("Round %d | Trial %d | Items: %d", round, trial, len(rankings))
	r.writeString(screen, startX, 1, header, tcell.StyleDefault.Bold(true))

	// Calculate max score for normalization
	maxScore := 0.0
	if len(rankings) > 0 {
		maxScore = rankings[len(rankings)-1].Score
	}

	// Rankings (one per line, starting at row 3)
	for i, doc := range rankings {
		row := i + 3
		if row >= maxHeight {
			break // Don't render beyond screen height
		}

		// Calculate bar length proportional to score (invert so lower score = longer bar)
		barLength := maxWidth - 10 // Reserve space for score display
		if maxScore > 0 {
			barLength = int((1.0 - doc.Score/maxScore) * float64(maxWidth-10))
		}
		if barLength < 0 {
			barLength = 0
		}

		// Styles
		whiteStyle := tcell.StyleDefault.Foreground(tcell.ColorWhite)
		grayStyle := tcell.StyleDefault.Foreground(tcell.ColorGray)

		// Draw document text as the bar
		x := startX
		for _, ch := range doc.Value {
			if x-startX >= maxWidth-10 {
				break // Leave room for score
			}

			// White for bar portion, gray for overflow
			style := whiteStyle
			if x-startX >= barLength {
				style = grayStyle
			}
			screen.SetContent(x, row, ch, nil, style)
			x++
		}

		// Fill remaining bar space with '+' if text is shorter than bar
		for x-startX < barLength && x-startX < maxWidth-10 {
			screen.SetContent(x, row, '+', nil, whiteStyle)
			x++
		}

		// Show score at the end
		scoreLabel := fmt.Sprintf(" %.2f", doc.Score)
		r.writeString(screen, startX+maxWidth-9, row, scoreLabel, tcell.StyleDefault.Foreground(tcell.ColorYellow))
	}
}

func (r *Ranker) recordTrialState(trialNum int, trialsCompleted int, scores map[string][]float64, documents []document) error {
	// Early exit if neither feature is enabled
	if r.traceFile == nil && !r.cfg.Watch {
		return nil
	}

	// Calculate current rankings from accumulated scores
	var rankings []traceDocument
	for id, scoreList := range scores {
		var sum float64
		for _, score := range scoreList {
			sum += score
		}
		avgScore := sum / float64(len(scoreList))

		// Find the document value
		var value string
		for _, doc := range documents {
			if doc.ID == id {
				value = doc.Value
				break
			}
		}

		rankings = append(rankings, traceDocument{
			ID:    id,
			Value: value,
			Score: avgScore,
		})
	}

	// Sort by score (lower is better)
	sort.Slice(rankings, func(i, j int) bool {
		return rankings[i].Score < rankings[j].Score
	})

	// Build trace line
	trace := traceLine{
		Round:             r.round,
		Trial:             trialNum,
		TrialsCompleted:   trialsCompleted,
		TrialsRemaining:   r.cfg.NumTrials - trialsCompleted,
		TotalInputTokens:  r.totalUsage.InputTokens,
		TotalOutputTokens: r.totalUsage.OutputTokens,
		Rankings:          rankings,
	}

	// Add convergence info if enabled
	if r.cfg.EnableConvergence {
		r.mu.Lock()
		if len(r.elbowPositions) > 0 {
			lastElbow := r.elbowPositions[len(r.elbowPositions)-1]
			if lastElbow >= 0 {
				trace.ElbowPosition = &lastElbow
			}

			// Calculate stability count using shared logic
			stableCount, _ := r.countStableElbows(len(rankings))
			if stableCount > 0 {
				trace.StableTrialsCount = stableCount
			}
		}
		r.mu.Unlock()
	}

	// File writing - only if trace enabled
	if r.traceFile != nil {
		data, err := json.Marshal(trace)
		if err != nil {
			return fmt.Errorf("failed to marshal trace line: %w", err)
		}

		if _, err := r.traceFile.Write(append(data, '\n')); err != nil {
			return fmt.Errorf("failed to write trace line: %w", err)
		}

		// Flush to disk immediately
		if err := r.traceFile.Sync(); err != nil {
			return fmt.Errorf("failed to sync trace file: %w", err)
		}
	}

	// Visualization - only if watch enabled
	if r.cfg.Watch && r.screen != nil {
		r.renderVisualization(rankings, r.round, trialNum)
	}

	return nil
}
