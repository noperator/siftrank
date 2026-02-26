package siftrank

import (
	"math"
	"sort"
)

// perpendicularDistance calculates perpendicular distance from point to line
func perpendicularDistance(x0, y0, x1, y1, x2, y2 float64) float64 {
	// Line from (x1, y1) to (x2, y2)
	// Point at (x0, y0)
	// Formula: |ax + by + c| / sqrt(a^2 + b^2)
	// where line is ax + by + c = 0

	numerator := math.Abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
	denominator := math.Sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1))

	if denominator == 0 {
		return 0
	}

	return numerator / denominator
}

// findElbowPerpendicular returns the index of the elbow using perpendicular distance method.
// Returns -1 if elbow cannot be determined (e.g., too few documents).
func findElbowPerpendicular(rankedDocs []RankedDocument) int {
	n := len(rankedDocs)

	// Need at least 3 documents to find an elbow
	if n < 3 {
		return -1
	}

	firstScore := rankedDocs[0].Score
	lastScore := rankedDocs[n-1].Score

	// If scores are identical (flat line), no elbow exists
	if firstScore == lastScore {
		return -1
	}

	maxDist := 0.0
	elbowIdx := -1

	// Check each document except first and last
	for i := 1; i < n-1; i++ {
		dist := perpendicularDistance(
			float64(i),
			rankedDocs[i].Score,
			0.0,
			firstScore,
			float64(n-1),
			lastScore,
		)

		if dist > maxDist {
			maxDist = dist
			elbowIdx = i
		}
	}

	return elbowIdx
}

// findElbow dispatches to the appropriate elbow detection method based on the method name.
// Valid methods: ElbowMethodCurvature (default), ElbowMethodPerpendicular.
func findElbow(rankedDocs []RankedDocument, method ElbowMethod) int {
	switch method {
	case ElbowMethodPerpendicular:
		return findElbowPerpendicular(rankedDocs)
	case ElbowMethodCurvature, "":
		return findElbowCurvature(rankedDocs)
	default:
		// Shouldn't happen if config validation is correct, but default to curvature
		return findElbowCurvature(rankedDocs)
	}
}

// gaussianSmooth applies 1D Gaussian smoothing to a slice of values.
// sigma controls the width of the Gaussian kernel.
func gaussianSmooth(data []float64, sigma float64) []float64 {
	n := len(data)
	if n == 0 {
		return data
	}

	// Kernel radius: typically 3*sigma is sufficient to capture 99.7% of the Gaussian
	radius := int(math.Ceil(3 * sigma))
	if radius < 1 {
		radius = 1
	}

	// Build Gaussian kernel
	kernelSize := 2*radius + 1
	kernel := make([]float64, kernelSize)
	var kernelSum float64
	for i := 0; i < kernelSize; i++ {
		x := float64(i - radius)
		kernel[i] = math.Exp(-(x * x) / (2 * sigma * sigma))
		kernelSum += kernel[i]
	}
	// Normalize kernel
	for i := range kernel {
		kernel[i] /= kernelSum
	}

	// Apply convolution with boundary handling (extend edge values)
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		var sum float64
		for k := 0; k < kernelSize; k++ {
			// Map kernel index to data index with boundary clamping
			dataIdx := i + (k - radius)
			if dataIdx < 0 {
				dataIdx = 0
			} else if dataIdx >= n {
				dataIdx = n - 1
			}
			sum += data[dataIdx] * kernel[k]
		}
		result[i] = sum
	}

	return result
}

// gradient calculates the numerical gradient (first derivative) of a slice.
// Uses central differences for interior points and forward/backward differences at edges.
func gradient(data []float64) []float64 {
	n := len(data)
	if n < 2 {
		return make([]float64, n)
	}

	result := make([]float64, n)

	// Forward difference at start
	result[0] = data[1] - data[0]

	// Central differences for interior
	for i := 1; i < n-1; i++ {
		result[i] = (data[i+1] - data[i-1]) / 2.0
	}

	// Backward difference at end
	result[n-1] = data[n-1] - data[n-2]

	return result
}

// findElbowCurvature returns the index of the elbow in a sorted list of ranked documents
// using curvature-based detection. It finds the point of maximum curvature
// (global minimum of 2nd derivative) which represents the transition from
// the steep "interesting" section to the flatter "tail" section.
// Returns -1 if elbow cannot be determined (e.g., too few documents or flat scores).
func findElbowCurvature(rankedDocs []RankedDocument) int {
	n := len(rankedDocs)

	// Need at least 4 documents to find an elbow meaningfully
	if n < 4 {
		return -1
	}

	// Extract scores
	scores := make([]float64, n)
	for i, doc := range rankedDocs {
		scores[i] = doc.Score
	}

	// Check if scores are flat (all identical within epsilon)
	const epsilon = 1e-9
	firstScore := scores[0]
	allFlat := true
	for _, score := range scores {
		if math.Abs(score-firstScore) > epsilon {
			allFlat = false
			break
		}
	}
	if allFlat {
		return -1
	}

	// Calculate sigma: 3% of dataset size, minimum 1.0
	sigma := math.Max(1.0, float64(n)*0.03)

	// Step 1: Smooth the scores
	smoothedScores := gaussianSmooth(scores, sigma)

	// Step 2: Calculate derivatives (cascade approach)
	// 1st derivative from smoothed scores
	firstDeriv := gradient(smoothedScores)
	// Smooth the 1st derivative
	smoothedFirstDeriv := gaussianSmooth(firstDeriv, sigma)
	// 2nd derivative from smoothed 1st derivative
	secondDeriv := gradient(smoothedFirstDeriv)
	// Smooth the 2nd derivative
	smoothedSecondDeriv := gaussianSmooth(secondDeriv, sigma)

	// Step 3: Find the global minimum of the 2nd derivative
	// This is the point of maximum curvature
	minVal := smoothedSecondDeriv[0]
	minIdx := 0
	for i := 1; i < n; i++ {
		if smoothedSecondDeriv[i] < minVal {
			minVal = smoothedSecondDeriv[i]
			minIdx = i
		}
	}

	return minIdx
}

// countStableElbows returns how many consecutive recent elbow positions are within tolerance
// Returns the count (1 to StableTrials) and the tolerance used
func (r *Ranker) countStableElbows(numDocuments int) (int, int) {
	n := len(r.elbowPositions)

	if n < 2 {
		return 0, 0
	}

	// Calculate tolerance in absolute terms (number of positions)
	tolerance := int(r.cfg.ElbowTolerance * float64(numDocuments))
	if tolerance < 1 {
		tolerance = 1 // At minimum, allow 1 position variance
	}

	// Check windows of increasing size to find largest that fits within tolerance
	stableCount := 1 // Current position always counts
	for windowSize := 2; windowSize <= r.cfg.StableTrials && windowSize <= n; windowSize++ {
		// Check if last 'windowSize' positions are all within tolerance
		start := n - windowSize
		minPos := r.elbowPositions[start]
		maxPos := r.elbowPositions[start]
		for i := start + 1; i < n; i++ {
			if r.elbowPositions[i] < minPos {
				minPos = r.elbowPositions[i]
			}
			if r.elbowPositions[i] > maxPos {
				maxPos = r.elbowPositions[i]
			}
		}

		if maxPos-minPos <= tolerance {
			stableCount = windowSize
		}
	}

	return stableCount, tolerance
}

// isElbowStable checks if recent elbow positions are within tolerance
// Returns (isStable, actualTolerance)
func (r *Ranker) isElbowStable(numDocuments int) (bool, int) {
	stableCount, tolerance := r.countStableElbows(numDocuments)
	return stableCount >= r.cfg.StableTrials, tolerance
}

// isRankingStable checks if the full ranking order has stabilized
// Returns (isStable, actualTrialsChecked)
func (r *Ranker) isRankingStable() (bool, int) {
	n := len(r.rankingOrders)

	// Need at least StableTrials to check
	if n < r.cfg.StableTrials {
		return false, 0
	}

	// Get the most recent ranking orders
	recentOrders := r.rankingOrders[n-r.cfg.StableTrials:]

	// Check if all recent orders are identical
	firstOrder := recentOrders[0]
	for i := 1; i < len(recentOrders); i++ {
		if len(recentOrders[i]) != len(firstOrder) {
			return false, len(recentOrders)
		}
		for j := range recentOrders[i] {
			if recentOrders[i][j] != firstOrder[j] {
				return false, len(recentOrders)
			}
		}
	}

	return true, len(recentOrders)
}

// hasConverged checks if the ranking has stabilized across trials
// Returns true if early stopping should occur
func (r *Ranker) hasConverged(scores map[string][]float64, completedTrialNum int) bool {
	// Early stopping disabled
	if !r.cfg.EnableConvergence {
		return false
	}

	// Check if we already detected convergence (quick exit)
	r.mu.Lock()
	alreadyConverged := r.converged
	r.mu.Unlock()

	if alreadyConverged {
		// Convergence already detected by another trial
		// Still return true to respect the cancellation, but don't log again
		return true
	}

	// Not enough trials yet
	if completedTrialNum < r.cfg.MinTrials {
		return false
	}

	// Calculate current rankings based on scores so far
	var currentRankings []RankedDocument
	for id, scoreList := range scores {
		var sum float64
		for _, score := range scoreList {
			sum += score
		}
		avgScore := sum / float64(len(scoreList))

		currentRankings = append(currentRankings, RankedDocument{
			Key:   id,
			Score: avgScore,
		})
	}

	// Sort by score (lower is better)
	sort.Slice(currentRankings, func(i, j int) bool {
		return currentRankings[i].Score < currentRankings[j].Score
	})

	// Store the ranking order for this trial
	rankingOrder := make([]string, len(currentRankings))
	for i, doc := range currentRankings {
		rankingOrder[i] = doc.Key
	}

	r.mu.Lock()
	r.rankingOrders = append(r.rankingOrders, rankingOrder)
	r.mu.Unlock()

	// Check criterion 1: Elbow stabilization
	var elbowStable bool
	var actualTolerance int

	elbowPos := findElbow(currentRankings, r.cfg.ElbowMethod)
	if elbowPos != -1 {
		// Store elbow position
		r.mu.Lock()
		r.elbowPositions = append(r.elbowPositions, elbowPos)
		r.mu.Unlock()

		r.cfg.Logger.Debug("Elbow detected",
			"trial", completedTrialNum,
			"position", elbowPos,
			"total_docs", len(currentRankings))

		elbowStable, actualTolerance = r.isElbowStable(len(currentRankings))
	} else {
		r.cfg.Logger.Debug("No elbow found in trial", "trial", completedTrialNum)
	}

	// Check criterion 2: Ranking order stabilization
	rankingStable, trialsChecked := r.isRankingStable()

	if rankingStable {
		r.cfg.Logger.Debug("Ranking order stable",
			"trial", completedTrialNum,
			"trials_checked", trialsChecked,
			"total_docs", len(currentRankings))
	}

	// Converge if EITHER criterion is met
	stable := elbowStable || rankingStable

	if stable {
		// Acquire lock to set convergence flag
		r.mu.Lock()
		// Double-check we haven't converged in the meantime (race condition)
		if !r.converged {
			r.converged = true

			// Use len(r.rankingOrders) for all convergence types
			// This shows how many trials were evaluated for convergence
			trialsEvaluated := len(r.rankingOrders)

			// Log which criterion triggered
			if elbowStable && rankingStable {
				r.cfg.Logger.Info("Convergence: elbow and ranking both stabilized",
					"round", r.round,
					"trials_evaluated", trialsEvaluated,
					"recent_elbow_positions", r.elbowPositions[len(r.elbowPositions)-r.cfg.StableTrials:],
					"elbow_tolerance", actualTolerance)
			} else if elbowStable {
				r.cfg.Logger.Info("Convergence: elbow position stabilized",
					"round", r.round,
					"trials_evaluated", trialsEvaluated,
					"recent_positions", r.elbowPositions[len(r.elbowPositions)-r.cfg.StableTrials:],
					"tolerance", actualTolerance)
			} else {
				r.cfg.Logger.Info("Convergence: ranking order stabilized",
					"round", r.round,
					"trials_evaluated", trialsEvaluated,
					"trials_checked", trialsChecked,
					"total_docs", len(currentRankings))
			}
		}
		r.mu.Unlock()
	}

	return stable
}

// setElbowCutoff determines the cutoff position for refinement based on elbow detection
func (r *Ranker) setElbowCutoff(numDocuments int) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !r.cfg.EnableConvergence {
		// Convergence disabled - cutoff won't be used
		r.elbowCutoff = -1
		return
	}

	// Check if we converged via ranking stability only (no valid elbows found)
	hasValidElbow := false
	for _, pos := range r.elbowPositions {
		if pos > 0 {
			hasValidElbow = true
			break
		}
	}

	if !hasValidElbow {
		// Either no elbows tracked OR all were -1
		// This means we converged via ranking stability (or didn't converge)
		// Don't refine - the ranking is stable as-is
		r.elbowCutoff = -1
		if r.converged {
			r.cfg.Logger.Debug("Ranking stabilized without elbow, skipping refinement",
				"round", r.round,
				"total_docs", numDocuments)
		}
		return
	}

	if r.converged {
		// Converged via elbow stability - use the last recorded elbow position
		r.elbowCutoff = r.elbowPositions[len(r.elbowPositions)-1]
		r.cfg.Logger.Debug("Using converged elbow position for refinement",
			"round", r.round,
			"cutoff", r.elbowCutoff,
			"total_docs", numDocuments)
	} else {
		// Max trials reached without convergence - use median of recent positions
		recentCount := r.cfg.StableTrials
		if recentCount > len(r.elbowPositions) {
			recentCount = len(r.elbowPositions)
		}
		recentPositions := r.elbowPositions[len(r.elbowPositions)-recentCount:]

		// Calculate median
		sortedPositions := make([]int, len(recentPositions))
		copy(sortedPositions, recentPositions)
		sort.Ints(sortedPositions)

		median := sortedPositions[len(sortedPositions)/2]
		r.elbowCutoff = median

		r.cfg.Logger.Debug("Using median elbow position for refinement",
			"round", r.round,
			"cutoff", r.elbowCutoff,
			"recent_positions", recentPositions,
			"total_docs", numDocuments)
	}
}
