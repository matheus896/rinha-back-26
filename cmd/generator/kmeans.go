package main

import (
	"math"
	"math/rand/v2"
	"runtime"
	"sync"
)

// InitCentroidsKMeansPP performs k-means++ initialization.
// The first centroid is chosen uniformly at random; subsequent centroids
// are chosen with probability proportional to squared distance to the
// nearest existing centroid.
func InitCentroidsKMeansPP(vectors [][]float64, K int, seed int64) [][]float64 {
	rng := rand.New(rand.NewPCG(uint64(seed), uint64(seed)))
	return initCentroidsKMeansPP(vectors, K, rng)
}

func initCentroidsKMeansPP(vectors [][]float64, K int, rng *rand.Rand) [][]float64 {
	N := len(vectors)
	if K >= N {
		// Each vector is its own centroid
		centroids := make([][]float64, N)
		for i := range vectors {
			centroids[i] = make([]float64, len(vectors[i]))
			copy(centroids[i], vectors[i])
		}
		return centroids
	}

	centroids := make([][]float64, K)
	for i := range centroids {
		centroids[i] = make([]float64, len(vectors[0]))
	}

	// First centroid: random
	first := rng.IntN(N)
	copy(centroids[0], vectors[first])

	// Distances from each vector to its nearest centroid
	dists := make([]float64, N)
	for i := range dists {
		dists[i] = math.MaxFloat64
	}

	for k := 1; k < K; k++ {
		// Update distances to nearest centroid
		for i := 0; i < N; i++ {
			d := squaredDistanceFloat64(vectors[i], centroids[k-1])
			if d < dists[i] {
				dists[i] = d
			}
		}

		// Weighted random selection
		var sum float64
		for _, d := range dists {
			sum += d
		}
		if sum == 0 {
			// All remaining points are identical to existing centroids
			// Pick uniformly at random among points not yet chosen as centroids
			idx := rng.IntN(N)
			copy(centroids[k], vectors[idx])
			continue
		}

		target := rng.Float64() * sum
		var cumulative float64
		chosen := 0
		for i, d := range dists {
			cumulative += d
			if cumulative >= target {
				chosen = i
				break
			}
		}
		copy(centroids[k], vectors[chosen])
	}

	return centroids
}

// AssignClustersParallel assigns each vector to its nearest centroid using
// up to numWorkers goroutines. If numWorkers <= 1, it runs sequentially.
func AssignClustersParallel(vectors [][]float64, centroids [][]float64, numWorkers int) []int {
	N := len(vectors)
	assignments := make([]int, N)
	if N == 0 {
		return assignments
	}

	if numWorkers <= 1 {
		for i := 0; i < N; i++ {
			assignments[i] = nearestCentroid(vectors[i], centroids)
		}
		return assignments
	}

	if numWorkers > N {
		numWorkers = N
	}

	chunkSize := (N + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if start >= N {
			break
		}
		if end > N {
			end = N
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				assignments[i] = nearestCentroid(vectors[i], centroids)
			}
		}(start, end)
	}
	wg.Wait()
	return assignments
}

// UpdateCentroids recomputes centroid positions as the mean of assigned vectors.
// Empty clusters keep their previous centroid.
func UpdateCentroids(vectors [][]float64, assignments []int, K int) [][]float64 {
	if K == 0 || len(vectors) == 0 {
		return nil
	}
	dim := len(vectors[0])
	centroids := make([][]float64, K)
	counts := make([]int, K)
	sums := make([][]float64, K)
	for k := 0; k < K; k++ {
		centroids[k] = make([]float64, dim)
		sums[k] = make([]float64, dim)
	}

	for i, a := range assignments {
		if a < 0 || a >= K {
			continue
		}
		counts[a]++
		for d := 0; d < dim; d++ {
			sums[a][d] += vectors[i][d]
		}
	}

	for k := 0; k < K; k++ {
		if counts[k] > 0 {
			for d := 0; d < dim; d++ {
				centroids[k][d] = sums[k][d] / float64(counts[k])
			}
		}
	}

	return centroids
}

// Train runs k-means++ initialization and Lloyd iterations on a sample,
// then performs refines on the full dataset. It uses 1 restart and returns
// the best result by inertia.
func Train(vectors [][]float64, K, iters, sampleSize, refines int, seed int64) ([][]float64, []int) {
	if len(vectors) == 0 || K <= 0 {
		return nil, nil
	}

	// Build sample if requested and beneficial
	trainSet := vectors
	if sampleSize > 0 && sampleSize < len(vectors) {
		trainSet = sampleVectors(vectors, sampleSize, seed)
	}

	// Run twice (1 restart) and pick best inertia
	rng1 := rand.New(rand.NewPCG(uint64(seed), uint64(seed)))
	c1, a1, in1 := trainOnce(trainSet, K, iters, rng1)

	rng2 := rand.New(rand.NewPCG(uint64(seed)+1, uint64(seed)+1))
	c2, a2, in2 := trainOnce(trainSet, K, iters, rng2)

	var bestCentroids [][]float64
	if in1 <= in2 {
		bestCentroids = c1
		_ = a1
	} else {
		bestCentroids = c2
		_ = a2
	}

	// Refine on full dataset
	numWorkers := runtime.GOMAXPROCS(0)
	finalAssignments := AssignClustersParallel(vectors, bestCentroids, numWorkers)
	for r := 0; r < refines; r++ {
		bestCentroids = UpdateCentroids(vectors, finalAssignments, K)
		finalAssignments = AssignClustersParallel(vectors, bestCentroids, numWorkers)
	}

	return bestCentroids, finalAssignments
}

func trainOnce(vectors [][]float64, K, iters int, rng *rand.Rand) (centroids [][]float64, assignments []int, inertia float64) {
	centroids = initCentroidsKMeansPP(vectors, K, rng)
	for i := 0; i < iters; i++ {
		assignments = AssignClustersParallel(vectors, centroids, runtime.GOMAXPROCS(0))
		centroids = UpdateCentroids(vectors, assignments, K)
	}
	// Final assignment
	assignments = AssignClustersParallel(vectors, centroids, runtime.GOMAXPROCS(0))
	inertia = computeInertia(vectors, centroids, assignments)
	return
}

func nearestCentroid(vec []float64, centroids [][]float64) int {
	best := 0
	bestDist := squaredDistanceFloat64(vec, centroids[0])
	for i := 1; i < len(centroids); i++ {
		d := squaredDistanceFloat64(vec, centroids[i])
		if d < bestDist {
			bestDist = d
			best = i
		}
	}
	return best
}

func squaredDistanceFloat64(a, b []float64) float64 {
	var sum float64
	for i := 0; i < len(a) && i < len(b); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

func computeInertia(vectors [][]float64, centroids [][]float64, assignments []int) float64 {
	var sum float64
	for i, a := range assignments {
		sum += squaredDistanceFloat64(vectors[i], centroids[a])
	}
	return sum
}

func sampleVectors(vectors [][]float64, n int, seed int64) [][]float64 {
	if n >= len(vectors) {
		return vectors
	}
	// Fisher-Yates shuffle on indices
	indices := make([]int, len(vectors))
	for i := range indices {
		indices[i] = i
	}
	rng := rand.New(rand.NewPCG(uint64(seed), 0))
	for i := len(indices) - 1; i > 0; i-- {
		j := rng.IntN(i + 1)
		indices[i], indices[j] = indices[j], indices[i]
	}
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = vectors[indices[i]]
	}
	return out
}
