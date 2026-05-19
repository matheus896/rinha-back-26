package main

import (
	"math"
	"testing"
)

func TestKMeansConverges(t *testing.T) {
	// 3 clusters in 2D space
	vectors := [][]float64{
		// cluster 0 around (0, 0)
		{0, 0}, {1, 0}, {0, 1}, {1, 1}, {-1, 0}, {0, -1},
		// cluster 1 around (10, 10)
		{10, 10}, {11, 10}, {10, 11}, {11, 11}, {9, 10}, {10, 9},
		// cluster 2 around (-10, 10)
		{-10, 10}, {-11, 10}, {-10, 11}, {-11, 11}, {-9, 10}, {-10, 9},
	}

	centroids, assignments := Train(vectors, 3, 30, len(vectors), 0, 42)

	if len(centroids) != 3 {
		t.Fatalf("expected 3 centroids, got %d", len(centroids))
	}
	if len(assignments) != len(vectors) {
		t.Fatalf("expected %d assignments, got %d", len(vectors), len(assignments))
	}

	// no empty clusters
	clusterCounts := make([]int, 3)
	for _, a := range assignments {
		clusterCounts[a]++
	}
	for i, c := range clusterCounts {
		if c == 0 {
			t.Fatalf("cluster %d is empty", i)
		}
	}

	// all points assigned to a valid cluster
	for i, a := range assignments {
		if a < 0 || a >= 3 {
			t.Fatalf("assignment %d out of range: %d", i, a)
		}
	}

	// centroids should be near their cluster centers
	// sort centroids by x+y for stable comparison
	type cidx struct {
		idx int
		sum float64
	}
	cis := make([]cidx, 3)
	for i, c := range centroids {
		cis[i] = cidx{i, c[0] + c[1]}
	}
	// simple bubble sort by sum
	for i := 0; i < 3; i++ {
		for j := i + 1; j < 3; j++ {
			if cis[j].sum < cis[i].sum {
				cis[i], cis[j] = cis[j], cis[i]
			}
		}
	}

	// cluster around (-10,10) has lowest sum
	c0 := centroids[cis[0].idx]
	if math.Abs(c0[0]+10) > 2 || math.Abs(c0[1]-10) > 2 {
		t.Errorf("centroid 0 too far from (-10,10): got (%.2f, %.2f)", c0[0], c0[1])
	}
	// cluster around (0,0) has middle sum
	c1 := centroids[cis[1].idx]
	if math.Abs(c1[0]) > 2 || math.Abs(c1[1]) > 2 {
		t.Errorf("centroid 1 too far from (0,0): got (%.2f, %.2f)", c1[0], c1[1])
	}
	// cluster around (10,10) has highest sum
	c2 := centroids[cis[2].idx]
	if math.Abs(c2[0]-10) > 2 || math.Abs(c2[1]-10) > 2 {
		t.Errorf("centroid 2 too far from (10,10): got (%.2f, %.2f)", c2[0], c2[1])
	}
}

func TestKMeansDeterministic(t *testing.T) {
	vectors := [][]float64{
		{0, 0}, {1, 0}, {0, 1},
		{10, 10}, {11, 10}, {10, 11},
	}

	c1, a1 := Train(vectors, 2, 20, len(vectors), 0, 42)
	c2, a2 := Train(vectors, 2, 20, len(vectors), 0, 42)

	for i := range c1 {
		for d := range c1[i] {
			if c1[i][d] != c2[i][d] {
				t.Fatalf("centroids differ at [%d][%d]: %.6f vs %.6f", i, d, c1[i][d], c2[i][d])
			}
		}
	}
	for i := range a1 {
		if a1[i] != a2[i] {
			t.Fatalf("assignments differ at %d: %d vs %d", i, a1[i], a2[i])
		}
	}
}

func TestKMeansParallelMatchesSequential(t *testing.T) {
	vectors := [][]float64{
		{0, 0}, {1, 0}, {0, 1},
		{10, 10}, {11, 10}, {10, 11},
		{20, 20}, {21, 20}, {20, 21},
	}

	centroids, _ := Train(vectors, 3, 10, len(vectors), 0, 42)

	seq := AssignClustersParallel(vectors, centroids, 1)
	par := AssignClustersParallel(vectors, centroids, 4)

	for i := range seq {
		if seq[i] != par[i] {
			t.Fatalf("parallel differs from sequential at %d: %d vs %d", i, seq[i], par[i])
		}
	}
}
