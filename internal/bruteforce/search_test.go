package bruteforce

import (
	"encoding/binary"
	"math"
	"testing"

	"rinha-backend-2026/internal/artifact"
)

// testTK is a simple TopK implementation for testing.
type testTK struct {
	dists  []int64
	labels []uint8
}

func (t *testTK) Insert(dist int64, label uint8) {
	t.dists = append(t.dists, dist)
	t.labels = append(t.labels, label)
}

func (t *testTK) MaxDist() int64 {
	return math.MaxInt64
}

func (t *testTK) FraudCount() int {
	n := 0
	for _, l := range t.labels {
		if l == artifact.LabelFraud {
			n++
		}
	}
	return n
}

func buildTestVectors(vecs []artifact.VectorRecord) []byte {
	buf := make([]byte, len(vecs)*artifact.VectorRecordSz)
	for i := range vecs {
		for d := 0; d < 14; d++ {
			binary.LittleEndian.PutUint16(buf[i*artifact.VectorRecordSz+d*2:], uint16(vecs[i].Dims[d]))
		}
		buf[i*artifact.VectorRecordSz+28] = vecs[i].Label
	}
	return buf
}

func TestBruteForceSearch(t *testing.T) {
	// Create vectors with known distances from query [0,0,...0]
	// v0: [1,0,...] dist=1, fraud
	// v1: [2,0,...] dist=4, legit
	// v2: [3,0,...] dist=9, fraud
	// v3: [4,0,...] dist=16, legit
	vecs := []artifact.VectorRecord{
		{Dims: [14]int16{1}, Label: artifact.LabelFraud},
		{Dims: [14]int16{2}, Label: artifact.LabelLegit},
		{Dims: [14]int16{3}, Label: artifact.LabelFraud},
		{Dims: [14]int16{4}, Label: artifact.LabelLegit},
	}
	data := buildTestVectors(vecs)
	query := [14]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

	var tk testTK
	SearchFlat(&query, data, uint32(len(vecs)), &tk)

	if len(tk.dists) != 4 {
		t.Fatalf("expected 4 results, got %d", len(tk.dists))
	}

	// Should be ordered by distance
	expectedDists := []int64{1, 4, 9, 16}
	for i, d := range expectedDists {
		if tk.dists[i] != d {
			t.Errorf("pos %d: dist=%d, want=%d", i, tk.dists[i], d)
		}
	}
	if tk.FraudCount() != 2 {
		t.Errorf("fraud count: got %d, want 2", tk.FraudCount())
	}
}

func TestBruteForceSearch_EarlyExit(t *testing.T) {
	// v0: [32767,0,...] dist = 32767² ≈ 1.07B
	// v1: [1,0,...] dist = 1
	vecs := []artifact.VectorRecord{
		{Dims: [14]int16{32767}, Label: artifact.LabelFraud},
		{Dims: [14]int16{1}, Label: artifact.LabelLegit},
	}
	data := buildTestVectors(vecs)
	query := [14]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

	var tk testTK
	SearchFlat(&query, data, uint32(len(vecs)), &tk)

	if tk.FraudCount() != 1 {
		t.Errorf("fraud count: got %d, want 1", tk.FraudCount())
	}
}
