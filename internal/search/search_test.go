package search

import (
	"encoding/binary"
	"math"
	"testing"

	"rinha-backend-2026/internal/artifact"
	"rinha-backend-2026/internal/ivf"
)

func TestSquaredDistance(t *testing.T) {
	zero := [14]int16{}
	got := SquaredDistance(&zero, &zero)
	if got != 0 {
		t.Errorf("equal vectors should give 0, got %d", got)
	}

	a := [14]int16{1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	b := [14]int16{3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	got = SquaredDistance(&a, &b)
	expected := int64(4 + 0 + 4)
	if got != expected {
		t.Errorf("got %d, want %d", got, expected)
	}

	c := [14]int16{artifact.SentinelInt16, artifact.SentinelInt16}
	d := [14]int16{0, 0}
	got = SquaredDistance(&c, &d)
	expectedDist := int64(32768*32768 + 32768*32768)
	if got != expectedDist {
		t.Errorf("got %d, want %d", got, expectedDist)
	}
}

func TestTopK(t *testing.T) {
	t.Run("insert 3 keeps sorted", func(t *testing.T) {
		var tk TopK
		tk.Insert(100, artifact.LabelFraud)
		tk.Insert(50, artifact.LabelLegit)
		tk.Insert(75, artifact.LabelFraud)

		sorted := tk.Sorted()
		if len(sorted) != 3 {
			t.Fatalf("expected 3 items, got %d", len(sorted))
		}
		if sorted[0].Dist != 50 || sorted[1].Dist != 75 || sorted[2].Dist != 100 {
			t.Errorf("wrong order: %v", sorted)
		}
	})

	t.Run("insert 10 returns top 5", func(t *testing.T) {
		var tk TopK
		dists := []int64{100, 30, 200, 10, 50, 80, 300, 40, 60, 20}
		labels := []uint8{1, 0, 1, 1, 0, 1, 0, 1, 0, 0}

		for i := range dists {
			tk.Insert(dists[i], labels[i])
		}

		sorted := tk.Sorted()
		if len(sorted) != 5 {
			t.Fatalf("expected 5 items, got %d", len(sorted))
		}
		if sorted[0].Dist != 10 || sorted[4].Dist != 50 {
			t.Errorf("expected top-5 [10,20,30,40,50], got distances: %v", sorted)
		}
		expectedDists := []int64{10, 20, 30, 40, 50}
		for i, d := range expectedDists {
			if sorted[i].Dist != d {
				t.Errorf("pos %d: got %d, want %d", i, sorted[i].Dist, d)
			}
		}
	})

	t.Run("fraud count", func(t *testing.T) {
		var tk TopK
		tk.Insert(10, artifact.LabelFraud)
		tk.Insert(20, artifact.LabelLegit)
		tk.Insert(30, artifact.LabelFraud)
		tk.Insert(40, artifact.LabelFraud)
		tk.Insert(50, artifact.LabelLegit)

		if tk.FraudCount() != 3 {
			t.Errorf("expected 3 frauds, got %d", tk.FraudCount())
		}
	})
}

func TestTopK_MaxDist(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		var tk TopK
		if tk.MaxDist() != math.MaxInt64 {
			t.Errorf("expected MaxInt64, got %d", tk.MaxDist())
		}
	})
	t.Run("partial", func(t *testing.T) {
		var tk TopK
		tk.Insert(10, artifact.LabelLegit)
		tk.Insert(20, artifact.LabelLegit)
		if tk.MaxDist() != math.MaxInt64 {
			t.Errorf("expected MaxInt64, got %d", tk.MaxDist())
		}
	})
	t.Run("full", func(t *testing.T) {
		var tk TopK
		tk.Insert(50, artifact.LabelLegit)
		tk.Insert(10, artifact.LabelLegit)
		tk.Insert(30, artifact.LabelLegit)
		tk.Insert(20, artifact.LabelLegit)
		tk.Insert(40, artifact.LabelLegit)
		if tk.MaxDist() != 50 {
			t.Errorf("expected 50, got %d", tk.MaxDist())
		}
	})
}

func TestDecide(t *testing.T) {
	tests := []struct {
		fraudCount int
		approved   bool
		fraudScore float64
	}{
		{0, true, 0.0},
		{2, true, 0.4},
		{3, false, 0.6},
		{5, false, 1.0},
	}
	for _, tt := range tests {
		approved, score := Decide(tt.fraudCount)
		if approved != tt.approved || score != tt.fraudScore {
			t.Errorf("fraudCount=%d: got approved=%v score=%.1f, want approved=%v score=%.1f",
				tt.fraudCount, approved, score, tt.approved, tt.fraudScore)
		}
	}
}

func buildIVF2Artifact(vectors []artifact.VectorRecord) []byte {
	const (
		headerSz       = 20
		centroidSz     = 28
		bboxSz         = 56
		clusterEntrySz = 8
	)
	K := 1
	N := len(vectors)
	centroidsSize := K * centroidSz
	bboxesSize := K * bboxSz
	clusterTableSize := K * clusterEntrySz
	vectorsSize := N * artifact.VectorRecordSz
	totalSize := headerSz + centroidsSize + bboxesSize + clusterTableSize + vectorsSize
	buf := make([]byte, totalSize)

	copy(buf[0:4], "IVF2")
	binary.LittleEndian.PutUint32(buf[4:], uint32(N))
	binary.LittleEndian.PutUint32(buf[8:], artifact.DimCount)
	binary.LittleEndian.PutUint32(buf[12:], uint32(K))
	binary.LittleEndian.PutUint16(buf[16:], artifact.FlagInt16Mask)

	centroidOff := headerSz
	for d := 0; d < 14; d++ {
		binary.LittleEndian.PutUint16(buf[centroidOff+d*2:], 0)
	}

	bboxOff := headerSz + centroidsSize
	for d := 0; d < 14; d++ {
		binary.LittleEndian.PutUint16(buf[bboxOff+d*2:], 32768)
		binary.LittleEndian.PutUint16(buf[bboxOff+28+d*2:], 32767)
	}

	clusterTableOff := headerSz + centroidsSize + bboxesSize
	vectorsOff := clusterTableOff + clusterTableSize
	binary.LittleEndian.PutUint32(buf[clusterTableOff:], uint32(vectorsOff))
	binary.LittleEndian.PutUint32(buf[clusterTableOff+4:], uint32(N))

	for i := range vectors {
		vecOff := vectorsOff + i*artifact.VectorRecordSz
		for d := 0; d < 14; d++ {
			binary.LittleEndian.PutUint16(buf[vecOff+d*2:], uint16(vectors[i].Dims[d]))
		}
		buf[vecOff+28] = vectors[i].Label
	}

	return buf
}

func TestEngineSearch(t *testing.T) {
	vecs := []artifact.VectorRecord{
		{Dims: [14]int16{1}, Label: artifact.LabelFraud},
		{Dims: [14]int16{2}, Label: artifact.LabelLegit},
		{Dims: [14]int16{3}, Label: artifact.LabelFraud},
		{Dims: [14]int16{4}, Label: artifact.LabelLegit},
		{Dims: [14]int16{5}, Label: artifact.LabelFraud},
		{Dims: [14]int16{6}, Label: artifact.LabelLegit},
	}
	data := buildIVF2Artifact(vecs)

	art, err := artifact.Load(data)
	if err != nil {
		t.Fatalf("load IVF2 artifact: %v", err)
	}

	cfg := ivf.Config{K: 1, NProbe: 1, RetryExtra: 0}
	engine := NewEngine(art, cfg)

	q := [14]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	tk, err := engine.Search(&q)
	if err != nil {
		t.Fatalf("search error: %v", err)
	}
	fc := tk.FraudCount()
	if fc != 3 {
		t.Errorf("fraud count: got %d, want 3", fc)
	}
	maxDist := tk.MaxDist()
	if maxDist <= 0 {
		t.Error("expected non-zero max dist")
	}
}
