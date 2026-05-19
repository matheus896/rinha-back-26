package ivf

import (
	"math"
	"testing"
)

func TestRankCentroids(t *testing.T) {
	// 4 centroids in 14D (only first 2 dims matter)
	centroids := []int16{
		// c0 at (0, 0)
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		// c1 at (10, 0)
		10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		// c2 at (0, 10)
		0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		// c3 at (10, 10)
		10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}
	K := len(centroids) / 14

	q := [14]int16{0, 0}
	dists := make([]CentroidDist, K)
	ranked := make([]int, 2)
	RankCentroids(&q, centroids, dists)
	selectTopN(dists, ranked, 2)
	if ranked[0] != 0 {
		t.Errorf("expected closest centroid 0, got %d", ranked[0])
	}
	if ranked[1] != 1 && ranked[1] != 2 {
		t.Errorf("expected second closest 1 or 2, got %d", ranked[1])
	}

	// Query near c3
	q = [14]int16{10, 10}
	RankCentroids(&q, centroids, dists)
	selectTopN(dists, ranked, 1)
	if ranked[0] != 3 {
		t.Errorf("expected closest centroid 3, got %d", ranked[0])
	}
}

func TestBboxPrune(t *testing.T) {
	bboxMin := []int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	bboxMax := []int16{10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10}

	// Query inside bbox: min possible dist is 0, should pass (return true)
	q := [14]int16{5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5}
	if !BboxPrune(&q, bboxMin, bboxMax, 100) {
		t.Error("query inside bbox should not be pruned")
	}

	// Query far outside: min dist > maxDist, should be pruned (return false)
	q = [14]int16{100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	if BboxPrune(&q, bboxMin, bboxMax, 100) {
		t.Error("query far outside bbox should be pruned")
	}

	// Query on edge: min dist is 0, should pass
	q = [14]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	if !BboxPrune(&q, bboxMin, bboxMax, 1) {
		t.Error("query on bbox edge should not be pruned")
	}
}

func TestSquaredDistanceEarlyExit(t *testing.T) {
	a := [14]int16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
	b := [14]int16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
	c := [14]int16{100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100}

	// Identical vectors: exact distance should be 0
	exact := refSquaredDistance(&a, &b)
	early := SquaredDistanceEarlyExit(&a, &b, math.MaxInt64)
	if early != exact {
		t.Errorf("early exit mismatch for identical: exact=%d early=%d", exact, early)
	}

	// Nearby vectors: exact distance should match
	b = [14]int16{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	exact = refSquaredDistance(&a, &b)
	early = SquaredDistanceEarlyExit(&a, &b, math.MaxInt64)
	if early != exact {
		t.Errorf("early exit mismatch for nearby: exact=%d early=%d", exact, early)
	}

	// Distant vectors with low maxDist: should exit early with value >= maxDist
	maxDist := int64(100)
	early = SquaredDistanceEarlyExit(&a, &c, maxDist)
	if early < maxDist {
		t.Errorf("early exit should return >= maxDist for distant vectors: got %d, maxDist=%d", early, maxDist)
	}
}

type testSink struct {
	items [5]struct {
		dist  int64
		label uint8
	}
	count int
}

func (ts *testSink) Insert(dist int64, label uint8) {
	pos := 0
	for pos < ts.count && pos < 5 && ts.items[pos].dist < dist {
		pos++
	}
	if pos == 5 {
		return
	}
	if ts.count == 5 {
		for i := 4; i > pos; i-- {
			ts.items[i] = ts.items[i-1]
		}
	} else {
		for i := ts.count; i > pos; i-- {
			ts.items[i] = ts.items[i-1]
		}
		ts.count++
	}
	ts.items[pos] = struct {
		dist  int64
		label uint8
	}{dist: dist, label: label}
}

func (ts *testSink) MaxDist() int64 {
	if ts.count < 5 {
		return math.MaxInt64
	}
	return ts.items[ts.count-1].dist
}

func (ts *testSink) FraudCount() int {
	n := 0
	for i := 0; i < ts.count; i++ {
		if ts.items[i].label == 1 {
			n++
		}
	}
	return n
}

func refSquaredDistance(q, r *[14]int16) int64 {
	var sum int64
	for i := 0; i < 14; i++ {
		diff := int64(q[i]) - int64(r[i])
		sum += diff * diff
	}
	return sum
}

func BenchmarkSquaredDistanceEarlyExit(b *testing.B) {
	q := [14]int16{2570, 5140, 7710, 10280, 12850, 15420, 17990, 20560, 23130, 25700, 28270, 30840, 32767, -32768}
	r := [14]int16{1285, 3855, 6425, 8995, 11565, 14135, 16705, 19275, 21845, 24415, 26985, 29555, 0, 0}
	b.ReportAllocs()
	for b.Loop() {
		SquaredDistanceEarlyExit(&q, &r, 1<<62)
	}
}

func BenchmarkScanCluster(b *testing.B) {
	const n = 5000
	vectors := make([]byte, n*30)
	rng := uint32(42)
	for i := 0; i < n; i++ {
		off := i * 30
		for d := 0; d < 14; d++ {
			rng = rng*1103515245 + 12345
			val := uint16((rng >> 16) & 0x7FFF)
			vectors[off+d*2] = byte(val)
			vectors[off+d*2+1] = byte(val >> 8)
		}
		vectors[off+28] = byte(i & 1)
	}
	q := [14]int16{2570, 5140, 7710, 10280, 12850, 15420, 17990, 20560, 23130, 25700, 28270, 30840, 32767, -32768}
	b.ReportAllocs()
	for b.Loop() {
		tk := &testSink{}
		ScanCluster(&q, vectors, 0, n, 1<<62, tk)
	}
}

func BenchmarkScanCluster_Uint16Decode(b *testing.B) {
	data := make([]byte, 5000*30)
	rng := uint32(42)
	for i := 0; i < 5000; i++ {
		off := i * 30
		for d := 0; d < 14; d++ {
			rng = rng*1103515245 + 12345
			val := uint16((rng >> 16) & 0x7FFF)
			data[off+d*2] = byte(val)
			data[off+d*2+1] = byte(val >> 8)
		}
	}
	b.ReportAllocs()
	for b.Loop() {
		var sum int16
		for i := 0; i < 5000*14; i++ {
			off := i * 2
			sum += int16(data[off]) | int16(data[off+1])<<8
		}
		_ = sum
	}
}

func TestScanCluster(t *testing.T) {
	// Build 3 vectors: 2 legit, 1 fraud
	// VectorRecord = 14 int16 LE + label + padding = 30 bytes
	vectors := make([]byte, 3*30)
	for i := 0; i < 3; i++ {
		base := i * 30
		for d := 0; d < 14; d++ {
			// Put dimension values: vector i has value i in all dims
			vectors[base+d*2] = byte(i)
			vectors[base+d*2+1] = 0
		}
		if i == 2 {
			vectors[base+28] = 1 // fraud
		} else {
			vectors[base+28] = 0 // legit
		}
	}

	// Query matches vector 0 exactly
	q := [14]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	tk := &testSink{}
	ScanCluster(&q, vectors, 0, 3, math.MaxInt64, tk)

	if tk.count != 3 {
		t.Fatalf("expected 3 items in topk, got %d", tk.count)
	}
	// Closest should be vector 0 (dist 0)
	if tk.items[0].dist != 0 || tk.items[0].label != 0 {
		t.Errorf("expected first item dist=0 label=0, got dist=%d label=%d", tk.items[0].dist, tk.items[0].label)
	}
	// Furthest should be vector 2 (dist = 14 * 2^2 = 56)
	if tk.items[2].dist != 56 || tk.items[2].label != 1 {
		t.Errorf("expected last item dist=56 label=1, got dist=%d label=%d", tk.items[2].dist, tk.items[2].label)
	}
}
