package search

import (
	"encoding/binary"
	"testing"

	"rinha-backend-2026/internal/artifact"
	"rinha-backend-2026/internal/ivf"
)

func BenchmarkSquaredDistance(b *testing.B) {
	q := [14]int16{2570, 5140, 7710, 10280, 12850, 15420, 17990, 20560, 23130, 25700, 28270, 30840, 32767, -32768}
	r := [14]int16{1285, 3855, 6425, 8995, 11565, 14135, 16705, 19275, 21845, 24415, 26985, 29555, 0, 0}

	b.ReportAllocs()
	for b.Loop() {
		SquaredDistance(&q, &r)
	}
}

func BenchmarkTopKInsert(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		var tk TopK
		tk.Insert(100, artifact.LabelFraud)
		tk.Insert(30, artifact.LabelLegit)
		tk.Insert(200, artifact.LabelFraud)
		tk.Insert(10, artifact.LabelLegit)
		tk.Insert(50, artifact.LabelFraud)
		tk.Insert(80, artifact.LabelLegit)
		tk.Insert(300, artifact.LabelFraud)
		tk.Insert(40, artifact.LabelLegit)
		tk.Insert(60, artifact.LabelFraud)
		tk.Insert(20, artifact.LabelLegit)
	}
}

func BenchmarkDecide(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		Decide(3)
	}
}

func BenchmarkSearch(b *testing.B) {
	art, err := artifact.LoadFromFile("../artifact/artifact.bin")
	if err != nil {
		b.Skipf("artifact load failed (may need regeneration): %v", err)
	}
	engine := NewEngine(art, ivf.Config{K: int(art.NumClusters()), NProbe: 8, RetryExtra: 8})

	q := [14]int16{2570, 5140, 7710, 10280, 12850, 15420, 17990, 20560, 23130, 25700, 28270, 30840, 32767, -32768}

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		engine.Search(&q)
	}
}

func buildBenchArtifactIVF2(n int) ([]byte, uint32, int) {
	const (
		headerSz       = 20
		centroidSz     = 28
		bboxSz         = 56
		clusterEntrySz = 8
	)
	K := 16
	centroidsSize := K * centroidSz
	bboxesSize := K * bboxSz
	clusterTableSize := K * clusterEntrySz
	vectorsSize := n * artifact.VectorRecordSz
	totalSize := headerSz + centroidsSize + bboxesSize + clusterTableSize + vectorsSize
	buf := make([]byte, totalSize)

	copy(buf[0:4], "IVF2")
	binary.LittleEndian.PutUint32(buf[4:], uint32(n))
	binary.LittleEndian.PutUint32(buf[8:], artifact.DimCount)
	binary.LittleEndian.PutUint32(buf[12:], uint32(K))
	binary.LittleEndian.PutUint16(buf[16:], artifact.FlagInt16Mask)

	centroidOff := headerSz
	for k := 0; k < K; k++ {
		for d := 0; d < 14; d++ {
			binary.LittleEndian.PutUint16(buf[centroidOff+k*centroidSz+d*2:], 0)
		}
	}

	bboxOff := headerSz + centroidsSize
	for k := 0; k < K; k++ {
		for d := 0; d < 14; d++ {
			binary.LittleEndian.PutUint16(buf[bboxOff+k*bboxSz+d*2:], 32768)
			binary.LittleEndian.PutUint16(buf[bboxOff+k*bboxSz+28+d*2:], 32767)
		}
	}

	clusterTableOff := headerSz + centroidsSize + bboxesSize
	vectorsOff := clusterTableOff + clusterTableSize
	binary.LittleEndian.PutUint32(buf[clusterTableOff:], uint32(vectorsOff))
	binary.LittleEndian.PutUint32(buf[clusterTableOff+4:], uint32(n))

	rng := uint32(42)
	for i := 0; i < n; i++ {
		vecOff := vectorsOff + i*artifact.VectorRecordSz
		for d := 0; d < 14; d++ {
			rng = rng*1103515245 + 12345
			val := int16((rng >> 16) & 0x7FFF)
			binary.LittleEndian.PutUint16(buf[vecOff+d*2:], uint16(val))
		}
		buf[vecOff+28] = artifact.LabelLegit
	}

	return buf, uint32(n), K
}

func BenchmarkSearch_Synthetic100K(b *testing.B) {
	data, _, K := buildBenchArtifactIVF2(100000)
	art, err := artifact.Load(data)
	if err != nil {
		b.Fatalf("load artifact: %v", err)
	}
	engine := NewEngine(art, ivf.Config{K: K, NProbe: 8, RetryExtra: 8})

	q := [14]int16{2570, 5140, 7710, 10280, 12850, 15420, 17990, 20560, 23130, 25700, 28270, 30840, 32767, -32768}

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		engine.Search(&q)
	}
}
