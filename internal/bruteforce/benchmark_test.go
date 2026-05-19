package bruteforce

import (
	"encoding/binary"
	"math"
	"testing"

	"rinha-backend-2026/internal/artifact"
)

// benchTK always accepts — maxDist = MaxInt64, so no early exit.
type benchTK struct {
	count int
}

func (t *benchTK) Insert(dist int64, label uint8) {
	t.count++
}

func (t *benchTK) MaxDist() int64 {
	return math.MaxInt64
}

func (t *benchTK) FraudCount() int { return 0 }

func buildBenchVectors(n int) []byte {
	buf := make([]byte, n*artifact.VectorRecordSz)
	rng := uint32(42)
	for i := 0; i < n; i++ {
		for d := 0; d < 14; d++ {
			rng = rng*1103515245 + 12345
			val := int16((rng >> 16) & 0x7FFF)
			binary.LittleEndian.PutUint16(buf[i*artifact.VectorRecordSz+d*2:], uint16(val))
		}
		buf[i*artifact.VectorRecordSz+28] = artifact.LabelLegit
	}
	return buf
}

func BenchmarkSearchFlat_100K(b *testing.B) {
	vectors := buildBenchVectors(100000)
	q := [14]int16{2570, 5140, 7710, 10280, 12850, 15420, 17990, 20560, 23130, 25700, 28270, 30840, 32767, -32768}

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		var tk benchTK
		SearchFlat(&q, vectors, uint32(100000), &tk)
	}
}
