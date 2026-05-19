package ivf

import (
	"math"
	"math/rand/v2"
	"testing"
	"unsafe"
)

func TestDistBlockDiff_MatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 42))
	q := &[14]int16{}
	for d := 0; d < 14; d++ {
		q[d] = int16(rng.IntN(65536) - 32768)
	}

	numBlocks := 3
	blocks := make([]byte, numBlocks*240)
	for b := 0; b < numBlocks; b++ {
		blockBase := b * 240
		for d := 0; d < 14; d++ {
			dimOff := blockBase + d*16
			for v := 0; v < 8; v++ {
				val := int16(rng.IntN(65536) - 32768)
				blocks[dimOff+v*2] = byte(val)
				blocks[dimOff+v*2+1] = byte(val >> 8)
			}
		}
	}

	for block := 0; block < numBlocks; block++ {
		blockOff := uint32(block * 240)
		for dim := 0; dim < 14; dim++ {
			var avx [8]int32
			distBlockDiff(q, unsafe.Pointer(&blocks[0]), blockOff, dim, &avx)

			dimOff := blockOff + uint32(dim)*16
			dimVals := (*[8]int16)(unsafe.Pointer(&blocks[dimOff]))
			qd := int32(q[dim])
			for v := 0; v < 8; v++ {
				expected := qd - int32(dimVals[v])
				if avx[v] != expected {
					t.Errorf("block=%d dim=%d v=%d: AVX2=%d expected=%d",
						block, dim, v, avx[v], expected)
				}
			}
		}
	}
}

func TestDistBlockFused_MatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewPCG(99, 99))
	writeVI16 := func(b []byte, v int, val int16) {
		b[v*2] = byte(val)
		b[v*2+1] = byte(val >> 8)
	}

	qs := make([]*[14]int16, 0, 4)
	blocksData := make([][]byte, 0, 4)
	maxDists := make([]int64, 0, 4)

	for i := 0; i < 4; i++ {
		q := &[14]int16{}
		for d := 0; d < 14; d++ {
			q[d] = int16(rng.IntN(10001) - 5000)
		}
		qs = append(qs, q)

		block := make([]byte, 240)
		for d := 0; d < 14; d++ {
			dimOff := d * 16
			for v := 0; v < 8; v++ {
				writeVI16(block[dimOff:], v, int16(rng.IntN(10001)-5000))
			}
		}
		blocksData = append(blocksData, block)
		maxDists = append(maxDists, int64(rng.IntN(5000000))+1000000)
	}
	maxDists[3] = math.MaxInt64

	for i := range qs {
		var avxSums, genSums [8]int64
		avxMask := distBlockFused(qs[i], unsafe.Pointer(&blocksData[i][0]), 0, &avxSums, maxDists[i])
		genMask := distBlockFusedGeneric(qs[i], unsafe.Pointer(&blocksData[i][0]), 0, &genSums, maxDists[i])

		if avxMask != genMask {
			t.Errorf("block=%d maxDist=%d: AVX2 mask=%08b generic=%08b", i, maxDists[i], avxMask, genMask)
		}

		for v := 0; v < 8; v++ {
			if genMask&(1<<v) != 0 {
				diff := avxSums[v] - genSums[v]
				if diff < 0 {
					diff = -diff
				}
				if diff > 100 {
					t.Errorf("block=%d v=%d (alive): AVX2 sum=%d generic=%d diff=%d maxDist=%d",
						i, v, avxSums[v], genSums[v], diff, maxDists[i])
				}
			}
		}
	}
}

func TestDistBlockFused_EarlyExit(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 42))
	q := &[14]int16{}
	for d := 0; d < 14; d++ {
		q[d] = 1000
	}

	block := make([]byte, 240)
	for d := 0; d < 14; d++ {
		dimOff := d * 16
		for v := 0; v < 8; v++ {
			block[dimOff+v*2] = 0
			block[dimOff+v*2+1] = 0
		}
	}

	var sums [8]int64
	_ = rng
	mask := distBlockFused(q, unsafe.Pointer(&block[0]), 0, &sums, 10)
	genMask := distBlockFusedGeneric(q, unsafe.Pointer(&block[0]), 0, &sums, 10)
	if mask != 0 {
		t.Errorf("expected all dead (mask=0), got mask=%08b", mask)
	}
	if genMask != 0 {
		t.Errorf("generic: expected all dead (mask=0), got mask=%08b", genMask)
	}
}

func TestDistBlockFused_AllAlive(t *testing.T) {
	rng := rand.New(rand.NewPCG(777, 777))
	q := &[14]int16{}
	for d := 0; d < 14; d++ {
		q[d] = int16(rng.IntN(10001) - 5000)
	}

	block := make([]byte, 240)
	for d := 0; d < 14; d++ {
		dimOff := d * 16
		for v := 0; v < 8; v++ {
			block[dimOff+v*2] = byte(q[d])
			block[dimOff+v*2+1] = byte(q[d] >> 8)
		}
	}

	var avxSums, genSums [8]int64
	avxMask := distBlockFused(q, unsafe.Pointer(&block[0]), 0, &avxSums, math.MaxInt64)
	genMask := distBlockFusedGeneric(q, unsafe.Pointer(&block[0]), 0, &genSums, math.MaxInt64)

	for v := 0; v < 8; v++ {
		if avxSums[v] != 0 {
			t.Errorf("v=%d: expected sum=0 (identical vectors), got %d", v, avxSums[v])
		}
	}
	if avxMask != 0xFF {
		t.Errorf("expected all alive (mask=0xFF), got mask=%08b", avxMask)
	}
	if genMask != 0xFF {
		t.Errorf("generic: expected all alive (mask=0xFF), got mask=%08b", genMask)
	}
}

func BenchmarkDistBlockDiff(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 42))
	q := &[14]int16{}
	for d := 0; d < 14; d++ {
		q[d] = int16(rng.IntN(65536) - 32768)
	}
	blocks := make([]byte, 10*240)
	for i := range blocks {
		blocks[i] = byte(rng.IntN(256))
	}
	var diffs [8]int32
	b.ReportAllocs()
	for b.Loop() {
		distBlockDiff(q, unsafe.Pointer(&blocks[0]), 0, 7, &diffs)
	}
}

func BenchmarkDistBlockFused(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 42))
	q := &[14]int16{}
	for d := 0; d < 14; d++ {
		q[d] = int16(rng.IntN(65536) - 32768)
	}
	blocks := make([]byte, 183*240)
	for i := range blocks {
		blocks[i] = byte(rng.IntN(256))
	}
	var sums [8]int64
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		for blk := 0; blk < 183; blk++ {
			distBlockFused(q, unsafe.Pointer(&blocks[0]), uint32(blk*240), &sums, 1<<62)
		}
	}
}
