//go:build recall

package ivf

import (
	"encoding/binary"
	"math/rand/v2"
	"testing"

	"rinha-backend-2026/internal/artifact"
)

func TestIVFRecall(t *testing.T) {
	art, err := artifact.LoadFromFile("../artifact/artifact.bin")
	if err != nil {
		t.Fatalf("load artifact: %v", err)
	}

	numVectors := int(art.NumVectors())
	if numVectors < 1000 {
		t.Skip("not enough vectors for recall test")
	}

	// Pre-compute flat arrays for fast brute-force access
	flatVectors := make([][14]int16, numVectors)
	flatLabels := make([]uint8, numVectors)
	idx := 0
	if art.IsIVF3() {
		for c := 0; c < len(art.ClusterTable); c++ {
			off := int(art.ClusterTable[c].Offset)
			cnt := int(art.ClusterTable[c].Count)
			numBlocks := (cnt + artifact.BlockVectors - 1) / artifact.BlockVectors
			for b := 0; b < numBlocks && idx < numVectors; b++ {
				blockBase := off + b*artifact.BlockSz
				nv := artifact.BlockVectors
				if b == numBlocks-1 && cnt%artifact.BlockVectors > 0 {
					nv = cnt % artifact.BlockVectors
				}
				for v := 0; v < nv && idx < numVectors; v++ {
					for d := 0; d < 14; d++ {
						dimOff := blockBase + d*artifact.BlockDimStride + v*2
						flatVectors[idx][d] = int16(binary.LittleEndian.Uint16(art.VectorsData[dimOff:]))
					}
					flatLabels[idx] = art.VectorsData[blockBase+artifact.BlockLabelOff+v]
					idx++
				}
			}
		}
	} else {
		for c := 0; c < len(art.ClusterTable); c++ {
			off := int(art.ClusterTable[c].Offset)
			cnt := int(art.ClusterTable[c].Count)
			for i := 0; i < cnt; i++ {
				vecOff := off + i*artifact.VectorRecordSz
				for d := 0; d < 14; d++ {
					flatVectors[idx][d] = int16(binary.LittleEndian.Uint16(art.VectorsData[vecOff+d*2:]))
				}
				flatLabels[idx] = art.VectorsData[vecOff+28]
				idx++
			}
		}
	}

	cfg := Config{
		K:          int(art.NumClusters()),
		NProbe:     16,
		RetryExtra: 8,
	}

	rng := rand.New(rand.NewPCG(123, 456))
	numQueries := 1000

	totalRecall := 0
	totalPossible := 0
	decisionAgree := 0
	decisionTotal := 0

	for q := 0; q < numQueries; q++ {
		qidx := rng.IntN(numVectors)
		query := flatVectors[qidx]

		// Brute-force top-5
		bf := &testSink{}
		for i := 0; i < numVectors; i++ {
			dist := refSquaredDistance(&query, &flatVectors[i])
			bf.Insert(dist, flatLabels[i])
		}

		// IVF top-5
		ivfTk := &testSink{}
		err := Search(
			&query,
			art.Centroids,
			art.Bboxes,
			clusterTableBytes(art.ClusterTable),
			art.VectorsData,
			cfg,
			ivfTk,
			art.IsIVF3(),
		)
		if err != nil {
			t.Fatalf("ivf search error: %v", err)
		}

		bfItems := bf.items[:bf.count]
		ivfItems := ivfTk.items[:ivfTk.count]

		matches := 0
		for _, bi := range bfItems {
			for _, ii := range ivfItems {
				if bi.dist == ii.dist && bi.label == ii.label {
					matches++
					break
				}
			}
		}
		totalRecall += matches
		totalPossible += len(bfItems)

		bfDecision := decide(bf.FraudCount())
		ivfDecision := decide(ivfTk.FraudCount())
		if bfDecision == ivfDecision {
			decisionAgree++
		}
		decisionTotal++
	}

	recall := float64(totalRecall) / float64(totalPossible)
	agreement := float64(decisionAgree) / float64(decisionTotal)

	t.Logf("recall@5 = %.4f (%d/%d)", recall, totalRecall, totalPossible)
	t.Logf("decision agreement = %.4f (%d/%d)", agreement, decisionAgree, decisionTotal)

	if recall < 0.99 {
		t.Errorf("recall@5 %.4f below 0.99 threshold", recall)
	}
	if agreement < 0.995 {
		t.Errorf("decision agreement %.4f below 0.995 threshold", agreement)
	}
}

func TestIVFRecallProduction(t *testing.T) {
	art, err := artifact.LoadFromFile("../artifact/artifact.bin")
	if err != nil {
		t.Fatalf("load artifact: %v", err)
	}

	numVectors := int(art.NumVectors())
	if numVectors < 1000 {
		t.Skip("not enough vectors for recall test")
	}

	flatVectors := flattenVectors(art, numVectors)
	flatLabels := flattenLabels(art, numVectors)

	rng := rand.New(rand.NewPCG(123, 456))
	numQueries := 1000

	Klocal := int(art.NumClusters())
	clusterTableBytes := clusterTableBytes(art.ClusterTable)
	scratch := &SearchScratch{
		CentroidDists: make([]int64, Klocal),
	}

	decisionAgree := 0

	for q := 0; q < numQueries; q++ {
		qidx := rng.IntN(numVectors)
		query := flatVectors[qidx]

		bf := &testSink{}
		for i := 0; i < numVectors; i++ {
			dist := refSquaredDistance(&query, &flatVectors[i])
			bf.Insert(dist, flatLabels[i])
		}

		fraudCount := int(FraudCountIVF(
			&query,
			art.Centroids,
			art.Bboxes,
			clusterTableBytes,
			art.VectorsData,
			scratch,
		))

		if decide(bf.FraudCount()) == decide(fraudCount) {
			decisionAgree++
		}
	}

	agreement := float64(decisionAgree) / float64(numQueries)
	t.Logf("decision agreement (production path) = %.4f (%d/%d)", agreement, decisionAgree, numQueries)

	if agreement < 0.995 {
		t.Errorf("decision agreement %.4f below 0.995 threshold (production path)", agreement)
	}
}

func flattenVectors(art *artifact.LoadedArtifact, numVectors int) [][14]int16 {
	out := make([][14]int16, numVectors)
	idx := 0
	if art.IsIVF3() {
		for c := 0; c < len(art.ClusterTable); c++ {
			off := int(art.ClusterTable[c].Offset)
			cnt := int(art.ClusterTable[c].Count)
			numBlocks := (cnt + artifact.BlockVectors - 1) / artifact.BlockVectors
			for b := 0; b < numBlocks && idx < numVectors; b++ {
				blockBase := off + b*artifact.BlockSz
				nv := artifact.BlockVectors
				if b == numBlocks-1 && cnt%artifact.BlockVectors > 0 {
					nv = cnt % artifact.BlockVectors
				}
				for v := 0; v < nv && idx < numVectors; v++ {
					for d := 0; d < 14; d++ {
						dimOff := blockBase + d*artifact.BlockDimStride + v*2
						out[idx][d] = int16(binary.LittleEndian.Uint16(art.VectorsData[dimOff:]))
					}
					idx++
				}
			}
		}
	}
	return out
}

func flattenLabels(art *artifact.LoadedArtifact, numVectors int) []uint8 {
	out := make([]uint8, numVectors)
	idx := 0
	if art.IsIVF3() {
		for c := 0; c < len(art.ClusterTable); c++ {
			off := int(art.ClusterTable[c].Offset)
			cnt := int(art.ClusterTable[c].Count)
			numBlocks := (cnt + artifact.BlockVectors - 1) / artifact.BlockVectors
			for b := 0; b < numBlocks && idx < numVectors; b++ {
				blockBase := off + b*artifact.BlockSz
				nv := artifact.BlockVectors
				if b == numBlocks-1 && cnt%artifact.BlockVectors > 0 {
					nv = cnt % artifact.BlockVectors
				}
				for v := 0; v < nv && idx < numVectors; v++ {
					out[idx] = art.VectorsData[blockBase+artifact.BlockLabelOff+v]
					idx++
				}
			}
		}
	}
	return out
}

func decide(fraudCount int) bool {
	return float64(fraudCount)/5.0 < 0.6
}

func clusterTableBytes(entries []artifact.ClusterEntry) []byte {
	const entrySz = 8
	buf := make([]byte, len(entries)*entrySz)
	for i, e := range entries {
		off := i * entrySz
		buf[off+0] = byte(e.Offset)
		buf[off+1] = byte(e.Offset >> 8)
		buf[off+2] = byte(e.Offset >> 16)
		buf[off+3] = byte(e.Offset >> 24)
		buf[off+4] = byte(e.Count)
		buf[off+5] = byte(e.Count >> 8)
		buf[off+6] = byte(e.Count >> 16)
		buf[off+7] = byte(e.Count >> 24)
	}
	return buf
}
