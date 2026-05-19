package ivf

import (
	"math"
	"unsafe"

	"rinha-backend-2026/internal/artifact"
)

func ComputeRadii(centroids []byte, blocks []byte, clusterTable []artifact.ClusterEntry, K int) []float32 {
	radii := make([]float32, K)
	for c := 0; c < K; c++ {
		count := clusterTable[c].Count
		if count == 0 {
			continue
		}
		offset := clusterTable[c].Offset
		centroidOff := c * artifact.CentroidSz
		centroidPtr := (*[14]int16)(unsafe.Pointer(&centroids[centroidOff]))
		var centroidF32 [14]float32
		for d := 0; d < 14; d++ {
			centroidF32[d] = float32(centroidPtr[d])
		}
		var maxDistSq int64
		numBlocks := int((count + artifact.BlockVectors - 1) / artifact.BlockVectors)
		for b := 0; b < numBlocks; b++ {
			blockOff := offset + uint32(b)*artifact.BlockSz
			nv := artifact.BlockVectors
			if rem := int(count) - b*artifact.BlockVectors; rem < artifact.BlockVectors {
				nv = rem
			}
			for v := 0; v < nv; v++ {
				var sum int64
				for d := 0; d < 14; d++ {
					dimOff := blockOff + uint32(d)*artifact.BlockDimStride
					val := (*[8]int16)(unsafe.Pointer(&blocks[dimOff]))
					diff := int64(val[v]) - int64(centroidPtr[d])
					sum += diff * diff
				}
				if sum > maxDistSq {
					maxDistSq = sum
				}
			}
		}
		radii[c] = float32(math.Sqrt(float64(maxDistSq)))
	}
	return radii
}
