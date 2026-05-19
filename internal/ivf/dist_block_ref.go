package ivf

import "unsafe"

func distBlockFusedGeneric(q *[14]int16, blocks unsafe.Pointer, blockOff uint32,
	sums *[8]int64, maxDist int64) (aliveMask uint32) {
	for v := 0; v < 8; v++ {
		var ssd int64
		for d := 0; d < 14; d++ {
			dimVal := *(*int16)(unsafe.Add(blocks, int(blockOff)+d*16+v*2))
			diff := int64(q[d]) - int64(dimVal)
			ssd += diff * diff
		}
		sums[v] = ssd
		if ssd < maxDist {
			aliveMask |= 1 << v
		}
	}
	return aliveMask
}
