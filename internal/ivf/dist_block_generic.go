//go:build !amd64

package ivf

import "unsafe"

func distBlockDiff(q *[14]int16, blocks unsafe.Pointer, blockOff uint32, dim int, diffs *[8]int32) {
	base := uintptr(blocks) + uintptr(blockOff) + uintptr(dim)*16
	dimVals := (*[8]int16)(unsafe.Pointer(base))
	qd := int32(q[dim])
	for v := 0; v < 8; v++ {
		diffs[v] = qd - int32(dimVals[v])
	}
}

func distBlockFused(q *[14]int16, blocks unsafe.Pointer, blockOff uint32,
	sums *[8]int64, maxDist int64) (aliveMask uint32) {
	return distBlockFusedGeneric(q, blocks, blockOff, sums, maxDist)
}
