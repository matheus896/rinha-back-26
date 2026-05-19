//go:build amd64

package ivf

import "unsafe"

//go:noescape
func distBlockDiff(q *[14]int16, blocks unsafe.Pointer, blockOff uint32, dim int, diffs *[8]int32)

//go:noescape
func distBlockFused(q *[14]int16, blocks unsafe.Pointer, blockOff uint32,
	sums *[8]int64, maxDist int64) (aliveMask uint32)
