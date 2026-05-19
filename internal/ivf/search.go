package ivf

import (
	"encoding/binary"
	"unsafe"

	"rinha-backend-2026/internal/artifact"
)

type TopK interface {
	Insert(dist int64, label uint8)
	MaxDist() int64
	FraudCount() int
}

func RankCentroids(q *[14]int16, centroids []int16, dists []CentroidDist) {
	K := len(centroids) / 14
	for k := 0; k < K; k++ {
		var sum int64
		for d := 0; d < 14; d++ {
			diff := int64(q[d]) - int64(centroids[k*14+d])
			sum += diff * diff
		}
		dists[k] = CentroidDist{Idx: k, Dist: sum}
	}
}

func FraudCountFull(q *[14]int16, centroids, bboxes, clusterTable, vectorsData []byte, scratch *SearchScratch) uint8 {
	Klocal := len(centroids) / artifact.CentroidSz
	if Klocal == 0 {
		return 0
	}
	if len(scratch.CentroidDists) < Klocal {
		scratch.CentroidDists = make([]int64, Klocal)
	}
	centroidsInt16 := unsafe.Slice((*int16)(unsafe.Pointer(&centroids[0])), Klocal*14)
	rankCentroidsI64(q, centroidsInt16, Klocal, scratch.CentroidDists)

	scratch.Top.Reset()
	for i := range scratch.Scanned {
		scratch.Scanned[i] = 0
	}

	for c := 0; c < Klocal; c++ {
		scanClusterIVF(uint16(c), q, bboxes, clusterTable, vectorsData, Klocal, scratch)
	}
	return uint8(scratch.Top.FraudCount())
}

func FraudCountFastOnly(q *[14]int16, centroids, bboxes, clusterTable, vectorsData []byte, scratch *SearchScratch) (uint8, int64) {
	Klocal := len(centroids) / artifact.CentroidSz
	if Klocal == 0 {
		return 0, 0
	}
	if len(scratch.CentroidDists) < Klocal {
		scratch.CentroidDists = make([]int64, Klocal)
	}
	centroidsInt16 := unsafe.Slice((*int16)(unsafe.Pointer(&centroids[0])), Klocal*14)

	rankCentroidsI64(q, centroidsInt16, Klocal, scratch.CentroidDists)
	pickTopNInt64(scratch.CentroidDists, Klocal, scratch.Picked[:FastNProbe])

	scratch.Top.Reset()
	for i := range scratch.Scanned {
		scratch.Scanned[i] = 0
	}

	for i := 0; i < FastNProbe; i++ {
		c := scratch.Picked[i]
		if c == ^uint16(0) {
			break
		}
		scanClusterIVF(c, q, bboxes, clusterTable, vectorsData, Klocal, scratch)
	}
	return uint8(scratch.Top.FraudCount()), scratch.Top.WorstI64()
}

func FraudCountTopN(q *[14]int16, centroids, bboxes, clusterTable, vectorsData []byte, scratch *SearchScratch, n int) uint8 {
	Klocal := len(centroids) / artifact.CentroidSz
	if Klocal == 0 {
		return 0
	}
	if len(scratch.CentroidDists) < Klocal {
		scratch.CentroidDists = make([]int64, Klocal)
	}
	centroidsInt16 := unsafe.Slice((*int16)(unsafe.Pointer(&centroids[0])), Klocal*14)

	rankCentroidsI64(q, centroidsInt16, Klocal, scratch.CentroidDists)
	pickTopNInt64(scratch.CentroidDists, Klocal, scratch.Picked[:FastNProbe])

	scratch.Top.Reset()
	for i := range scratch.Scanned {
		scratch.Scanned[i] = 0
	}

	for i := 0; i < FastNProbe; i++ {
		c := scratch.Picked[i]
		if c == ^uint16(0) {
			break
		}
		scanClusterIVF(c, q, bboxes, clusterTable, vectorsData, Klocal, scratch)
		scratch.Scanned[c/64] |= 1 << (c % 64)
	}

	count := uint8(scratch.Top.FraudCount())
	needSweep := count == 2 || count == 3 || count == 4
	if !needSweep {
		thr := ExtremeWorstThreshold[count]
		if thr > 0 && scratch.Top.WorstI64() > thr {
			needSweep = true
		}
	}
	if needSweep {
		var escPicked [EscalateNProbe]uint16
		pickNextNUnscanned(scratch.CentroidDists, scratch.Scanned[:], Klocal, n, escPicked[:])
		for _, c := range escPicked {
			if c == ^uint16(0) {
				break
			}
			scanClusterIVF(c, q, bboxes, clusterTable, vectorsData, Klocal, scratch)
			scratch.Scanned[c/64] |= 1 << (c % 64)
		}
		count = uint8(scratch.Top.FraudCount())
	}
	return count
}

func selectTopN(dists []CentroidDist, ranked []int, n int) {
	for i := range n {
		minIdx := i
		for j := i + 1; j < len(dists); j++ {
			if dists[j].Dist < dists[minIdx].Dist {
				minIdx = j
			}
		}
		if minIdx != i {
			dists[i], dists[minIdx] = dists[minIdx], dists[i]
		}
		ranked[i] = dists[i].Idx
	}
}

func BboxPrune(q *[14]int16, bboxMin, bboxMax []int16, maxDist int64) bool {
	var sum int64
	for i := 0; i < 14; i++ {
		if q[i] < bboxMin[i] {
			diff := int64(bboxMin[i]) - int64(q[i])
			sum += diff * diff
		} else if q[i] > bboxMax[i] {
			diff := int64(q[i]) - int64(bboxMax[i])
			sum += diff * diff
		}
		if sum >= maxDist {
			return false
		}
	}
	return true
}

func SquaredDistanceEarlyExit(q, r *[14]int16, maxDist int64) int64 {
	var sum int64
	dims := [14]int{5, 6, 2, 0, 7, 8, 11, 12, 9, 10, 1, 13, 3, 4}
	for _, d := range dims {
		diff := int64(q[d]) - int64(r[d])
		sum += diff * diff
		if sum >= maxDist {
			return sum
		}
	}
	return sum
}

var dimOrder = [14]int{5, 6, 2, 0, 7, 8, 11, 12, 9, 10, 1, 13, 3, 4}

func ScanCluster(q *[14]int16, vectors []byte, offset, count uint32, maxDist int64, tk TopK) int64 {
	if count == 0 {
		return maxDist
	}
	recSize := uint32(artifact.VectorRecordSz)
	for i := uint32(0); i < count; i++ {
		base := offset + i*recSize
		vec := (*[14]int16)(unsafe.Pointer(&vectors[base]))
		var sum int64
		for _, d := range dimOrder {
			diff := int64(q[d]) - int64(vec[d])
			sum += diff * diff
			if sum >= maxDist {
				goto next
			}
		}
		if sum < maxDist {
			label := vectors[base+28]
			tk.Insert(sum, label)
			maxDist = tk.MaxDist()
		}
	next:
	}
	return maxDist
}

func ScanClusterSoA(q *[14]int16, blocks []byte, offset, count uint32, maxDist int64, tk TopK) int64 {
	if count == 0 {
		return maxDist
	}
	var sums [8]int64
	numBlocks := int((count + artifact.BlockVectors - 1) / artifact.BlockVectors)
	for b := 0; b < numBlocks; b++ {
		blockOff := offset + uint32(b)*artifact.BlockSz
		nv := artifact.BlockVectors
		if rem := int(count) - b*artifact.BlockVectors; rem < artifact.BlockVectors {
			nv = rem
		}
		aliveMask := distBlockFused(q, unsafe.Pointer(&blocks[0]), blockOff, &sums, maxDist)
		for v := 0; v < nv; v++ {
			if aliveMask&(1<<v) == 0 {
				continue
			}
			if sums[v] < maxDist {
				exact := exactI64Dist(q, unsafe.Pointer(&blocks[0]), blockOff, v)
				label := blocks[blockOff+artifact.BlockLabelOff+uint32(v)]
				tk.Insert(exact, label)
				maxDist = tk.MaxDist()
			}
		}
	}
	return maxDist
}

func exactI64Dist(q *[14]int16, blocks unsafe.Pointer, blockOff uint32, v int) int64 {
	var sum int64
	for d := 0; d < 14; d++ {
		dimVal := *(*int16)(unsafe.Add(blocks, int(blockOff)+d*16+v*2))
		diff := int64(q[d]) - int64(dimVal)
		sum += diff * diff
	}
	return sum
}

func SearchWithBuf(q *[14]int16, centroids, bboxes, clusterTable, vectorsData []byte, cfg Config, tk TopK, buf *CentroidRankBuf, isIVF3 bool) error {
	K := len(centroids) / artifact.CentroidSz
	if K == 0 || cfg.NProbe <= 0 {
		return nil
	}

	centroidsInt16 := unsafe.Slice((*int16)(unsafe.Pointer(&centroids[0])), len(centroids)/2)

	totalNeeded := cfg.NProbe
	if cfg.RetryExtra > 0 {
		totalNeeded += cfg.RetryExtra
	}
	if totalNeeded > K {
		totalNeeded = K
	}
	RankCentroids(q, centroidsInt16, buf.Dists)
	selectTopN(buf.Dists, buf.Ranked, totalNeeded)
	ranked := buf.Ranked[:totalNeeded]

	scanOne := func(cid int) {
		if cid < 0 || cid >= K {
			return
		}
		entryOff := cid * artifact.ClusterEntrySz
		offset := binary.LittleEndian.Uint32(clusterTable[entryOff:])
		count := binary.LittleEndian.Uint32(clusterTable[entryOff+4:])
		if count == 0 {
			return
		}

		maxDist := tk.MaxDist()
		bboxOff := cid * artifact.BBoxSz
		bboxMin := unsafe.Slice((*int16)(unsafe.Pointer(&bboxes[bboxOff])), 14)
		bboxMax := unsafe.Slice((*int16)(unsafe.Pointer(&bboxes[bboxOff+28])), 14)
		if !BboxPrune(q, bboxMin, bboxMax, maxDist) {
			return
		}
		if isIVF3 {
			ScanClusterSoA(q, vectorsData, offset, count, maxDist, tk)
		} else {
			ScanCluster(q, vectorsData, offset, count, maxDist, tk)
		}
	}

	for i := 0; i < cfg.NProbe && i < len(ranked); i++ {
		scanOne(ranked[i])
	}

	fraudCount := tk.FraudCount()
	if fraudCount >= 2 && fraudCount <= 3 {
		for i := cfg.NProbe; i < totalNeeded && i < len(ranked); i++ {
			scanOne(ranked[i])
		}
	}

	return nil
}

func Search(q *[14]int16, centroids, bboxes, clusterTable, vectorsData []byte, cfg Config, tk TopK, isIVF3 bool) error {
	K := len(centroids) / artifact.CentroidSz
	if K == 0 || cfg.NProbe <= 0 {
		return nil
	}

	centroidsInt16 := unsafe.Slice((*int16)(unsafe.Pointer(&centroids[0])), len(centroids)/2)

	totalNeeded := cfg.NProbe
	if cfg.RetryExtra > 0 {
		totalNeeded += cfg.RetryExtra
	}
	if totalNeeded > K {
		totalNeeded = K
	}

	dists := make([]CentroidDist, K)
	RankCentroids(q, centroidsInt16, dists)
	ranked := make([]int, totalNeeded)
	selectTopN(dists, ranked, totalNeeded)

	scanOne := func(cid int) {
		if cid < 0 || cid >= K {
			return
		}
		entryOff := cid * artifact.ClusterEntrySz
		offset := binary.LittleEndian.Uint32(clusterTable[entryOff:])
		count := binary.LittleEndian.Uint32(clusterTable[entryOff+4:])
		if count == 0 {
			return
		}

		maxDist := tk.MaxDist()
		bboxOff := cid * artifact.BBoxSz
		bboxMin := unsafe.Slice((*int16)(unsafe.Pointer(&bboxes[bboxOff])), 14)
		bboxMax := unsafe.Slice((*int16)(unsafe.Pointer(&bboxes[bboxOff+28])), 14)
		if !BboxPrune(q, bboxMin, bboxMax, maxDist) {
			return
		}
		if isIVF3 {
			ScanClusterSoA(q, vectorsData, offset, count, maxDist, tk)
		} else {
			ScanCluster(q, vectorsData, offset, count, maxDist, tk)
		}
	}

	for i := 0; i < cfg.NProbe && i < len(ranked); i++ {
		scanOne(ranked[i])
	}

	fraudCount := tk.FraudCount()
	if fraudCount >= 2 && fraudCount <= 3 {
		for i := cfg.NProbe; i < totalNeeded && i < len(ranked); i++ {
			scanOne(ranked[i])
		}
	}

	return nil
}

func FraudCountIVF(q *[14]int16, centroids, bboxes, clusterTable, vectorsData []byte, scratch *SearchScratch) uint8 {
	Klocal := len(centroids) / artifact.CentroidSz
	if Klocal == 0 {
		return 0
	}
	if len(scratch.CentroidDists) < Klocal {
		scratch.CentroidDists = make([]int64, Klocal)
	}
	centroidsInt16 := unsafe.Slice((*int16)(unsafe.Pointer(&centroids[0])), Klocal*14)

	rankCentroidsI64(q, centroidsInt16, Klocal, scratch.CentroidDists)

	pickTopNInt64(scratch.CentroidDists, Klocal, scratch.Picked[:FastNProbe])

	scratch.Top.Reset()
	for i := range scratch.Scanned {
		scratch.Scanned[i] = 0
	}

	for i := 0; i < FastNProbe; i++ {
		c := scratch.Picked[i]
		if c == ^uint16(0) {
			break
		}
		scanClusterIVF(c, q, bboxes, clusterTable, vectorsData, Klocal, scratch)
		scratch.Scanned[c/64] |= 1 << (c % 64)
	}

	count := uint8(scratch.Top.FraudCount())
	needSweep := count == 2 || count == 3 || count == 4
	if !needSweep {
		thr := ExtremeWorstThreshold[count]
		if thr > 0 && scratch.Top.WorstI64() > thr {
			needSweep = true
		}
	}
	if needSweep {
		var escPicked [EscalateNProbe]uint16
		pickNextNUnscanned(scratch.CentroidDists, scratch.Scanned[:], Klocal, EscalateNProbe, escPicked[:])
		for _, c := range escPicked {
			if c == ^uint16(0) {
				break
			}
			scanClusterIVF(c, q, bboxes, clusterTable, vectorsData, Klocal, scratch)
			scratch.Scanned[c/64] |= 1 << (c % 64)
		}
		count = uint8(scratch.Top.FraudCount())
	}
	return count
}

func scanClusterIVF(cid uint16, q *[14]int16, bboxes, clusterTable, vectorsData []byte, Klocal int, scratch *SearchScratch) {
	c := int(cid)
	if c < 0 || c >= Klocal {
		return
	}
	entryOff := c * artifact.ClusterEntrySz
	offset := binary.LittleEndian.Uint32(clusterTable[entryOff:])
	count := binary.LittleEndian.Uint32(clusterTable[entryOff+4:])
	if count == 0 {
		return
	}

	maxDist := scratch.Top.MaxDist()
	bboxOff := c * artifact.BBoxSz
	bboxMin := unsafe.Slice((*int16)(unsafe.Pointer(&bboxes[bboxOff])), 14)
	bboxMax := unsafe.Slice((*int16)(unsafe.Pointer(&bboxes[bboxOff+28])), 14)
	if !BboxPrune(q, bboxMin, bboxMax, maxDist) {
		return
	}

	ScanClusterSoA(q, vectorsData, offset, count, maxDist, &scratch.Top)
}

func rankCentroidsI64(q *[14]int16, centroids []int16, K int, dists []int64) {
	for k := 0; k < K; k++ {
		var sum int64
		base := k * 14
		for d := 0; d < 14; d++ {
			diff := int64(q[d]) - int64(centroids[base+d])
			sum += diff * diff
		}
		dists[k] = sum
	}
}

func pickTopNInt64(dists []int64, K int, picked []uint16) {
	n := len(picked)
	for i := range picked {
		picked[i] = ^uint16(0)
	}
	worst := int64(1<<63 - 1)
	worstPos := 0
	for c := 0; c < K; c++ {
		d := dists[c]
		if d >= worst {
			continue
		}
		pos := 0
		for pos < n {
			id := picked[pos]
			if id == ^uint16(0) || dists[id] > d {
				break
			}
			pos++
		}
		if pos == n {
			continue
		}
		for i := n - 1; i > pos; i-- {
			picked[i] = picked[i-1]
		}
		picked[pos] = uint16(c)
		last := picked[n-1]
		if last == ^uint16(0) {
			worst = 1<<63 - 1
		} else {
			worst = dists[last]
			worstPos = int(last)
			_ = worstPos
		}
	}
}

func pickNextNUnscanned(dists []int64, scanned []uint64, K, n int, picked []uint16) {
	if n > len(picked) {
		n = len(picked)
	}
	if n == 0 {
		return
	}
	var heapDist [64]int64
	var heapIdx [64]uint16
	heapSize := 0
	worst := int64(1<<63 - 1)
	for w := 0; w*64 < K; w++ {
		mask := scanned[w]
		base := w * 64
		end := base + 64
		if end > K {
			end = K
		}
		for c := base; c < end; c++ {
			if mask&(1<<uint(c-base)) != 0 {
				continue
			}
			d := dists[c]
			if heapSize == n && d >= worst {
				continue
			}
			if heapSize < n {
				i := heapSize
				heapDist[i] = d
				heapIdx[i] = uint16(c)
				for i > 0 {
					parent := (i - 1) >> 1
					if heapDist[parent] >= heapDist[i] {
						break
					}
					heapDist[parent], heapDist[i] = heapDist[i], heapDist[parent]
					heapIdx[parent], heapIdx[i] = heapIdx[i], heapIdx[parent]
					i = parent
				}
				heapSize++
				if heapSize == n {
					worst = heapDist[0]
				}
			} else {
				heapDist[0] = d
				heapIdx[0] = uint16(c)
				i := 0
				for {
					l := 2*i + 1
					r := 2*i + 2
					largest := i
					if l < n && heapDist[l] > heapDist[largest] {
						largest = l
					}
					if r < n && heapDist[r] > heapDist[largest] {
						largest = r
					}
					if largest == i {
						break
					}
					heapDist[largest], heapDist[i] = heapDist[i], heapDist[largest]
					heapIdx[largest], heapIdx[i] = heapIdx[i], heapIdx[largest]
					i = largest
				}
				worst = heapDist[0]
			}
		}
	}
	for i := 0; i < n; i++ {
		picked[i] = ^uint16(0)
	}
	for size := heapSize; size > 0; size-- {
		picked[size-1] = heapIdx[0]
		heapDist[0] = heapDist[size-1]
		heapIdx[0] = heapIdx[size-1]
		i := 0
		end := size - 1
		for {
			l := 2*i + 1
			r := 2*i + 2
			largest := i
			if l < end && heapDist[l] > heapDist[largest] {
				largest = l
			}
			if r < end && heapDist[r] > heapDist[largest] {
				largest = r
			}
			if largest == i {
				break
			}
			heapDist[largest], heapDist[i] = heapDist[i], heapDist[largest]
			heapIdx[largest], heapIdx[i] = heapIdx[i], heapIdx[largest]
			i = largest
		}
	}
}
