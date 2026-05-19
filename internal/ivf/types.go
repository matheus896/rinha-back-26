package ivf

const (
	FastNProbe     = 2
	EscalateNProbe = 32
	MaxNProbe      = 256
	K              = 4096
)

// Config holds tunable IVF search parameters.
type Config struct {
	K          int
	NProbe     int
	RetryExtra int
}

// CentroidDist pairs a cluster index with its centroid distance.
type CentroidDist struct {
	Idx  int
	Dist int64
}

// CentroidRankBuf is a pre-allocated buffer reused across searches via sync.Pool.
type CentroidRankBuf struct {
	Dists  []CentroidDist
	Ranked []int
}

// SearchScratch holds per-query reusable buffers for the two-tier IVF search.
type SearchScratch struct {
	CentroidDists []int64
	Picked        [MaxNProbe]uint16
	Scanned       [(K + 63) / 64]uint64
	Top           Top5
}
