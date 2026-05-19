package search

import (
	"encoding/binary"
	"math/rand/v2"
	"sync"

	"rinha-backend-2026/internal/artifact"
	"rinha-backend-2026/internal/ivf"
)

type Engine struct {
	art               *artifact.LoadedArtifact
	cfg               ivf.Config
	clusterTableBytes []byte
	centroidBuf       *sync.Pool
	radii             []float32
	scratchPool       *sync.Pool
}

func NewEngine(art *artifact.LoadedArtifact, cfg ivf.Config) *Engine {
	K := int(art.NumClusters())
	var radii []float32
	if art.IsIVF3() {
		radii = ivf.ComputeRadii(art.Centroids, art.VectorsData, art.ClusterTable, K)
	}
	e := &Engine{
		art:               art,
		cfg:               cfg,
		clusterTableBytes: serializeClusterTable(art.ClusterTable),
		radii:             radii,
	}
	e.centroidBuf = &sync.Pool{
		New: func() any {
			n := cfg.NProbe + cfg.RetryExtra
			if n > K {
				n = K
			}
			buf := &ivf.CentroidRankBuf{
				Dists:  make([]ivf.CentroidDist, K),
				Ranked: make([]int, n),
			}
			return buf
		},
	}
	e.scratchPool = &sync.Pool{
		New: func() any {
			return &ivf.SearchScratch{
				CentroidDists: make([]int64, K),
			}
		},
	}
	return e
}

func serializeClusterTable(entries []artifact.ClusterEntry) []byte {
	const entrySz = 8
	buf := make([]byte, len(entries)*entrySz)
	for i, e := range entries {
		off := i * entrySz
		binary.LittleEndian.PutUint32(buf[off:], e.Offset)
		binary.LittleEndian.PutUint32(buf[off+4:], e.Count)
	}
	return buf
}

func (e *Engine) Warmup(iters int) {
	if !e.art.IsIVF3() {
		return
	}
	rng := rand.New(rand.NewPCG(0xC0FFEE, 0xCAFE))
	scratch := e.scratchPool.Get().(*ivf.SearchScratch)
	defer e.scratchPool.Put(scratch)

	var qi [14]int16
	for i := 0; i < iters; i++ {
		for d := 0; d < 14; d++ {
			qi[d] = int16(rng.IntN(65534) - 32767)
		}
		_ = ivf.FraudCountIVF(
			&qi,
			e.art.Centroids,
			e.art.Bboxes,
			e.clusterTableBytes,
			e.art.VectorsData,
			scratch,
		)
	}
}

func (e *Engine) Search(query *[14]int16) (*ivf.Top5, error) {
	if e.art.IsIVF3() {
		scratch := e.scratchPool.Get().(*ivf.SearchScratch)
		fraudCount := ivf.FraudCountIVF(
			query,
			e.art.Centroids,
			e.art.Bboxes,
			e.clusterTableBytes,
			e.art.VectorsData,
			scratch,
		)
		top := scratch.Top
		e.scratchPool.Put(scratch)
		_ = fraudCount
		return &top, nil
	}
	var tk ivf.Top5
	tk.Reset()
	buf := e.centroidBuf.Get().(*ivf.CentroidRankBuf)
	_ = ivf.SearchWithBuf(
		query,
		e.art.Centroids,
		e.art.Bboxes,
		e.clusterTableBytes,
		e.art.VectorsData,
		e.cfg,
		&tk,
		buf,
		false,
	)
	e.centroidBuf.Put(buf)
	return &tk, nil
}
