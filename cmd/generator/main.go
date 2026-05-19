package main

import (
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"flag"
	"log"
	"math"
	"os"
	"sort"

	"rinha-backend-2026/internal/artifact"
	"rinha-backend-2026/internal/norm"
)

type refRecord struct {
	Vector []float64 `json:"vector"`
	Label  string    `json:"label"`
}

func main() {
	inPath := flag.String("in", "rinha-de-backend-2026/resources/references.json.gz", "path to references.json.gz")
	outPath := flag.String("out", "internal/artifact/artifact.bin", "output artifact.bin path")
	flag.Parse()

	f, err := os.Open(*inPath)
	if err != nil {
		log.Fatalf("open input: %v", err)
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		log.Fatalf("gzip reader: %v", err)
	}
	defer gz.Close()

	var records []refRecord
	dec := json.NewDecoder(gz)

	if _, err := dec.Token(); err != nil {
		log.Fatalf("expect array start: %v", err)
	}

	for dec.More() {
		var rec refRecord
		if err := dec.Decode(&rec); err != nil {
			log.Fatalf("decode record: %v", err)
		}
		records = append(records, rec)
	}

	if _, err := dec.Token(); err != nil {
		log.Fatalf("expect array end: %v", err)
	}

	N := len(records)
	log.Printf("loaded %d reference vectors", N)

	floatVectors := make([][]float64, N)
	quantVectors := make([]artifact.VectorRecord, N)
	for i, rec := range records {
		floatVectors[i] = rec.Vector
		quantVectors[i] = recordToVector(rec)
	}

	// k-means parameters
	K := 4096
	iters := 30
	sampleSize := 1000000
	refines := 3
	seed := int64(42)

	log.Printf("training k-means: K=%d iters=%d sample=%d refines=%d seed=%d", K, iters, sampleSize, refines, seed)
	centroids, assignments := Train(floatVectors, K, iters, sampleSize, refines, seed)
	log.Printf("k-means training complete")

	// Group vectors by cluster and sort by distance to centroid within cluster
	type clusterVec struct {
		vec  artifact.VectorRecord
		dist float64
	}
	clusters := make([][]clusterVec, K)
	for i := range quantVectors {
		c := assignments[i]
		dist := squaredDistanceFloat64(floatVectors[i], centroids[c])
		clusters[c] = append(clusters[c], clusterVec{vec: quantVectors[i], dist: dist})
	}
	for c := range clusters {
		sort.Slice(clusters[c], func(i, j int) bool {
			return clusters[c][i].dist < clusters[c][j].dist
		})
	}

	// Compute per-cluster bounding boxes
	bboxMin := make([][14]int16, K)
	bboxMax := make([][14]int16, K)
	for k := 0; k < K; k++ {
		for d := 0; d < 14; d++ {
			bboxMin[k][d] = math.MaxInt16
			bboxMax[k][d] = math.MinInt16
		}
	}
	for k := 0; k < K; k++ {
		for _, cv := range clusters[k] {
			for d := 0; d < 14; d++ {
				v := cv.vec.Dims[d]
				if v < bboxMin[k][d] {
					bboxMin[k][d] = v
				}
				if v > bboxMax[k][d] {
					bboxMax[k][d] = v
				}
			}
		}
	}

	// Quantize centroids to int16
	centroidsInt16 := make([][14]int16, K)
	for k := 0; k < K; k++ {
		for d := 0; d < 14; d++ {
			centroidsInt16[k][d] = quantizeDim(centroids[k][d])
		}
	}

	// IVF3 binary layout sizes
	const (
		headerSz       = 20
		centroidSz     = 28  // 14 × int16
		bboxSz         = 56  // 14 × 2 × int16
		clusterEntrySz = 8   // offset + count
	)

	numVectors := uint32(N)
	numClusters := uint32(K)

	centroidsSize := K * centroidSz
	bboxesSize := K * bboxSz
	clusterTableSize := K * clusterEntrySz

	// Compute total block count per cluster and total bytes
	blockCounts := make([]int, K)
	var totalBlocks int
	for k := 0; k < K; k++ {
		bc := (len(clusters[k]) + artifact.BlockVectors - 1) / artifact.BlockVectors
		blockCounts[k] = bc
		totalBlocks += bc
	}
	blocksSize := totalBlocks * artifact.BlockSz

	totalSize := headerSz + centroidsSize + bboxesSize + clusterTableSize + blocksSize
	buf := make([]byte, totalSize)

	// Header (20 bytes)
	copy(buf[0:4], "IVF3")
	binary.LittleEndian.PutUint32(buf[4:], numVectors)
	binary.LittleEndian.PutUint32(buf[8:], artifact.DimCount)
	binary.LittleEndian.PutUint32(buf[12:], numClusters)
	binary.LittleEndian.PutUint16(buf[16:], artifact.FlagInt16Mask)

	// Centroids
	off := headerSz
	for k := 0; k < K; k++ {
		for d := 0; d < 14; d++ {
			binary.LittleEndian.PutUint16(buf[off+k*centroidSz+d*2:], uint16(centroidsInt16[k][d]))
		}
	}

	// Bboxes
	off = headerSz + centroidsSize
	for k := 0; k < K; k++ {
		for d := 0; d < 14; d++ {
			binary.LittleEndian.PutUint16(buf[off+k*bboxSz+d*2:], uint16(bboxMin[k][d]))
			binary.LittleEndian.PutUint16(buf[off+k*bboxSz+28+d*2:], uint16(bboxMax[k][d]))
		}
	}

	// Cluster table + Blocks (SoA)
	clusterTableOff := headerSz + centroidsSize + bboxesSize
	blockOff := uint32(clusterTableOff + clusterTableSize)
	for k := 0; k < K; k++ {
		clusterByteOff := blockOff
		clusterVecs := clusters[k]
		numBlocks := blockCounts[k]

		for b := 0; b < numBlocks; b++ {
			blockBase := blockOff + uint32(b)*artifact.BlockSz
			// Write dims 0..13, each 8 int16 values
			for d := 0; d < 14; d++ {
				dimBase := blockBase + uint32(d)*artifact.BlockDimStride
				for v := 0; v < artifact.BlockVectors; v++ {
					idx := b*artifact.BlockVectors + v
					var val int16
					if idx < len(clusterVecs) {
						val = clusterVecs[idx].vec.Dims[d]
					}
					binary.LittleEndian.PutUint16(buf[dimBase+uint32(v*2):], uint16(val))
				}
			}
			// Write 8 labels
			labelBase := blockBase + artifact.BlockLabelOff
			for v := 0; v < artifact.BlockVectors; v++ {
				idx := b*artifact.BlockVectors + v
				if idx < len(clusterVecs) {
					buf[labelBase+uint32(v)] = clusterVecs[idx].vec.Label
				}
			}
		}

		blockOff += uint32(numBlocks) * artifact.BlockSz

		entryOff := clusterTableOff + k*clusterEntrySz
		binary.LittleEndian.PutUint32(buf[entryOff:], clusterByteOff)
		binary.LittleEndian.PutUint32(buf[entryOff+4:], uint32(len(clusterVecs)))
	}

	if err := os.WriteFile(*outPath, buf, 0644); err != nil {
		log.Fatalf("write artifact: %v", err)
	}

	minSize, maxSize := len(clusters[0]), len(clusters[0])
	for k := range clusters {
		if len(clusters[k]) < minSize {
			minSize = len(clusters[k])
		}
		if len(clusters[k]) > maxSize {
			maxSize = len(clusters[k])
		}
	}
	log.Printf("artifact written: %s (%.1f MB)", *outPath, float64(totalSize)/(1024*1024))
	log.Printf("cluster distribution: min=%d max=%d mean=%.1f", minSize, maxSize, float64(N)/float64(K))
}

func recordToVector(rec refRecord) artifact.VectorRecord {
	var vr artifact.VectorRecord
	for i, v := range rec.Vector {
		vr.Dims[i] = quantizeDim(v)
	}
	switch rec.Label {
	case "fraud":
		vr.Label = artifact.LabelFraud
	case "legit":
		vr.Label = artifact.LabelLegit
	default:
		log.Fatalf("unexpected label %q", rec.Label)
	}
	return vr
}

func quantizeDim(v float64) int16 {
	if v < 0 {
		return artifact.SentinelInt16
	}
	clamped := norm.Clamp(v, 0.0, 1.0)
	return int16(math.Round(clamped * 32767))
}
