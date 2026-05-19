package artifact

import (
	"encoding/binary"
	"fmt"
	"unsafe"
)

type LoadedArtifact struct {
	Header       *IVFHeader
	Centroids    []byte
	Bboxes       []byte
	ClusterTable []ClusterEntry
	VectorsData  []byte
	numClusters  uint32
	numVectors   uint32
	magic        string
	isIVF3       bool
	raw          []byte // backing store for mmap; kept alive to prevent GC from unmapping
}

func Load(data []byte) (*LoadedArtifact, error) {
	if len(data) < 16 {
		return nil, fmt.Errorf("artifact too small: %d bytes, minimum 16", len(data))
	}

	magic := string(data[0:4])

	switch magic {
	case "BFV1":
		return loadBFV1(data)
	case "IVF2":
		return loadIVF2(data)
	case "IVF3":
		return loadIVF3(data)
	default:
		return nil, fmt.Errorf("invalid artifact magic: got %q, expected BFV1, IVF2 or IVF3", magic)
	}
}

func loadBFV1(data []byte) (*LoadedArtifact, error) {
	if len(data) < 16 {
		return nil, fmt.Errorf("BFV1 too small: %d bytes", len(data))
	}

	numVectors := binary.LittleEndian.Uint32(data[4:8])
	dimCount := binary.LittleEndian.Uint32(data[8:12])
	flags := binary.LittleEndian.Uint16(data[12:14])

	if dimCount != DimCount {
		return nil, fmt.Errorf("invalid dim_count: got %d, expected %d", dimCount, DimCount)
	}
	if flags&FlagInt16Mask == 0 {
		return nil, fmt.Errorf("BFV1 flags indicate non-int16 format: flags=%02x", flags)
	}

	expectedSize := 16 + int(numVectors)*VectorRecordSz
	if len(data) < expectedSize {
		return nil, fmt.Errorf("BFV1 truncated: got %d bytes, expected %d", len(data), expectedSize)
	}

	h := &IVFHeader{}
	copy(h.Magic[:], "BFV1")
	h.NumVectors = numVectors
	h.DimCount = dimCount
	h.Flags = flags

	vectorsData := data[16:expectedSize]

	// Validate labels
	allVectors := VectorsSlice(vectorsData, numVectors)
	for i := range allVectors {
		l := allVectors[i].Label
		if l != LabelLegit && l != LabelFraud {
			return nil, fmt.Errorf("vector %d has invalid label 0x%02x", i, l)
		}
	}

	return &LoadedArtifact{
		Header:      h,
		VectorsData: vectorsData,
		numVectors:  numVectors,
		numClusters: 0,
		magic:       "BFV1",
	}, nil
}

func loadIVF2(data []byte) (*LoadedArtifact, error) {
	if len(data) < HeaderSz {
		return nil, fmt.Errorf("IVF2 artifact too small: %d bytes, minimum %d", len(data), HeaderSz)
	}

	h := &IVFHeader{}
	copy(h.Magic[:], "IVF2")
	h.NumVectors = binary.LittleEndian.Uint32(data[4:8])
	h.DimCount = binary.LittleEndian.Uint32(data[8:12])
	h.NumClusters = binary.LittleEndian.Uint32(data[12:16])
	h.Flags = binary.LittleEndian.Uint16(data[16:18])

	if h.DimCount != DimCount {
		return nil, fmt.Errorf("invalid dim_count: got %d, expected %d", h.DimCount, DimCount)
	}

	if h.Flags&FlagInt16Mask == 0 {
		return nil, fmt.Errorf("IVF2 flags indicate non-int16 format: flags=%02x", h.Flags)
	}

	K := int(h.NumClusters)
	centroidsSize := K * CentroidSz
	bboxesSize := K * BBoxSz
	clusterTableSize := K * ClusterEntrySz
	vectorsSize := int(h.NumVectors) * VectorRecordSz

	expectedSize := HeaderSz + centroidsSize + bboxesSize + clusterTableSize + vectorsSize
	if len(data) < expectedSize {
		return nil, fmt.Errorf("IVF2 truncated: got %d bytes, expected %d", len(data), expectedSize)
	}

	centroids := data[HeaderSz : HeaderSz+centroidsSize]
	bboxes := data[HeaderSz+centroidsSize : HeaderSz+centroidsSize+bboxesSize]
	clusterTableBytes := data[HeaderSz+centroidsSize+bboxesSize : HeaderSz+centroidsSize+bboxesSize+clusterTableSize]
	vectorsOffset := HeaderSz + centroidsSize + bboxesSize + clusterTableSize
	vectorsData := data[vectorsOffset:]

	clusterTable := make([]ClusterEntry, K)
	for i := 0; i < K; i++ {
		off := i * ClusterEntrySz
		clusterTable[i].Offset = binary.LittleEndian.Uint32(clusterTableBytes[off:])
		clusterTable[i].Count = binary.LittleEndian.Uint32(clusterTableBytes[off+4:])
	}

	var totalVectors uint32
	for i := 0; i < K; i++ {
		totalVectors += clusterTable[i].Count
		vecEnd := int(clusterTable[i].Offset) + int(clusterTable[i].Count)*VectorRecordSz
		if vecEnd > len(data) {
			return nil, fmt.Errorf("cluster %d vectors overflow: offset=%d count=%d, data len=%d",
				i, clusterTable[i].Offset, clusterTable[i].Count, len(data))
		}
	}
	if totalVectors != h.NumVectors {
		return nil, fmt.Errorf("cluster checksum mismatch: sum(counts)=%d, num_vectors=%d", totalVectors, h.NumVectors)
	}

	// Validate labels
	vectors := VectorsSlice(vectorsData, h.NumVectors)
	for i := range vectors {
		l := vectors[i].Label
		if l != LabelLegit && l != LabelFraud {
			return nil, fmt.Errorf("vector %d has invalid label 0x%02x", i, l)
		}
	}

	return &LoadedArtifact{
		Header:       h,
		Centroids:    centroids,
		Bboxes:       bboxes,
		ClusterTable: clusterTable,
		VectorsData:  data,
		numClusters:  h.NumClusters,
		numVectors:   h.NumVectors,
		magic:        "IVF2",
	}, nil
}

func loadIVF3(data []byte) (*LoadedArtifact, error) {
	if len(data) < HeaderSz {
		return nil, fmt.Errorf("IVF3 artifact too small: %d bytes, minimum %d", len(data), HeaderSz)
	}

	h := &IVFHeader{}
	copy(h.Magic[:], "IVF3")
	h.NumVectors = binary.LittleEndian.Uint32(data[4:8])
	h.DimCount = binary.LittleEndian.Uint32(data[8:12])
	h.NumClusters = binary.LittleEndian.Uint32(data[12:16])
	h.Flags = binary.LittleEndian.Uint16(data[16:18])

	if h.DimCount != DimCount {
		return nil, fmt.Errorf("invalid dim_count: got %d, expected %d", h.DimCount, DimCount)
	}
	if h.Flags&FlagInt16Mask == 0 {
		return nil, fmt.Errorf("IVF3 flags indicate non-int16 format: flags=%02x", h.Flags)
	}

	K := int(h.NumClusters)
	centroidsSize := K * CentroidSz
	bboxesSize := K * BBoxSz
	clusterTableSize := K * ClusterEntrySz

	centroids := data[HeaderSz : HeaderSz+centroidsSize]
	bboxes := data[HeaderSz+centroidsSize : HeaderSz+centroidsSize+bboxesSize]
	clusterTableBytes := data[HeaderSz+centroidsSize+bboxesSize : HeaderSz+centroidsSize+bboxesSize+clusterTableSize]

	clusterTable := make([]ClusterEntry, K)
	var totalVectors uint32
	for i := 0; i < K; i++ {
		off := i * ClusterEntrySz
		clusterTable[i].Offset = binary.LittleEndian.Uint32(clusterTableBytes[off:])
		clusterTable[i].Count = binary.LittleEndian.Uint32(clusterTableBytes[off+4:])
		totalVectors += clusterTable[i].Count
	}
	if totalVectors != h.NumVectors {
		return nil, fmt.Errorf("cluster checksum mismatch: sum(counts)=%d, num_vectors=%d", totalVectors, h.NumVectors)
	}

	// Verify blocks fit in data
	for i := 0; i < K; i++ {
		entry := &clusterTable[i]
		numBlocks := int((entry.Count + BlockVectors - 1) / BlockVectors)
		blockEnd := int(entry.Offset) + numBlocks*BlockSz
		if blockEnd > len(data) {
			return nil, fmt.Errorf("cluster %d blocks overflow: offset=%d count=%d blocks=%d, data len=%d",
				i, entry.Offset, entry.Count, numBlocks, len(data))
		}
	}

	// Validate labels in blocks
	for i := 0; i < K; i++ {
		entry := &clusterTable[i]
		numBlocks := int((entry.Count + BlockVectors - 1) / BlockVectors)
		for b := 0; b < numBlocks; b++ {
			blockBase := entry.Offset + uint32(b)*BlockSz
			nv := int(BlockVectors)
			if b == numBlocks-1 {
				if rem := int(entry.Count % BlockVectors); rem > 0 {
					nv = rem
				}
			}
			for v := 0; v < nv; v++ {
				l := data[blockBase+BlockLabelOff+uint32(v)]
				if l != LabelLegit && l != LabelFraud {
					return nil, fmt.Errorf("cluster %d block %d vector %d has invalid label 0x%02x", i, b, v, l)
				}
			}
		}
	}

	return &LoadedArtifact{
		Header:       h,
		Centroids:    centroids,
		Bboxes:       bboxes,
		ClusterTable: clusterTable,
		VectorsData:  data,
		numClusters:  h.NumClusters,
		numVectors:   h.NumVectors,
		magic:        "IVF3",
		isIVF3:       true,
	}, nil
}

func (a *LoadedArtifact) IsIVF3() bool {
	return a.isIVF3
}

func (a *LoadedArtifact) ClusterVectors(clusterID uint32) ([]byte, uint32, uint32) {
	if a.magic == "BFV1" {
		if clusterID != 0 {
			return nil, 0, 0
		}
		return a.VectorsData, 0, a.numVectors
	}
	if clusterID >= a.numClusters {
		return nil, 0, 0
	}
	entry := &a.ClusterTable[clusterID]
	return a.VectorsData, entry.Offset, entry.Count
}

func (a *LoadedArtifact) CentroidsSlice() []int16 {
	if len(a.Centroids) == 0 {
		return nil
	}
	return unsafeSliceInt16(a.Centroids)
}

func (a *LoadedArtifact) BboxesSlice() []int16 {
	if len(a.Bboxes) == 0 {
		return nil
	}
	return unsafeSliceInt16(a.Bboxes)
}

func (a *LoadedArtifact) NumClusters() uint32 {
	return a.numClusters
}

func (a *LoadedArtifact) NumVectors() uint32 {
	return a.numVectors
}

func (a *LoadedArtifact) IsBFV1() bool {
	return a.magic == "BFV1"
}

func unsafeSliceInt16(data []byte) []int16 {
	return unsafe.Slice((*int16)(unsafe.Pointer(&data[0])), len(data)/2)
}
