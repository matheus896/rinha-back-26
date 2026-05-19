package artifact

import "unsafe"

const (
	VectorRecordSz = 30
	DimCount       = 14

	FlagInt16Mask = 0x02

	SentinelInt16 int16 = -32768

	LabelLegit uint8 = 0x00
	LabelFraud uint8 = 0x01
)

// BFV1 artifact format constants.
const (
	BFV1HeaderSz = 16
)

// IVF2/IVF3 shared artifact format constants.
const (
	HeaderSz       = 20
	CentroidSz     = 28 // 14 × int16
	BBoxSz         = 56 // 14 × 2 × int16
	ClusterEntrySz = 8  // offset + count
)

// IVF3 block-SoA constants.
const (
	BlockVectors   = 8
	BlockDimStride = 16 // 8 × int16
	BlockLabelOff  = 224 // 14 dims × 16 bytes
	BlockSz        = 240 // 14×16 + 8 labels + 8 padding
)

type IVFHeader struct {
	Magic       [4]byte
	NumVectors  uint32
	DimCount    uint32
	NumClusters uint32
	Flags       uint16
}

type ClusterEntry struct {
	Offset uint32
	Count  uint32
}

type VectorRecord struct {
	Dims  [14]int16
	Label uint8
	_     uint8
}

// ClusterTable casts a byte slice to a typed ClusterEntry slice.
// The caller must ensure alignment and size.
func ClusterTable(data []byte, numClusters uint32) []ClusterEntry {
	offset := HeaderSz
	size := int(numClusters) * ClusterEntrySz
	return unsafe.Slice((*ClusterEntry)(unsafe.Pointer(&data[offset])), size/ClusterEntrySz)
}

// VectorsSlice casts a byte slice to a typed VectorRecord slice.
func VectorsSlice(data []byte, count uint32) []VectorRecord {
	if count == 0 {
		return nil
	}
	return unsafe.Slice((*VectorRecord)(unsafe.Pointer(&data[0])), int(count))
}
