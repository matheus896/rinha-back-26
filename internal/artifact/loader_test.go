package artifact

import (
	"encoding/binary"
	"testing"
)

func buildBFV1(vectors []VectorRecord) []byte {
	totalSize := 16 + len(vectors)*VectorRecordSz
	buf := make([]byte, totalSize)
	copy(buf[0:4], "BFV1")
	binary.LittleEndian.PutUint32(buf[4:], uint32(len(vectors)))
	binary.LittleEndian.PutUint32(buf[8:], DimCount)
	binary.LittleEndian.PutUint16(buf[12:], FlagInt16Mask)
	off := 16
	for i := range vectors {
		for d := 0; d < 14; d++ {
			binary.LittleEndian.PutUint16(buf[off+i*VectorRecordSz+d*2:], uint16(vectors[i].Dims[d]))
		}
		buf[off+i*VectorRecordSz+28] = vectors[i].Label
	}
	return buf
}

func TestLoad_BFV1_Valid(t *testing.T) {
	vecs := []VectorRecord{
		{Dims: [14]int16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, Label: LabelFraud},
		{Dims: [14]int16{14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}, Label: LabelLegit},
	}
	data := buildBFV1(vecs)

	art, err := Load(data)
	if err != nil {
		t.Fatalf("Load BFV1: %v", err)
	}

	if art.NumVectors() != 2 {
		t.Errorf("num vectors: got %d, want 2", art.NumVectors())
	}
	if art.NumClusters() != 0 {
		t.Errorf("BFV1 should have 0 clusters, got %d", art.NumClusters())
	}
	if string(art.Header.Magic[:]) != "BFV1" {
		t.Errorf("magic: got %q, want BFV1", string(art.Header.Magic[:]))
	}

	vectors := VectorsSlice(art.VectorsData, art.NumVectors())
	if len(vectors) != 2 {
		t.Fatalf("vectors slice len: got %d, want 2", len(vectors))
	}
	if vectors[0].Dims[0] != 1 || vectors[0].Label != LabelFraud {
		t.Errorf("vector[0]: dims[0]=%d label=%d", vectors[0].Dims[0], vectors[0].Label)
	}
	if vectors[1].Dims[0] != 14 || vectors[1].Label != LabelLegit {
		t.Errorf("vector[1]: dims[0]=%d label=%d", vectors[1].Dims[0], vectors[1].Label)
	}
}

func TestLoad_BFV1_InvalidMagic(t *testing.T) {
	data := buildBFV1([]VectorRecord{{}})
	data[0] = 'X'
	_, err := Load(data)
	if err == nil {
		t.Fatal("expected error for invalid magic")
	}
}

func TestLoad_BFV1_Truncated(t *testing.T) {
	vecs := []VectorRecord{{}}
	data := buildBFV1(vecs)
	_, err := Load(data[:len(data)-5])
	if err == nil {
		t.Fatal("expected error for truncated BFV1")
	}
}

func TestLoad_BFV1_InvalidLabel(t *testing.T) {
	vecs := []VectorRecord{{Label: 0xFF}}
	data := buildBFV1(vecs)
	_, err := Load(data)
	if err == nil {
		t.Fatal("expected error for invalid label in BFV1")
	}
}

func TestLoad_BFV1_WrongDimCount(t *testing.T) {
	data := buildBFV1([]VectorRecord{{}})
	binary.LittleEndian.PutUint32(data[8:], 999)
	_, err := Load(data)
	if err == nil {
		t.Fatal("expected error for wrong dim count")
	}
}
