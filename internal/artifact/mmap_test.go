package artifact

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestLoadMmap_MatchesLoad(t *testing.T) {
	srcData, err := os.ReadFile("artifact.bin")
	if err != nil {
		t.Skipf("artifact.bin not found, skipping mmap comparison: %v", err)
	}

	dir := t.TempDir()
	tmpFile := filepath.Join(dir, "artifact.bin")
	if err := os.WriteFile(tmpFile, srcData, 0644); err != nil {
		t.Fatalf("write temp file: %v", err)
	}

	mmapArt, err := LoadMmap(tmpFile)
	if err != nil {
		t.Fatalf("LoadMmap: %v", err)
	}

	loadArt, err := Load(srcData)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if mmapArt.magic != loadArt.magic {
		t.Errorf("magic: mmap=%q load=%q", mmapArt.magic, loadArt.magic)
	}
	if mmapArt.Header.NumVectors != loadArt.Header.NumVectors {
		t.Errorf("numVectors: mmap=%d load=%d", mmapArt.Header.NumVectors, loadArt.Header.NumVectors)
	}
	if mmapArt.Header.NumClusters != loadArt.Header.NumClusters {
		t.Errorf("numClusters: mmap=%d load=%d", mmapArt.Header.NumClusters, loadArt.Header.NumClusters)
	}
	if mmapArt.Header.DimCount != loadArt.Header.DimCount {
		t.Errorf("dimCount: mmap=%d load=%d", mmapArt.Header.DimCount, loadArt.Header.DimCount)
	}
	if mmapArt.Header.Flags != loadArt.Header.Flags {
		t.Errorf("flags: mmap=%d load=%d", mmapArt.Header.Flags, loadArt.Header.Flags)
	}

	if len(mmapArt.Centroids) != len(loadArt.Centroids) {
		t.Errorf("centroids len: mmap=%d load=%d", len(mmapArt.Centroids), len(loadArt.Centroids))
	} else if !reflect.DeepEqual(mmapArt.Centroids, loadArt.Centroids) {
		t.Errorf("centroids differ")
	}

	if len(mmapArt.Bboxes) != len(loadArt.Bboxes) {
		t.Errorf("bboxes len: mmap=%d load=%d", len(mmapArt.Bboxes), len(loadArt.Bboxes))
	} else if !reflect.DeepEqual(mmapArt.Bboxes, loadArt.Bboxes) {
		t.Errorf("bboxes differ")
	}

	if len(mmapArt.ClusterTable) != len(loadArt.ClusterTable) {
		t.Errorf("clusterTable len: mmap=%d load=%d", len(mmapArt.ClusterTable), len(loadArt.ClusterTable))
	} else if !reflect.DeepEqual(mmapArt.ClusterTable, loadArt.ClusterTable) {
		t.Errorf("clusterTable differs")
	}

	if mmapArt.IsIVF3() != loadArt.IsIVF3() {
		t.Error("IsIVF3 mismatch")
	}
}

func TestLoadMmap_ClusterVectors(t *testing.T) {
	srcData, err := os.ReadFile("artifact.bin")
	if err != nil {
		t.Skipf("artifact.bin not found, skipping: %v", err)
	}

	dir := t.TempDir()
	tmpFile := filepath.Join(dir, "artifact.bin")
	if err := os.WriteFile(tmpFile, srcData, 0644); err != nil {
		t.Fatalf("write temp file: %v", err)
	}

	mmapArt, err := LoadMmap(tmpFile)
	if err != nil {
		t.Fatalf("LoadMmap: %v", err)
	}

	loadArt, err := Load(srcData)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	nc := mmapArt.NumClusters()
	if nc == 0 {
		t.Skip("no clusters in artifact")
	}
	for clusterID := uint32(0); clusterID < nc && clusterID < 3; clusterID++ {
		mmapData, mmapOff, mmapCount := mmapArt.ClusterVectors(clusterID)
		loadData, loadOff, loadCount := loadArt.ClusterVectors(clusterID)

		if mmapOff != loadOff {
			t.Errorf("cluster %d: offset mmap=%d load=%d", clusterID, mmapOff, loadOff)
		}
		if mmapCount != loadCount {
			t.Errorf("cluster %d: count mmap=%d load=%d", clusterID, mmapCount, loadCount)
		}

		if len(mmapData) != len(loadData) {
			t.Errorf("cluster %d: data len mmap=%d load=%d", clusterID, len(mmapData), len(loadData))
			continue
		}
		if !reflect.DeepEqual(mmapData, loadData) {
			t.Errorf("cluster %d: data differs", clusterID)
		}
	}
}

func TestLoadMmap_BadPath(t *testing.T) {
	_, err := LoadMmap("/nonexistent/path/artifact.bin")
	if err == nil {
		t.Fatal("expected error for nonexistent path")
	}
}
