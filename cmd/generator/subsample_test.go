package main

import (
	"testing"
)

func TestStratifiedSample_RatioPreserved(t *testing.T) {
	t.Skip("stratifiedSample removed in Path A (full dataset, no subsampling)")
}

func TestStratifiedSample_Deterministic(t *testing.T) {
	t.Skip("stratifiedSample removed in Path A (full dataset, no subsampling)")
}

func TestBFV1WriteReadRoundtrip(t *testing.T) {
	t.Skip("writeBFV1 removed in Path A (IVF2 format only)")
}
