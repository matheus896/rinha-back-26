package server

import (
	"compress/gzip"
	"encoding/json"
	"math/rand/v2"
	"os"
	"testing"

	"rinha-backend-2026/internal/artifact"
	"rinha-backend-2026/internal/bruteforce"
	"rinha-backend-2026/internal/mcc"
	"rinha-backend-2026/internal/norm"
	"rinha-backend-2026/internal/search"
)

type validationRec struct {
	Vector []float64 `json:"vector"`
	Label  string    `json:"label"`
}

func loadFullReferences(t testing.TB) ([]artifact.VectorRecord, *norm.Normalization, mcc.Risks) {
	t.Helper()

	path := "../../rinha-de-backend-2026/resources/references.json.gz"
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("references.json.gz not found at %s, skipping validation", path)
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		t.Fatalf("gzip reader: %v", err)
	}
	defer gz.Close()

	var records []validationRec
	dec := json.NewDecoder(gz)
	if _, err := dec.Token(); err != nil {
		t.Fatalf("expect array start: %v", err)
	}
	for dec.More() {
		var rec validationRec
		if err := dec.Decode(&rec); err != nil {
			t.Fatalf("decode record: %v", err)
		}
		records = append(records, rec)
	}
	if _, err := dec.Token(); err != nil {
		t.Fatalf("expect array end: %v", err)
	}

	normCfg, err := norm.Load()
	if err != nil {
		t.Fatalf("norm load: %v", err)
	}

	mccRisks, err := mcc.Load()
	if err != nil {
		t.Fatalf("mcc load: %v", err)
	}

	vecs := make([]artifact.VectorRecord, len(records))
	for i, rec := range records {
		var vr artifact.VectorRecord
		for d, v := range rec.Vector {
			vr.Dims[d] = quantizeForTest(v)
		}
		switch rec.Label {
		case "fraud":
			vr.Label = artifact.LabelFraud
		case "legit":
			vr.Label = artifact.LabelLegit
		}
		vecs[i] = vr
	}

	return vecs, normCfg, mccRisks
}

func quantizeForTest(v float64) int16 {
	if v < 0 {
		return artifact.SentinelInt16
	}
	clamped := norm.Clamp(v, 0.0, 1.0)
	return int16(clamped * 32767)
}

func vecsToBytes(vecs []artifact.VectorRecord) []byte {
	buf := make([]byte, len(vecs)*artifact.VectorRecordSz)
	for i := range vecs {
		for d := 0; d < 14; d++ {
			buf[i*artifact.VectorRecordSz+d*2] = byte(uint16(vecs[i].Dims[d]))
			buf[i*artifact.VectorRecordSz+d*2+1] = byte(uint16(vecs[i].Dims[d]) >> 8)
		}
		buf[i*artifact.VectorRecordSz+28] = vecs[i].Label
	}
	return buf
}

// fullResult captures all neighbors from a brute-force search.
type fullResult struct {
	dists  []int64
	labels []uint8
	count  int
}

func (r *fullResult) Insert(dist int64, label uint8) {
	r.dists = append(r.dists, dist)
	r.labels = append(r.labels, label)
	r.count++
}

func (r *fullResult) MaxDist() int64 {
	// Capture all — never early exit
	return 1<<63 - 1
}

func (r *fullResult) FraudCount() int {
	n := 0
	for _, l := range r.labels {
		if l == artifact.LabelFraud {
			n++
		}
	}
	return n
}

func (r *fullResult) top5FraudCount() int {
	type pair struct {
		dist  int64
		label uint8
	}
	pairs := make([]pair, r.count)
	for i := 0; i < r.count; i++ {
		pairs[i] = pair{r.dists[i], r.labels[i]}
	}
	// Selection sort for top 5
	for i := 0; i < 5 && i < len(pairs); i++ {
		minIdx := i
		for j := i + 1; j < len(pairs); j++ {
			if pairs[j].dist < pairs[minIdx].dist {
				minIdx = j
			}
		}
		pairs[i], pairs[minIdx] = pairs[minIdx], pairs[i]
	}
	n := 0
	for i := 0; i < 5 && i < len(pairs); i++ {
		if pairs[i].label == artifact.LabelFraud {
			n++
		}
	}
	return n
}

func TestSubsetValidation(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping validation test in short mode")
	}

	// Load subset BFV1 artifact
	subsetArt, err := artifact.LoadFromFile("../artifact/artifact.bin")
	if err != nil {
		t.Fatalf("load BFV1 subset: %v", err)
	}
	if !subsetArt.IsBFV1() {
		t.Fatal("artifact is not BFV1, regenerate with: go generate ./internal/artifact/")
	}

	subsetVectors := subsetArt.VectorsData
	subsetCount := subsetArt.NumVectors()
	t.Logf("subset: %d vectors, %.2f MB", subsetCount, float64(len(subsetVectors))/1024/1024)

	// Load full references
	fullVectors, _, _ := loadFullReferences(t)
	fullBytes := vecsToBytes(fullVectors)
	t.Logf("full set: %d vectors, %.2f MB", len(fullVectors), float64(len(fullBytes))/1024/1024)

	// Select 1000 random queries from full set with deterministic seed
	rng := rand.New(rand.NewPCG(42, 42))
	queryCount := 1000
	if len(fullVectors) < queryCount {
		queryCount = len(fullVectors)
	}

	type queryPair struct {
		vec   [14]int16
		label uint8
	}
	queries := make([]queryPair, queryCount)
	indices := make([]int, len(fullVectors))
	for i := range indices {
		indices[i] = i
	}
	rng.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})
	for i := 0; i < queryCount; i++ {
		v := fullVectors[indices[i]]
		queries[i] = queryPair{vec: v.Dims, label: v.Label}
	}

	// Run validation
	agreeCount := 0

	for i, q := range queries {
		// Subset brute-force (use TopK for decisions)
		var subsetTK search.TopK
		bruteforce.SearchFlat(&q.vec, subsetVectors, subsetCount, &subsetTK)
		subsetFraud := subsetTK.FraudCount()
		_, subsetScore := search.Decide(subsetFraud)
		subsetDecision := subsetScore >= 0.6

		// Full brute-force (collect all)
		var fullRes fullResult
		bruteforce.SearchFlat(&q.vec, fullBytes, uint32(len(fullVectors)), &fullRes)
		fullFraudTop5 := fullRes.top5FraudCount()
		_, fullScore := search.Decide(fullFraudTop5)
		fullDecision := fullScore >= 0.6

		if subsetDecision == fullDecision {
			agreeCount++
		}

		if i%100 == 0 {
			t.Logf("progress: %d/%d, agreement so far: %d/%d", i, queryCount, agreeCount, i+1)
		}
	}

	agreement := float64(agreeCount) / float64(queryCount)
	t.Logf("decision_agreement: %.4f (%d/%d)", agreement, agreeCount, queryCount)

	if agreement < 0.995 {
		t.Errorf("DECISION AGREEMENT GATE FAILED: %.4f < 0.995", agreement)
	}
}
