//go:build validation

package validation

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"

	"rinha-backend-2026/internal/mcc"
	"rinha-backend-2026/internal/norm"
	"rinha-backend-2026/internal/vectorizer"
)

type refEntry struct {
	vector [14]float64
	fraud  bool
}

type pairFloat struct {
	dist  float64
	fraud bool
}

type pairInt64 struct {
	dist  int64
	fraud bool
}

func TestQuantizationDivergence(t *testing.T) {
	refs := loadRefs(t, 50000)

	n, err := norm.Load()
	if err != nil {
		t.Fatal(err)
	}

	r, err := mcc.Load()
	if err != nil {
		t.Fatal(err)
	}

	payloads := generateRandomPayloads(2000)

	divergences := 0
	total := 0
	for i := range payloads {
		p := &payloads[i]

		fv, err := vectorizer.Vectorize(p, n, r)
		if err != nil {
			continue
		}

		iv, err := vectorizer.VectorizeInt16(p, n, r)
		if err != nil {
			continue
		}

		floatApproved := knnFloat(fv, refs)
		quantizedApproved := knnInt16(iv, refs)

		if floatApproved != quantizedApproved {
			divergences++
		}
		total++
	}

	divRate := float64(divergences) / float64(total)
	t.Logf("Quantization divergence (int16 vs float64): %.4f%% (%d/%d)", divRate*100, divergences, total)

	if divRate > 0.001 {
		t.Errorf("Divergence %.4f%% exceeds 0.1%% threshold", divRate*100)
	}
}

func TestRecallGate(t *testing.T) {
	refs := loadRefs(t, 50000)

	n, err := norm.Load()
	if err != nil {
		t.Fatal(err)
	}

	r, err := mcc.Load()
	if err != nil {
		t.Fatal(err)
	}

	payloads := generateRandomPayloads(500)

	totalIntersection := 0
	totalProcessed := 0

	for i := range payloads {
		p := &payloads[i]

		fv, err := vectorizer.Vectorize(p, n, r)
		if err != nil {
			continue
		}

		iv, err := vectorizer.VectorizeInt16(p, n, r)
		if err != nil {
			continue
		}

		floatTop5 := top5IndicesFloat(fv, refs)
		quantizedTop5 := top5IndicesInt16(iv, refs)

		matches := intersectCount(floatTop5, quantizedTop5)
		totalIntersection += matches
		totalProcessed += 5
	}

	recall := float64(totalIntersection) / float64(totalProcessed)
	t.Logf("Top-5 recall (int16 vs float64): %.4f (%d/%d)", recall, totalIntersection, totalProcessed)

	if recall < 0.99 {
		t.Errorf("Recall %.4f below minimum 0.99", recall)
	}
}

func knnFloat(query [14]float64, refs []refEntry) bool {
	const k = 5
	top := make([]pairFloat, 0, k)

	for _, ref := range refs {
		d := euclidean(query, ref.vector)
		if len(top) < k {
			top = append(top, pairFloat{dist: d, fraud: ref.fraud})
			sortPairFloat(top)
		} else if d < top[k-1].dist {
			top[k-1] = pairFloat{dist: d, fraud: ref.fraud}
			sortPairFloat(top)
		}
	}

	frauds := 0
	for _, p := range top {
		if p.fraud {
			frauds++
		}
	}
	return float64(frauds)/float64(k) < 0.6
}

func knnInt16(query [14]int16, refs []refEntry) bool {
	const k = 5
	top := make([]pairInt64, 0, k)

	for _, ref := range refs {
		d := distInt16(query, ref.vector)
		if len(top) < k {
			top = append(top, pairInt64{dist: d, fraud: ref.fraud})
			sortPairInt64(top)
		} else if d < top[k-1].dist {
			top[k-1] = pairInt64{dist: d, fraud: ref.fraud}
			sortPairInt64(top)
		}
	}

	frauds := 0
	for _, p := range top {
		if p.fraud {
			frauds++
		}
	}
	return float64(frauds)/float64(k) < 0.6
}

func sortPairFloat(pairs []pairFloat) {
	for i := 0; i < len(pairs)-1; i++ {
		for j := i + 1; j < len(pairs); j++ {
			if pairs[j].dist < pairs[i].dist {
				pairs[i], pairs[j] = pairs[j], pairs[i]
			}
		}
	}
}

func sortPairInt64(pairs []pairInt64) {
	for i := 0; i < len(pairs)-1; i++ {
		for j := i + 1; j < len(pairs); j++ {
			if pairs[j].dist < pairs[i].dist {
				pairs[i], pairs[j] = pairs[j], pairs[i]
			}
		}
	}
}

func top5IndicesFloat(query [14]float64, refs []refEntry) []int {
	const k = 5
	type scored struct {
		dist  float64
		index int
	}
	top := make([]scored, k)
	for i := range top {
		top[i] = scored{dist: math.MaxFloat64, index: -1}
	}

	for idx, ref := range refs {
		d := euclidean(query, ref.vector)
		if d < top[k-1].dist {
			ins := 0
			for ins < k && top[ins].dist < d {
				ins++
			}
			if ins < k {
				copy(top[ins+1:], top[ins:k-1])
				top[ins] = scored{dist: d, index: idx}
			}
		}
	}

	out := make([]int, k)
	for i, sc := range top {
		out[i] = sc.index
	}
	return out
}

func top5IndicesInt16(query [14]int16, refs []refEntry) []int {
	const k = 5
	type scored struct {
		dist  int64
		index int
	}
	top := make([]scored, k)
	for i := range top {
		top[i] = scored{dist: math.MaxInt64, index: -1}
	}

	for idx, ref := range refs {
		d := distInt16(query, ref.vector)
		if d < top[k-1].dist {
			ins := 0
			for ins < k && top[ins].dist < d {
				ins++
			}
			if ins < k {
				copy(top[ins+1:], top[ins:k-1])
				top[ins] = scored{dist: d, index: idx}
			}
		}
	}

	out := make([]int, k)
	for i, sc := range top {
		out[i] = sc.index
	}
	return out
}

func distInt16(q [14]int16, r [14]float64) int64 {
	var sum int64
	for i := 0; i < 14; i++ {
		ri := vectorizer.QuantizeInt16(r[i])
		diff := int64(q[i]) - int64(ri)
		sum += diff * diff
	}
	return sum
}

func euclidean(a, b [14]float64) float64 {
	var sum float64
	for i := 0; i < 14; i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

func intersectCount(groundTruth, candidates []int) int {
	set := make(map[int]bool, len(groundTruth))
	for _, idx := range groundTruth {
		set[idx] = true
	}
	count := 0
	for _, idx := range candidates {
		if set[idx] {
			count++
		}
	}
	return count
}

func loadRefs(t *testing.T, limit int) []refEntry {
	t.Helper()
	data, err := os.ReadFile("../../rinha-de-backend-2026/resources/references.json.gz")
	if err != nil {
		t.Skipf("skipping: cannot read references.json.gz: %v", err)
	}

	gr, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		t.Skipf("skipping: cannot decompress: %v", err)
	}
	defer gr.Close()

	dec := json.NewDecoder(gr)

	if _, err := dec.Token(); err != nil {
		t.Fatalf("expected array: %v", err)
	}

	type rawRef struct {
		Vector [14]float64 `json:"vector"`
		Label  string      `json:"label"`
	}

	refs := make([]refEntry, 0, limit)
	for dec.More() && len(refs) < limit {
		var rec rawRef
		if err := dec.Decode(&rec); err != nil {
			t.Fatalf("decode: %v", err)
		}
		refs = append(refs, refEntry{
			vector: rec.Vector,
			fraud:  rec.Label == "fraud",
		})
	}
	return refs
}

func generateRandomPayloads(n int) []vectorizer.Payload {
	amounts := []float64{41.12, 384.88, 9505.97, 150.0, 5000.0, 12000.0, 1.0, 9999.0}
	mccs := []string{"5411", "5812", "5912", "5944", "7801", "7802", "7995", "5999", "4511", "5311"}
	merchIDs := []string{"MERC-001", "MERC-016", "MERC-068", "MERC-003", "MERC-999"}
	knownSets := [][]string{
		{"MERC-003", "MERC-016"},
		{},
		{"MERC-008", "MERC-007", "MERC-005"},
		{"MERC-001", "MERC-999"},
	}
	hasLastTx := []bool{true, false}
	onlineVals := []bool{true, false}
	cardVals := []bool{true, false}

	rng := rand.New(rand.NewSource(42))
	payloads := make([]vectorizer.Payload, n)

	for i := 0; i < n; i++ {
		p := &payloads[i]
		p.ID = fmt.Sprintf("tx-val-%d", i)
		p.Transaction.Amount = amounts[rng.Intn(len(amounts))]
		p.Transaction.Installments = rng.Intn(13) + 1
		p.Transaction.RequestedAt = fmt.Sprintf("2026-03-%02dT%02d:%02d:%02dZ",
			rng.Intn(28)+1, rng.Intn(24), rng.Intn(60), rng.Intn(60))

		p.Customer.AvgAmount = float64(rng.Intn(1000)) + rng.Float64()*100
		p.Customer.TxCount24h = rng.Intn(25)
		p.Customer.KnownMerchants = knownSets[rng.Intn(len(knownSets))]

		p.Merchant.ID = merchIDs[rng.Intn(len(merchIDs))]
		p.Merchant.MCC = mccs[rng.Intn(len(mccs))]
		p.Merchant.AvgAmount = float64(rng.Intn(10000)) + rng.Float64()*100

		p.Terminal.IsOnline = onlineVals[rng.Intn(2)]
		p.Terminal.CardPresent = cardVals[rng.Intn(2)]
		p.Terminal.KmFromHome = rng.Float64() * 1000

		if hasLastTx[rng.Intn(2)] {
			p.LastTransaction = &vectorizer.LastTransaction{
				Timestamp: fmt.Sprintf("2026-03-%02dT%02d:%02d:%02dZ",
					rng.Intn(28)+1, rng.Intn(24), rng.Intn(60), rng.Intn(60)),
				KmFromCurrent: rng.Float64() * 1000,
			}
		}
	}
	return payloads
}
