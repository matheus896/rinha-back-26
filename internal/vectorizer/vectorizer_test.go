package vectorizer

import (
	"math"
	"testing"

	"rinha-backend-2026/internal/artifact"
	"rinha-backend-2026/internal/mcc"
	"rinha-backend-2026/internal/norm"
)

const epsilon = 1e-4

func floatEqual(a, b float64) bool {
	return math.Abs(a-b) < epsilon
}

func testNormCfg() *norm.Normalization {
	return &norm.Normalization{
		MaxAmount:            10000,
		MaxInstallments:      12,
		AmountVsAvgRatio:     10,
		MaxMinutes:           1440,
		MaxKm:                1000,
		MaxTxCount24h:        20,
		MaxMerchantAvgAmount: 10000,
	}
}

func testMCCRisks() mcc.Risks {
	return mcc.Risks{
		"5411": 0.15,
		"5912": 0.20,
		"7802": 0.75,
	}
}

func TestVectorize_Dim0_Amount(t *testing.T) {
	n := testNormCfg()
	r := testMCCRisks()

	tests := []struct {
		name     string
		amount   float64
		expected float64
	}{
		{"normal", 5000, 0.5},
		{"clamped", 15000, 1.0},
		{"small", 41.12, 0.004112},
		{"zero", 0, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &Payload{
				ID: "tx-test",
				Transaction: Transaction{
					Amount:       tt.amount,
					Installments: 1,
					RequestedAt:  "2026-03-11T12:00:00Z",
				},
				Customer: Customer{
					AvgAmount:      100,
					KnownMerchants: []string{},
				},
				Merchant:        Merchant{ID: "M", MCC: "5411", AvgAmount: 100},
				Terminal:        Terminal{},
				LastTransaction: nil,
			}
			v, err := Vectorize(p, n, r)
			if err != nil {
				t.Fatal(err)
			}
			if !floatEqual(v[0], tt.expected) {
				t.Errorf("dim 0: got %v, want %v", v[0], tt.expected)
			}
		})
	}
}

func TestVectorize_Dim1_Installments(t *testing.T) {
	n := testNormCfg()
	r := testMCCRisks()

	tests := []struct {
		name         string
		installments int
		expected     float64
	}{
		{"six", 6, 0.5},
		{"twelve", 12, 1.0},
		{"exceed", 24, 1.0},
		{"zero", 0, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &Payload{
				ID: "tx-test",
				Transaction: Transaction{
					Amount:       100,
					Installments: tt.installments,
					RequestedAt:  "2026-03-11T12:00:00Z",
				},
				Customer: Customer{
					AvgAmount:      100,
					KnownMerchants: []string{},
				},
				Merchant:        Merchant{ID: "M", MCC: "5411", AvgAmount: 100},
				Terminal:        Terminal{},
				LastTransaction: nil,
			}
			v, err := Vectorize(p, n, r)
			if err != nil {
				t.Fatal(err)
			}
			if v[1] != tt.expected {
				t.Errorf("dim 1: got %v, want %v", tt.expected, v[1])
			}
		})
	}
}

func TestVectorize_Dim2_AmountVsAvg(t *testing.T) {
	n := testNormCfg()
	r := testMCCRisks()

	p := &Payload{
		ID: "tx-test",
		Transaction: Transaction{
			Amount:       1000,
			Installments: 1,
			RequestedAt:  "2026-03-11T12:00:00Z",
		},
		Customer: Customer{
			AvgAmount:      200,
			KnownMerchants: []string{},
		},
		Merchant:        Merchant{ID: "M", MCC: "5411", AvgAmount: 100},
		Terminal:        Terminal{},
		LastTransaction: nil,
	}
	v, err := Vectorize(p, n, r)
	if err != nil {
		t.Fatal(err)
	}
	expected := (1000.0 / 200.0) / 10.0 // = 0.5
	if v[2] != expected {
		t.Errorf("dim 2: got %v, want %v", v[2], expected)
	}
}

func TestVectorize_Dim2_AmountVsAvgClamped(t *testing.T) {
	n := testNormCfg()
	r := testMCCRisks()

	p := &Payload{
		ID: "tx-test",
		Transaction: Transaction{
			Amount:       10000,
			Installments: 1,
			RequestedAt:  "2026-03-11T12:00:00Z",
		},
		Customer: Customer{
			AvgAmount:      50,
			KnownMerchants: []string{},
		},
		Merchant:        Merchant{ID: "M", MCC: "5411", AvgAmount: 100},
		Terminal:        Terminal{},
		LastTransaction: nil,
	}
	v, err := Vectorize(p, n, r)
	if err != nil {
		t.Fatal(err)
	}
	if v[2] != 1.0 {
		t.Errorf("dim 2 clamped: got %v, want 1.0", v[2])
	}
}

func TestVectorize_Dim3_HourOfDay(t *testing.T) {
	n := testNormCfg()
	r := testMCCRisks()

	tests := []struct {
		name     string
		ts       string
		expected float64
	}{
		{"midnight", "2026-03-11T00:00:00Z", 0.0},
		{"noon", "2026-03-11T12:00:00Z", 12.0 / 23.0},
		{"11pm", "2026-03-11T23:30:00Z", 23.0 / 23.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &Payload{
				ID: "tx-test",
				Transaction: Transaction{
					Amount:       100,
					Installments: 1,
					RequestedAt:  tt.ts,
				},
				Customer: Customer{
					AvgAmount:      100,
					KnownMerchants: []string{},
				},
				Merchant:        Merchant{ID: "M", MCC: "5411", AvgAmount: 100},
				Terminal:        Terminal{},
				LastTransaction: nil,
			}
			v, err := Vectorize(p, n, r)
			if err != nil {
				t.Fatal(err)
			}
			if v[3] != tt.expected {
				t.Errorf("dim 3: got %v, want %v", v[3], tt.expected)
			}
		})
	}
}

func TestVectorize_Dim4_DayOfWeek(t *testing.T) {
	n := testNormCfg()
	r := testMCCRisks()

	tests := []struct {
		name     string
		ts       string
		expected float64
	}{
		{"monday", "2026-03-09T12:00:00Z", 0.0 / 6.0},
		{"wednesday", "2026-03-11T12:00:00Z", 2.0 / 6.0},
		{"sunday", "2026-03-15T12:00:00Z", 6.0 / 6.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &Payload{
				ID: "tx-test",
				Transaction: Transaction{
					Amount:       100,
					Installments: 1,
					RequestedAt:  tt.ts,
				},
				Customer: Customer{
					AvgAmount:      100,
					KnownMerchants: []string{},
				},
				Merchant:        Merchant{ID: "M", MCC: "5411", AvgAmount: 100},
				Terminal:        Terminal{},
				LastTransaction: nil,
			}
			v, err := Vectorize(p, n, r)
			if err != nil {
				t.Fatal(err)
			}
			if v[4] != tt.expected {
				t.Errorf("dim 4: got %v, want %v", v[4], tt.expected)
			}
		})
	}
}

func TestVectorize_LastTransactionNull(t *testing.T) {
	n := testNormCfg()
	r := testMCCRisks()

	p := &Payload{
		ID: "tx-test",
		Transaction: Transaction{
			Amount:       100,
			Installments: 1,
			RequestedAt:  "2026-03-11T12:00:00Z",
		},
		Customer: Customer{
			AvgAmount:      100,
			KnownMerchants: []string{},
		},
		Merchant:        Merchant{ID: "M", MCC: "5411", AvgAmount: 100},
		Terminal:        Terminal{},
		LastTransaction: nil,
	}
	v, err := Vectorize(p, n, r)
	if err != nil {
		t.Fatal(err)
	}
	if v[5] != -1 {
		t.Errorf("dim 5 (minutes_since_last_tx) should be -1 for null, got %v", v[5])
	}
	if v[6] != -1 {
		t.Errorf("dim 6 (km_from_last_tx) should be -1 for null, got %v", v[6])
	}
}

func TestVectorize_LastTransactionPresent(t *testing.T) {
	n := testNormCfg()
	r := testMCCRisks()

	p := &Payload{
		ID: "tx-test",
		Transaction: Transaction{
			Amount:       100,
			Installments: 1,
			RequestedAt:  "2026-03-11T12:00:00Z",
		},
		Customer: Customer{
			AvgAmount:      100,
			KnownMerchants: []string{},
		},
		Merchant: Merchant{ID: "M", MCC: "5411", AvgAmount: 100},
		Terminal: Terminal{},
		LastTransaction: &LastTransaction{
			Timestamp:     "2026-03-11T11:00:00Z",
			KmFromCurrent: 500,
		},
	}
	v, err := Vectorize(p, n, r)
	if err != nil {
		t.Fatal(err)
	}
	if v[5] == -1 {
		t.Error("dim 5 should not be -1 when last_transaction present")
	}
	expectedMinutes := norm.Clamp(60.0/n.MaxMinutes, 0, 1)
	if v[5] != expectedMinutes {
		t.Errorf("dim 5: got %v, want %v", v[5], expectedMinutes)
	}
	expectedKm := norm.Clamp(500.0/n.MaxKm, 0, 1)
	if v[6] != expectedKm {
		t.Errorf("dim 6: got %v, want %v", v[6], expectedKm)
	}
}

func TestVectorize_Dim7_KmFromHome(t *testing.T) {
	n := testNormCfg()
	r := testMCCRisks()

	p := &Payload{
		ID: "tx-test",
		Transaction: Transaction{
			Amount:       100,
			Installments: 1,
			RequestedAt:  "2026-03-11T12:00:00Z",
		},
		Customer: Customer{
			AvgAmount:      100,
			KnownMerchants: []string{},
		},
		Merchant:        Merchant{ID: "M", MCC: "5411", AvgAmount: 100},
		Terminal:        Terminal{KmFromHome: 500},
		LastTransaction: nil,
	}
	v, err := Vectorize(p, n, r)
	if err != nil {
		t.Fatal(err)
	}
	expected := norm.Clamp(500.0/n.MaxKm, 0, 1)
	if v[7] != expected {
		t.Errorf("dim 7: got %v, want %v", v[7], expected)
	}
}

func TestVectorize_Dim8_TxCount24h(t *testing.T) {
	n := testNormCfg()
	r := testMCCRisks()

	tests := []struct {
		name   string
		count  int
		expect float64
	}{
		{"zero", 0, 0},
		{"half", 10, 0.5},
		{"clamped", 25, 1.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &Payload{
				ID: "tx-test",
				Transaction: Transaction{
					Amount:       100,
					Installments: 1,
					RequestedAt:  "2026-03-11T12:00:00Z",
				},
				Customer: Customer{
					AvgAmount:      100,
					TxCount24h:     tt.count,
					KnownMerchants: []string{},
				},
				Merchant:        Merchant{ID: "M", MCC: "5411", AvgAmount: 100},
				Terminal:        Terminal{},
				LastTransaction: nil,
			}
			v, err := Vectorize(p, n, r)
			if err != nil {
				t.Fatal(err)
			}
			if v[8] != tt.expect {
				t.Errorf("dim 8: got %v, want %v", v[8], tt.expect)
			}
		})
	}
}

func TestVectorize_Dim9_IsOnline(t *testing.T) {
	n := testNormCfg()
	r := testMCCRisks()

	tests := []struct {
		name     string
		online   bool
		expected float64
	}{
		{"online", true, 1.0},
		{"not_online", false, 0.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &Payload{
				ID: "tx-test",
				Transaction: Transaction{
					Amount:       100,
					Installments: 1,
					RequestedAt:  "2026-03-11T12:00:00Z",
				},
				Customer: Customer{
					AvgAmount:      100,
					KnownMerchants: []string{},
				},
				Merchant:        Merchant{ID: "M", MCC: "5411", AvgAmount: 100},
				Terminal:        Terminal{IsOnline: tt.online},
				LastTransaction: nil,
			}
			v, err := Vectorize(p, n, r)
			if err != nil {
				t.Fatal(err)
			}
			if v[9] != tt.expected {
				t.Errorf("dim 9: got %v, want %v", v[9], tt.expected)
			}
		})
	}
}

func TestVectorize_Dim10_CardPresent(t *testing.T) {
	n := testNormCfg()
	r := testMCCRisks()

	tests := []struct {
		name     string
		present  bool
		expected float64
	}{
		{"card_present", true, 1.0},
		{"card_not_present", false, 0.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &Payload{
				ID: "tx-test",
				Transaction: Transaction{
					Amount:       100,
					Installments: 1,
					RequestedAt:  "2026-03-11T12:00:00Z",
				},
				Customer: Customer{
					AvgAmount:      100,
					KnownMerchants: []string{},
				},
				Merchant:        Merchant{ID: "M", MCC: "5411", AvgAmount: 100},
				Terminal:        Terminal{CardPresent: tt.present},
				LastTransaction: nil,
			}
			v, err := Vectorize(p, n, r)
			if err != nil {
				t.Fatal(err)
			}
			if v[10] != tt.expected {
				t.Errorf("dim 10: got %v, want %v", v[10], tt.expected)
			}
		})
	}
}

func TestVectorize_Dim11_UnknownMerchant(t *testing.T) {
	n := testNormCfg()
	r := testMCCRisks()

	tests := []struct {
		name           string
		merchantID     string
		knownMerchants []string
		expected       float64
	}{
		{"known", "MERC-001", []string{"MERC-001", "MERC-002"}, 0.0},
		{"unknown", "MERC-999", []string{"MERC-001", "MERC-002"}, 1.0},
		{"empty_known", "MERC-001", []string{}, 1.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &Payload{
				ID: "tx-test",
				Transaction: Transaction{
					Amount:       100,
					Installments: 1,
					RequestedAt:  "2026-03-11T12:00:00Z",
				},
				Customer: Customer{
					AvgAmount:      100,
					KnownMerchants: tt.knownMerchants,
				},
				Merchant:        Merchant{ID: tt.merchantID, MCC: "5411", AvgAmount: 100},
				Terminal:        Terminal{},
				LastTransaction: nil,
			}
			v, err := Vectorize(p, n, r)
			if err != nil {
				t.Fatal(err)
			}
			if v[11] != tt.expected {
				t.Errorf("dim 11: got %v, want %v", v[11], tt.expected)
			}
		})
	}
}

func TestVectorize_Dim12_MCCRisk(t *testing.T) {
	n := testNormCfg()
	r := testMCCRisks()

	tests := []struct {
		name     string
		mcc      string
		expected float64
	}{
		{"known_mcc", "5411", 0.15},
		{"unknown_mcc_default", "9999", 0.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &Payload{
				ID: "tx-test",
				Transaction: Transaction{
					Amount:       100,
					Installments: 1,
					RequestedAt:  "2026-03-11T12:00:00Z",
				},
				Customer: Customer{
					AvgAmount:      100,
					KnownMerchants: []string{},
				},
				Merchant:        Merchant{ID: "M", MCC: tt.mcc, AvgAmount: 100},
				Terminal:        Terminal{},
				LastTransaction: nil,
			}
			v, err := Vectorize(p, n, r)
			if err != nil {
				t.Fatal(err)
			}
			if v[12] != tt.expected {
				t.Errorf("dim 12: got %v, want %v", v[12], tt.expected)
			}
		})
	}
}

func TestVectorize_Dim13_MerchantAvgAmount(t *testing.T) {
	n := testNormCfg()
	r := testMCCRisks()

	tests := []struct {
		name     string
		avg      float64
		expected float64
	}{
		{"normal", 5000, 0.5},
		{"clamped", 15000, 1.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &Payload{
				ID: "tx-test",
				Transaction: Transaction{
					Amount:       100,
					Installments: 1,
					RequestedAt:  "2026-03-11T12:00:00Z",
				},
				Customer: Customer{
					AvgAmount:      100,
					KnownMerchants: []string{},
				},
				Merchant:        Merchant{ID: "M", MCC: "5411", AvgAmount: tt.avg},
				Terminal:        Terminal{},
				LastTransaction: nil,
			}
			v, err := Vectorize(p, n, r)
			if err != nil {
				t.Fatal(err)
			}
			if v[13] != tt.expected {
				t.Errorf("dim 13: got %v, want %v", v[13], tt.expected)
			}
		})
	}
}

func TestVectorizeInt16_NullLastTx(t *testing.T) {
	n := testNormCfg()
	r := testMCCRisks()

	p := &Payload{
		ID: "tx-test",
		Transaction: Transaction{
			Amount:       100,
			Installments: 1,
			RequestedAt:  "2026-03-11T12:00:00Z",
		},
		Customer: Customer{
			AvgAmount:      100,
			KnownMerchants: []string{},
		},
		Merchant:        Merchant{ID: "M", MCC: "5411", AvgAmount: 100},
		Terminal:        Terminal{},
		LastTransaction: nil,
	}
	iv, err := VectorizeInt16(p, n, r)
	if err != nil {
		t.Fatal(err)
	}
	if iv[5] != artifact.SentinelInt16 {
		t.Errorf("dim 5 int16: got %d, want %d", iv[5], artifact.SentinelInt16)
	}
	if iv[6] != artifact.SentinelInt16 {
		t.Errorf("dim 6 int16: got %d, want %d", iv[6], artifact.SentinelInt16)
	}
}

func TestVectorizeInt16_PresentLastTx(t *testing.T) {
	n := testNormCfg()
	r := testMCCRisks()

	p := &Payload{
		ID: "tx-test",
		Transaction: Transaction{
			Amount:       100,
			Installments: 1,
			RequestedAt:  "2026-03-11T12:00:00Z",
		},
		Customer: Customer{
			AvgAmount:      100,
			KnownMerchants: []string{},
		},
		Merchant: Merchant{ID: "M", MCC: "5411", AvgAmount: 100},
		Terminal: Terminal{},
		LastTransaction: &LastTransaction{
			Timestamp:     "2026-03-11T11:50:00Z",
			KmFromCurrent: 0,
		},
	}
	iv, err := VectorizeInt16(p, n, r)
	if err != nil {
		t.Fatal(err)
	}
	if iv[5] == artifact.SentinelInt16 {
		t.Error("dim 5 should not be sentinel when last_transaction present")
	}
}

func TestVectorize_DetectLegitTransaction(t *testing.T) {
	n := testNormCfg()
	r := testMCCRisks()

	p := &Payload{
		ID: "tx-1329056812",
		Transaction: Transaction{
			Amount:       41.12,
			Installments: 2,
			RequestedAt:  "2026-03-11T18:45:53Z",
		},
		Customer: Customer{
			AvgAmount:      82.24,
			TxCount24h:     3,
			KnownMerchants: []string{"MERC-003", "MERC-016"},
		},
		Merchant:        Merchant{ID: "MERC-016", MCC: "5411", AvgAmount: 60.25},
		Terminal:        Terminal{IsOnline: false, CardPresent: true, KmFromHome: 29.23},
		LastTransaction: nil,
	}
	v, err := Vectorize(p, n, r)
	if err != nil {
		t.Fatal(err)
	}

	if !floatEqual(v[0], 0.0041) {
		t.Errorf("dim 0: got %v, want ~0.0041", v[0])
	}
	if v[5] != -1 {
		t.Errorf("dim 5: got %v, want -1 (null last_transaction)", v[5])
	}
	if v[6] != -1 {
		t.Errorf("dim 6: got %v, want -1 (null last_transaction)", v[6])
	}
	if v[11] != 0 {
		t.Errorf("dim 11: got %v, want 0 (known merchant)", v[11])
	}
	if !floatEqual(v[12], 0.15) {
		t.Errorf("dim 12: got %v, want 0.15 (MCC 5411)", v[12])
	}
}

func TestVectorize_DetectFraudTransaction(t *testing.T) {
	n := testNormCfg()
	r := testMCCRisks()

	p := &Payload{
		ID: "tx-3330991687",
		Transaction: Transaction{
			Amount:       9505.97,
			Installments: 10,
			RequestedAt:  "2026-03-14T05:15:12Z",
		},
		Customer: Customer{
			AvgAmount:      81.28,
			TxCount24h:     20,
			KnownMerchants: []string{"MERC-008", "MERC-007", "MERC-005"},
		},
		Merchant:        Merchant{ID: "MERC-068", MCC: "7802", AvgAmount: 54.86},
		Terminal:        Terminal{IsOnline: false, CardPresent: true, KmFromHome: 952.27},
		LastTransaction: nil,
	}
	v, err := Vectorize(p, n, r)
	if err != nil {
		t.Fatal(err)
	}

	if !floatEqual(v[0], 0.9506) {
		t.Errorf("dim 0: got %v, want ~0.9506", v[0])
	}
	if v[5] != -1 {
		t.Errorf("dim 5: got %v, want -1 (null last_transaction)", v[5])
	}
	if v[11] != 1 {
		t.Errorf("dim 11: got %v, want 1 (unknown merchant)", v[11])
	}
	if !floatEqual(v[12], 0.75) {
		t.Errorf("dim 12: got %v, want 0.75 (MCC 7802)", v[12])
	}
}

func TestVectorize_InvalidTimestamp(t *testing.T) {
	n := testNormCfg()
	r := testMCCRisks()

	p := &Payload{
		ID: "tx-test",
		Transaction: Transaction{
			Amount:       100,
			Installments: 1,
			RequestedAt:  "not-a-timestamp",
		},
		Customer: Customer{
			AvgAmount:      100,
			KnownMerchants: []string{},
		},
		Merchant:        Merchant{ID: "M", MCC: "5411", AvgAmount: 100},
		Terminal:        Terminal{},
		LastTransaction: nil,
	}
	_, err := Vectorize(p, n, r)
	if err == nil {
		t.Error("expected error for invalid timestamp")
	}
}

func TestVectorize_InvalidLastTxTimestamp(t *testing.T) {
	n := testNormCfg()
	r := testMCCRisks()

	p := &Payload{
		ID: "tx-test",
		Transaction: Transaction{
			Amount:       100,
			Installments: 1,
			RequestedAt:  "2026-03-11T12:00:00Z",
		},
		Customer: Customer{
			AvgAmount:      100,
			KnownMerchants: []string{},
		},
		Merchant: Merchant{ID: "M", MCC: "5411", AvgAmount: 100},
		Terminal: Terminal{},
		LastTransaction: &LastTransaction{
			Timestamp:     "bad-timestamp",
			KmFromCurrent: 1,
		},
	}
	_, err := Vectorize(p, n, r)
	if err == nil {
		t.Error("expected error for invalid last_transaction timestamp")
	}
}
