package norm

import (
	_ "embed"
	"encoding/json"
)

//go:embed normalization.json
var normData []byte

type Normalization struct {
	MaxAmount             float64 `json:"max_amount"`
	MaxInstallments       float64 `json:"max_installments"`
	AmountVsAvgRatio      float64 `json:"amount_vs_avg_ratio"`
	MaxMinutes            float64 `json:"max_minutes"`
	MaxKm                 float64 `json:"max_km"`
	MaxTxCount24h         float64 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount  float64 `json:"max_merchant_avg_amount"`
}

func Load() (*Normalization, error) {
	var n Normalization
	if err := json.Unmarshal(normData, &n); err != nil {
		return nil, err
	}
	return &n, nil
}

func Clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
