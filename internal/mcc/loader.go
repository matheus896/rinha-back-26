package mcc

import (
	_ "embed"
	"encoding/json"
)

//go:embed mcc_risk.json
var mccData []byte

const DefaultRisk = 0.5

type Risks map[string]float64

func Load() (Risks, error) {
	var r Risks
	if err := json.Unmarshal(mccData, &r); err != nil {
		return nil, err
	}
	return r, nil
}

func (r Risks) Lookup(mcc string) float64 {
	if v, ok := r[mcc]; ok {
		return v
	}
	return DefaultRisk
}
