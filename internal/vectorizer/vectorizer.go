package vectorizer

import (
	"time"

	"rinha-backend-2026/internal/mcc"
	"rinha-backend-2026/internal/norm"
)

type Payload struct {
	ID              string           `json:"id"`
	Transaction     Transaction      `json:"transaction"`
	Customer        Customer         `json:"customer"`
	Merchant        Merchant         `json:"merchant"`
	Terminal        Terminal         `json:"terminal"`
	LastTransaction *LastTransaction `json:"last_transaction"`
}

type Transaction struct {
	Amount       float64 `json:"amount"`
	Installments int     `json:"installments"`
	RequestedAt  string  `json:"requested_at"`
}

type Customer struct {
	AvgAmount      float64  `json:"avg_amount"`
	TxCount24h     int      `json:"tx_count_24h"`
	KnownMerchants []string `json:"known_merchants"`
}

type Merchant struct {
	ID        string  `json:"id"`
	MCC       string  `json:"mcc"`
	AvgAmount float64 `json:"avg_amount"`
}

type Terminal struct {
	IsOnline    bool    `json:"is_online"`
	CardPresent bool    `json:"card_present"`
	KmFromHome  float64 `json:"km_from_home"`
}

type LastTransaction struct {
	Timestamp     string  `json:"timestamp"`
	KmFromCurrent float64 `json:"km_from_current"`
}

func Vectorize(p *Payload, n *norm.Normalization, risks mcc.Risks) ([14]float64, error) {
	var v [14]float64

	timestamp, err := time.Parse(time.RFC3339, p.Transaction.RequestedAt)
	if err != nil {
		return v, err
	}

	v[0] = norm.Clamp(p.Transaction.Amount/n.MaxAmount, 0, 1)
	v[1] = norm.Clamp(float64(p.Transaction.Installments)/n.MaxInstallments, 0, 1)

	amountVsAvg := (p.Transaction.Amount / p.Customer.AvgAmount) / n.AmountVsAvgRatio
	v[2] = norm.Clamp(amountVsAvg, 0, 1)

	v[3] = float64(timestamp.Hour()) / 23.0

	specDay := ((int(timestamp.Weekday()) + 6) % 7)
	v[4] = float64(specDay) / 6.0

	if p.LastTransaction == nil {
		v[5] = -1
		v[6] = -1
	} else {
		prevTs, err := time.Parse(time.RFC3339, p.LastTransaction.Timestamp)
		if err != nil {
			return v, err
		}
		minutesSince := timestamp.Sub(prevTs).Minutes()
		v[5] = norm.Clamp(minutesSince/n.MaxMinutes, 0, 1)

		v[6] = norm.Clamp(p.LastTransaction.KmFromCurrent/n.MaxKm, 0, 1)
	}

	v[7] = norm.Clamp(p.Terminal.KmFromHome/n.MaxKm, 0, 1)
	v[8] = norm.Clamp(float64(p.Customer.TxCount24h)/n.MaxTxCount24h, 0, 1)

	v[9] = float64(BoolToInt8(p.Terminal.IsOnline))
	v[10] = float64(BoolToInt8(p.Terminal.CardPresent))

	if contains(p.Customer.KnownMerchants, p.Merchant.ID) {
		v[11] = 0
	} else {
		v[11] = 1
	}

	v[12] = risks.Lookup(p.Merchant.MCC)
	v[13] = norm.Clamp(p.Merchant.AvgAmount/n.MaxMerchantAvgAmount, 0, 1)

	return v, nil
}

func VectorizeInt16(p *Payload, n *norm.Normalization, risks mcc.Risks) ([14]int16, error) {
	fv, err := Vectorize(p, n, risks)
	if err != nil {
		return [14]int16{}, err
	}
	var iv [14]int16
	for i, f := range fv {
		iv[i] = QuantizeInt16(f)
	}
	return iv, nil
}

func contains(slice []string, target string) bool {
	for _, s := range slice {
		if s == target {
			return true
		}
	}
	return false
}
