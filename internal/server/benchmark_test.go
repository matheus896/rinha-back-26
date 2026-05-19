package server

import (
	"log/slog"
	"testing"

	"github.com/valyala/fasthttp"

	"rinha-backend-2026/internal/artifact"
	"rinha-backend-2026/internal/ivf"
	"rinha-backend-2026/internal/mcc"
	"rinha-backend-2026/internal/norm"
	"rinha-backend-2026/internal/search"
)

var testPayload = []byte(`{
	"id": "tx-1329056812",
	"transaction": {
		"amount": 41.12,
		"installments": 2,
		"requested_at": "2026-03-11T18:45:53Z"
	},
	"customer": {
		"avg_amount": 82.24,
		"tx_count_24h": 3,
		"known_merchants": ["MERC-003", "MERC-016"]
	},
	"merchant": {
		"id": "MERC-016",
		"mcc": "5411",
		"avg_amount": 60.25
	},
	"terminal": {
		"is_online": false,
		"card_present": true,
		"km_from_home": 29.23
	},
	"last_transaction": null
}`)

func setupHandler(b *testing.B) *FraudScoreHandler {
	b.Helper()
	art, err := artifact.LoadFromFile("../artifact/artifact.bin")
	if err != nil {
		b.Fatalf("load artifact: %v", err)
	}
	mccRisks, err := mcc.Load()
	if err != nil {
		b.Fatalf("load mcc: %v", err)
	}
	normCfg, err := norm.Load()
	if err != nil {
		b.Fatalf("load norm: %v", err)
	}
	engine := search.NewEngine(art, ivf.Config{K: int(art.NumClusters()), NProbe: 8, RetryExtra: 8})
	readiness := &Readiness{}
	readiness.SetReady()
	return &FraudScoreHandler{
		Engine:    engine,
		NormCfg:   normCfg,
		MCCRisks:  mccRisks,
		Readiness: readiness,
	}
}

func BenchmarkHandlerE2E(b *testing.B) {
	handler := setupHandler(b)
	slog.SetLogLoggerLevel(slog.LevelError)

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		var ctx fasthttp.RequestCtx
		ctx.Request.Header.SetMethod("POST")
		ctx.Request.SetRequestURI("/fraud-score")
		ctx.Request.SetBody(testPayload)

		handler.HandleFastHTTP(&ctx)

		if ctx.Response.StatusCode() != fasthttp.StatusOK {
			b.Fatalf("expected 200, got %d: %s", ctx.Response.StatusCode(), ctx.Response.Body())
		}
	}
}
