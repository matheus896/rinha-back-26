package server

import (
	"encoding/json"
	"log/slog"

	"github.com/valyala/fasthttp"

	"rinha-backend-2026/internal/mcc"
	"rinha-backend-2026/internal/norm"
	"rinha-backend-2026/internal/search"
	"rinha-backend-2026/internal/vectorizer"
)

var fraudResponseBodies = [...][]byte{
	[]byte(`{"approved":true,"fraud_score":0.0}`),
	[]byte(`{"approved":true,"fraud_score":0.2}`),
	[]byte(`{"approved":true,"fraud_score":0.4}`),
	[]byte(`{"approved":false,"fraud_score":0.6}`),
	[]byte(`{"approved":false,"fraud_score":0.8}`),
	[]byte(`{"approved":false,"fraud_score":1.0}`),
}

var errResponseBody = fraudResponseBodies[0]

type FraudScoreHandler struct {
	Engine    *search.Engine
	NormCfg   *norm.Normalization
	MCCRisks  mcc.Risks
	Readiness *Readiness
}

func (h *FraudScoreHandler) HandleFastHTTP(ctx *fasthttp.RequestCtx) {
	if !h.Readiness.IsReady() {
		ctx.SetContentType("application/json")
		ctx.SetStatusCode(fasthttp.StatusServiceUnavailable)
		ctx.SetBodyString(`{"status":"not ready"}`)
		return
	}

	var payload vectorizer.Payload
	if err := json.Unmarshal(ctx.PostBody(), &payload); err != nil {
		slog.Error("failed to decode payload", "error", err)
		writeFraudResponse(ctx, errResponseBody)
		return
	}

	queryVec, err := vectorizer.VectorizeInt16(&payload, h.NormCfg, h.MCCRisks)
	if err != nil {
		slog.Error("failed to vectorize", "error", err)
		writeFraudResponse(ctx, errResponseBody)
		return
	}

	topK, err := h.Engine.Search(&queryVec)
	if err != nil {
		slog.Error("search failed", "error", err)
		writeFraudResponse(ctx, errResponseBody)
		return
	}

	fraudCount := topK.FraudCount()
	body := errResponseBody
	if fraudCount >= 0 && fraudCount <= 5 {
		body = fraudResponseBodies[fraudCount]
	}
	writeFraudResponse(ctx, body)
}

func writeFraudResponse(ctx *fasthttp.RequestCtx, body []byte) {
	ctx.SetContentType("application/json")
	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetBody(body)
}
