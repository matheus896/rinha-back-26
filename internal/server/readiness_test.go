package server

import (
	"testing"

	"github.com/valyala/fasthttp"
)

func TestReadiness_NotReady(t *testing.T) {
	rdy := NewReadiness()

	if rdy.IsReady() {
		t.Error("expected not ready initially")
	}
}

func TestReadiness_SetReady(t *testing.T) {
	rdy := NewReadiness()
	rdy.SetReady()

	if !rdy.IsReady() {
		t.Error("expected ready after SetReady")
	}
}

func TestReadiness_HandlerReturns503WhenNotReady(t *testing.T) {
	rdy := NewReadiness()

	handler := &FraudScoreHandler{
		Readiness: rdy,
	}

	var ctx fasthttp.RequestCtx
	ctx.Request.Header.SetMethod("POST")
	ctx.Request.SetRequestURI("/fraud-score")

	handler.HandleFastHTTP(&ctx)

	if ctx.Response.StatusCode() != fasthttp.StatusServiceUnavailable {
		t.Errorf("expected 503 when not ready, got %d", ctx.Response.StatusCode())
	}
}

func TestReadiness_HandlerReturns200EvenWithInvalidBody(t *testing.T) {
	rdy := NewReadiness()
	rdy.SetReady()

	handler := &FraudScoreHandler{
		Readiness: rdy,
	}

	var ctx fasthttp.RequestCtx
	ctx.Request.Header.SetMethod("POST")
	ctx.Request.SetRequestURI("/fraud-score")
	ctx.Request.SetBody([]byte("invalid"))

	handler.HandleFastHTTP(&ctx)

	if ctx.Response.StatusCode() != fasthttp.StatusOK {
		t.Errorf("expected 200 even with invalid body (containment), got %d", ctx.Response.StatusCode())
	}
}
