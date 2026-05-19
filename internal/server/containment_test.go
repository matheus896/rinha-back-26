package server

import (
	"testing"

	"github.com/valyala/fasthttp"
)

func panickingHandler() fasthttp.RequestHandler {
	return func(ctx *fasthttp.RequestCtx) {
		panic("test panic")
	}
}

func TestContainment_RecoversPanic(t *testing.T) {
	c := &Containment{}

	var ctx fasthttp.RequestCtx
	ctx.Request.Header.SetMethod("POST")
	ctx.Request.SetRequestURI("/fraud-score")

	c.Wrap(panickingHandler())(&ctx)

	if ctx.Response.StatusCode() != fasthttp.StatusOK {
		t.Errorf("expected 200 on containment, got %d", ctx.Response.StatusCode())
	}

	expectedBody := `{"approved":true,"fraud_score":0.0}`
	if string(ctx.Response.Body()) != expectedBody {
		t.Errorf("expected body %q, got %q", expectedBody, string(ctx.Response.Body()))
	}

	if c.Recoveries.Load() != 1 {
		t.Errorf("expected Recoveries=1, got %d", c.Recoveries.Load())
	}
}

func TestContainment_CounterIncrements(t *testing.T) {
	c := &Containment{}

	handler := c.Wrap(panickingHandler())

	var ctx1 fasthttp.RequestCtx
	ctx1.Request.Header.SetMethod("POST")
	ctx1.Request.SetRequestURI("/fraud-score")
	handler(&ctx1)

	var ctx2 fasthttp.RequestCtx
	ctx2.Request.Header.SetMethod("POST")
	ctx2.Request.SetRequestURI("/fraud-score")
	handler(&ctx2)

	if c.Recoveries.Load() != 2 {
		t.Errorf("expected Recoveries=2 after two panics, got %d", c.Recoveries.Load())
	}
}

func TestContainment_NoPanic_PassesThrough(t *testing.T) {
	c := &Containment{}

	normalHandler := func(ctx *fasthttp.RequestCtx) {
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.SetBodyString("ok")
	}

	var ctx fasthttp.RequestCtx
	ctx.Request.Header.SetMethod("GET")
	ctx.Request.SetRequestURI("/")

	c.Wrap(normalHandler)(&ctx)

	if ctx.Response.StatusCode() != fasthttp.StatusOK {
		t.Errorf("expected 200, got %d", ctx.Response.StatusCode())
	}
	if c.Recoveries.Load() != 0 {
		t.Errorf("expected Recoveries=0, got %d", c.Recoveries.Load())
	}
}
