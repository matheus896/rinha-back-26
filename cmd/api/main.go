package main

import (
	"log/slog"
	"net"
	"os"
	"runtime"
	"strconv"
	"time"

	"github.com/valyala/fasthttp"

	"rinha-backend-2026/internal/artifact"
	"rinha-backend-2026/internal/ivf"
	"rinha-backend-2026/internal/mcc"
	"rinha-backend-2026/internal/norm"
	"rinha-backend-2026/internal/search"
	"rinha-backend-2026/internal/server"
)

func main() {
	runtime.GOMAXPROCS(1)

	logger := slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	artifactPath := os.Getenv("ARTIFACT_PATH")
	if artifactPath == "" {
		slog.Error("ARTIFACT_PATH not set")
		os.Exit(1)
	}
	art, err := artifact.LoadMmap(artifactPath)
	if err != nil {
		slog.Error("mmap load failed", "error", err, "path", artifactPath)
		os.Exit(1)
	}
	slog.Info("artifact loaded",
		"magic", string(art.Header.Magic[:]),
		"num_vectors", art.NumVectors(),
		"num_clusters", art.NumClusters(),
	)

	mccRisks, err := mcc.Load()
	if err != nil {
		slog.Error("mcc load failed", "error", err)
		os.Exit(1)
	}
	slog.Info("mcc risks loaded")

	normCfg, err := norm.Load()
	if err != nil {
		slog.Error("norm load failed", "error", err)
		os.Exit(1)
	}
	slog.Info("normalization loaded")

	cfg := ivf.Config{
		K:          envInt("IVF_K", 1024),
		NProbe:     envInt("IVF_NPROBE", 16),
		RetryExtra: envInt("IVF_RETRY_EXTRA", 8),
	}

	engine := search.NewEngine(art, cfg)

	engine.Warmup(500)

	readiness := server.NewReadiness()
	containment := &server.Containment{}

	fraudHandler := &server.FraudScoreHandler{
		Engine:    engine,
		NormCfg:   normCfg,
		MCCRisks:  mccRisks,
		Readiness: readiness,
	}

	readiness.SetReady()

	handler := func(ctx *fasthttp.RequestCtx) {
		defer func() {
			if rec := recover(); rec != nil {
				containment.Recoveries.Add(1)
				slog.Error("containment: panic recovered", "panic", rec)
				ctx.SetContentType("application/json")
				ctx.SetStatusCode(fasthttp.StatusOK)
				ctx.SetBodyString(`{"approved":true,"fraud_score":0.0}`)
			}
		}()

		switch string(ctx.Path()) {
		case "/ready":
			ctx.SetContentType("application/json")
			if readiness.IsReady() {
				ctx.SetStatusCode(fasthttp.StatusOK)
				ctx.SetBodyString(`{"status":"ready"}`)
			} else {
				ctx.SetStatusCode(fasthttp.StatusServiceUnavailable)
				ctx.SetBodyString(`{"status":"not ready"}`)
			}
		case "/fraud-score":
			if ctx.IsPost() {
				fraudHandler.HandleFastHTTP(ctx)
			} else {
				ctx.SetStatusCode(fasthttp.StatusMethodNotAllowed)
			}
		default:
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	}

	socketPath := os.Getenv("SOCKET_PATH")
	if socketPath == "" {
		slog.Error("SOCKET_PATH not set")
		os.Exit(1)
	}

	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		slog.Error("unix listen failed", "error", err, "path", socketPath)
		os.Exit(1)
	}
	defer listener.Close()

	if err := os.Chmod(socketPath, 0666); err != nil {
		slog.Error("chmod socket failed", "error", err)
		os.Exit(1)
	}

	server := &fasthttp.Server{
		Handler:                       handler,
		Name:                          "rinha-backend",
		ReadTimeout:                   750 * time.Millisecond,
		WriteTimeout:                  750 * time.Millisecond,
		IdleTimeout:                   10 * time.Second,
		ReadBufferSize:                1024,
		WriteBufferSize:               1024,
		MaxRequestBodySize:            4 * 1024,
		DisableHeaderNamesNormalizing: true,
		NoDefaultDate:                 true,
		NoDefaultServerHeader:         true,
		NoDefaultContentType:          true,
		DisablePreParseMultipartForm:  true,
		Concurrency:                   4096,
		DisableKeepalive:              false,
		TCPKeepalive:                  true,
		LogAllErrors:                  false,
	}

	slog.Info("starting server", "socket", socketPath)
	if err := server.Serve(listener); err != nil {
		slog.Error("server failed", "error", err)
		os.Exit(1)
	}
}

func envInt(name string, defaultVal int) int {
	v := os.Getenv(name)
	if v == "" {
		return defaultVal
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return defaultVal
	}
	return n
}
