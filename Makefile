.PHONY: build test bench docker-build docker-up generate tidy vet pgo-profile pgo-build

generate:
	go generate ./...

build: generate
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o bin/api ./cmd/api

pgo-profile:
	go test -run=^$$ -bench=BenchmarkSearch -cpuprofile=cmd/api/default.pgo ./internal/search

pgo-build: generate
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o bin/api ./cmd/api
	@echo "PGO build complete (uses default.pgo if present)"

test:
	go test ./...

bench:
	go test -bench=. -benchmem -count=10 ./internal/search/... ./internal/vectorizer/... | tee bench.txt
	@echo ""
	@echo "Run 'benchstat bench.txt' for statistical comparison."

vet:
	go vet ./...

tidy:
	go mod tidy

docker-build:
	docker compose build

docker-up:
	docker compose up

docker-down:
	docker compose down
