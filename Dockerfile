FROM golang:1.24-alpine AS builder

WORKDIR /build

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 GOAMD64=v3 \
    go build -trimpath -ldflags="-s -w" -o /api ./cmd/api

FROM alpine:3.19

RUN apk add --no-cache curl && adduser -D app

COPY --from=builder /api /api

COPY internal/artifact/artifact.bin /artifact.bin

COPY entrypoint.sh /entrypoint.sh
COPY warmup.sh /warmup.sh
RUN chmod +x /entrypoint.sh /warmup.sh

USER app

ENTRYPOINT ["/entrypoint.sh"]
