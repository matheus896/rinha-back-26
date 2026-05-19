package vectorizer

import (
	"math"

	"rinha-backend-2026/internal/artifact"
	"rinha-backend-2026/internal/norm"
)

func QuantizeInt16(v float64) int16 {
	if v < 0 {
		return artifact.SentinelInt16
	}
	clamped := norm.Clamp(v, 0.0, 1.0)
	return int16(math.Round(clamped * 32767))
}

func BoolToInt8(b bool) int8 {
	if b {
		return 1
	}
	return 0
}
