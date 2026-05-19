package vectorizer

import (
	"testing"

	"rinha-backend-2026/internal/artifact"
)

func TestQuantizeInt16(t *testing.T) {
	tests := []struct {
		name     string
		input    float64
		expected int16
	}{
		{"zero", 0.0, 0},
		{"half", 0.5, 16384},
		{"one", 1.0, 32767},
		{"micro_value", 0.0041, 134},
		{"over_one", 1.5, 32767},
		{"sentinel_neg", -1.0, artifact.SentinelInt16},
		{"any_negative", -0.001, artifact.SentinelInt16},
		{"round_up", 0.75, 24575},
		{"mid_range", 0.25, 8192},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := QuantizeInt16(tt.input)
			if got != tt.expected {
				t.Errorf("QuantizeInt16(%v) = %d, want %d", tt.input, got, tt.expected)
			}
		})
	}
}

func TestBoolToInt8(t *testing.T) {
	tests := []struct {
		name     string
		input    bool
		expected int8
	}{
		{"true", true, 1},
		{"false", false, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BoolToInt8(tt.input)
			if got != tt.expected {
				t.Errorf("BoolToInt8(%v) = %d, want %d", tt.input, got, tt.expected)
			}
		})
	}
}
