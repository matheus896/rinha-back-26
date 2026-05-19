package search

import (
	"math"

	"rinha-backend-2026/internal/artifact"
)

type neighbor struct {
	Dist  int64
	Label uint8
}

type TopK struct {
	items [5]neighbor
	count int
}

func (tk *TopK) Insert(dist int64, label uint8) {
	pos := 0
	for pos < tk.count && pos < 5 && tk.items[pos].Dist < dist {
		pos++
	}
	if pos == 5 {
		return
	}
	if tk.count == 5 {
		for i := 4; i > pos; i-- {
			tk.items[i] = tk.items[i-1]
		}
	} else {
		for i := tk.count; i > pos; i-- {
			tk.items[i] = tk.items[i-1]
		}
		tk.count++
	}
	tk.items[pos] = neighbor{Dist: dist, Label: label}
}

func (tk *TopK) MaxDist() int64 {
	if tk.count < 5 {
		return math.MaxInt64
	}
	return tk.items[tk.count-1].Dist
}

func (tk *TopK) FraudCount() int {
	n := 0
	for i := 0; i < tk.count; i++ {
		if tk.items[i].Label == artifact.LabelFraud {
			n++
		}
	}
	return n
}

func (tk *TopK) Sorted() []neighbor {
	out := make([]neighbor, tk.count)
	copy(out, tk.items[:tk.count])
	return out
}
