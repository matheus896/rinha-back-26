package ivf

const kernelSafety = 65536

type Top5 struct {
	dist  [5]int64
	label [5]uint8
}

func (t *Top5) Reset() {
	for i := 0; i < 5; i++ {
		t.dist[i] = 1 << 62
		t.label[i] = 0
	}
}

func (t *Top5) WorstI64() int64   { return t.dist[4] }
func (t *Top5) MaxDist() int64    { return t.dist[4] }

func (t *Top5) WorstF32() float32 {
	return float32(t.dist[4]) + kernelSafety
}

func (t *Top5) InsertI64(d int64, l uint8) {
	if d >= t.dist[4] {
		return
	}
	pos := 4
	for pos > 0 && t.dist[pos-1] > d {
		t.dist[pos] = t.dist[pos-1]
		t.label[pos] = t.label[pos-1]
		pos--
	}
	t.dist[pos] = d
	t.label[pos] = l
}

func (t *Top5) Insert(dist int64, label uint8) {
	t.InsertI64(dist, label)
}

func (t *Top5) FraudCount() int {
	return int(t.label[0] + t.label[1] + t.label[2] + t.label[3] + t.label[4])
}
