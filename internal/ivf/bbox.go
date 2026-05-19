package ivf

func AABBLowerBoundF32(q *[14]float32, bmin, bmax *[16]int16) float32 {
	var sum float32
	for d := 0; d < 14; d++ {
		qd := q[d]
		bnLo := float32(bmin[d])
		bnHi := float32(bmax[d])
		var t float32
		switch {
		case qd > bnHi:
			t = qd - bnHi
		case qd < bnLo:
			t = bnLo - qd
		default:
			t = 0
		}
		sum += t * t
	}
	return sum
}
