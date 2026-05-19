package search

func SquaredDistance(q, r *[14]int16) int64 {
	var sum int64
	for i := 0; i < 14; i++ {
		diff := int64(q[i]) - int64(r[i])
		sum += diff * diff
	}
	return sum
}
