package bruteforce

func SquaredDistanceEarlyExit(q, r *[14]int16, maxDist int64) int64 {
	var sum int64
	dims := [14]int{5, 6, 2, 0, 7, 8, 11, 12, 9, 10, 1, 13, 3, 4}
	for _, d := range dims {
		diff := int64(q[d]) - int64(r[d])
		sum += diff * diff
		if sum >= maxDist {
			return sum
		}
	}
	return sum
}
