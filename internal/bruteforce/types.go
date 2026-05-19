package bruteforce

type TopK interface {
	Insert(dist int64, label uint8)
	MaxDist() int64
	FraudCount() int
}
