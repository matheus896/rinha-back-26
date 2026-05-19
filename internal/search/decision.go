package search

const DecisionThreshold = 0.6

func Decide(fraudCount int) (approved bool, fraudScore float64) {
	fraudScore = float64(fraudCount) / 5.0
	approved = fraudScore < DecisionThreshold
	return
}
