package bruteforce

import (
	"encoding/binary"

	"rinha-backend-2026/internal/artifact"
)

func SearchFlat(q *[14]int16, vectors []byte, count uint32, tk TopK) {
	const recSize = uint32(artifact.VectorRecordSz)
	maxDist := tk.MaxDist()
	for i := uint32(0); i < count; i++ {
		base := i * recSize
		var vec [14]int16
		for d := 0; d < 14; d++ {
			vec[d] = int16(binary.LittleEndian.Uint16(vectors[base+uint32(d*2):]))
		}
		label := vectors[base+28]
		dist := SquaredDistanceEarlyExit(q, &vec, maxDist)
		if dist < maxDist {
			tk.Insert(dist, label)
			maxDist = tk.MaxDist()
		}
	}
}
