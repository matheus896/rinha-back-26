package artifact

import "os"

//go:generate go run ../../cmd/generator/ -in ../../resources/references.json.gz -out artifact.bin

// LoadFromFile reads artifact bytes from disk and loads the artifact.
// Only for tests and tools. Production code should use LoadMmap.
func LoadFromFile(path string) (*LoadedArtifact, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Load(data)
}
