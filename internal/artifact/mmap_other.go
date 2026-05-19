//go:build !linux

package artifact

import "os"

func LoadMmap(path string) (*LoadedArtifact, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	art, err := Load(data)
	if err != nil {
		return nil, err
	}
	art.raw = data
	return art, nil
}
