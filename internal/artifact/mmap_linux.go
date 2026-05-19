package artifact

import (
	"fmt"
	"os"

	"golang.org/x/sys/unix"
)

func LoadMmap(path string) (*LoadedArtifact, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}
	size := int(stat.Size())

	data, err := unix.Mmap(int(f.Fd()), 0, size, unix.PROT_READ, unix.MAP_PRIVATE)
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}

	_ = unix.Madvise(data, unix.MADV_RANDOM)

	art, err := Load(data)
	if err != nil {
		_ = unix.Munmap(data)
		return nil, err
	}
	art.raw = data
	return art, nil
}
