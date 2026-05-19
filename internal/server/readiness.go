package server

import "sync/atomic"

type Readiness struct {
	ready atomic.Bool
}

func NewReadiness() *Readiness {
	return &Readiness{}
}

func (r *Readiness) SetReady() {
	r.ready.Store(true)
}

func (r *Readiness) IsReady() bool {
	return r.ready.Load()
}
