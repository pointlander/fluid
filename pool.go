// Copyright 2026 The Fluid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package main

import "sync"

// Pool runs jobs on a fixed set of goroutines.
//
// Each worker has its own inbox, so a fast worker cannot take another
// worker's job. Run must not be called reentrantly from inside a job.
type Pool struct {
	n     int
	inbox []chan func(worker int)
	once  sync.Once
}

// NewPool starts n worker goroutines. n must be at least 1.
func NewPool(n int) *Pool {
	if n < 1 {
		n = 1
	}
	p := &Pool{
		n:     n,
		inbox: make([]chan func(worker int), n),
	}
	for i := 0; i < n; i++ {
		p.inbox[i] = make(chan func(worker int))
		go worker(p.inbox[i], i)
	}
	return p
}

func worker(inbox chan func(worker int), id int) {
	for fn := range inbox {
		fn(id)
	}
}

// Run calls fn once on each worker and waits until every call returns.
// fn receives the worker index in [0, n).
func (p *Pool) Run(fn func(worker int)) {
	var wg sync.WaitGroup
	wg.Add(p.n)
	call := func(id int) {
		defer wg.Done()
		fn(id)
	}
	for i := 0; i < p.n; i++ {
		p.inbox[i] <- call
	}
	wg.Wait()
}

// Close stops the workers. It is safe to call more than once.
// Run must not be in progress.
func (p *Pool) Close() {
	p.once.Do(func() {
		for _, ch := range p.inbox {
			close(ch)
		}
	})
}
