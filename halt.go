// Copyright 2026 The Fluid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package main

import (
	"fmt"
	"math"
)

// Easy program:
//
//	x := 0
//	x = x + 1
//	x = x + 1
//	x = x + 1
//	halt
//
// The counter is a marker in one sealed body of water. A curl-bearing
// streaming jet carries it downstream. Each station it crosses is one
// increment. Entering the last station is the halt witness. This decides
// one short, halting run. It is not a decider for arbitrary programs.

const (
	haltNX    = 48
	haltNY    = 32
	haltWidth = 0.016
	haltAccel = 8.0   // m/s², centerline streaming acceleration
	haltSigma = 0.002 // m, jet width
	haltLimit = 0.20  // s, long enough for three increments
	haltSteps = 3
)

// HaltResult is the witness for the easy program.
type HaltResult struct {
	Halted bool
	Time   float32
	X      float32
	Counts int
}

// RunEasyHalt integrates the jet until the marker has crossed every
// increment station, or until the time limit.
func RunEasyHalt(workers int) (HaltResult, error) {
	cfg := DefaultConfig(haltNX, haltNY)
	cfg.Workers = workers
	cfg.Width = haltWidth
	cfg.Gravity = 0
	cfg.Nu = xorNu
	cfg.DTMax = 1.0 / 400.0
	cfg.PCGTol = 1e-3
	cfg.PCGMax = 200
	cfg.ParticlesPerEdge = 1
	sim, err := New(cfg)
	if err != nil {
		return HaltResult{}, err
	}
	defer sim.Close()
	sim.FillBody()

	width := sim.Width()
	mid := sim.Height() * 0.5
	start := 0.18 * width
	// Station i is the line the marker must cross to complete increment i+1.
	stations := []float32{0.36 * width, 0.54 * width, 0.72 * width}
	sim.particles = append(sim.particles, particle{x: start, y: mid})

	au := make([]float32, len(sim.u))
	av := make([]float32, len(sim.v))
	if err := sim.SetStreaming(au, av); err != nil {
		return HaltResult{}, err
	}
	fillJet(sim, au, mid, haltSigma, haltAccel)

	var res HaltResult
	res.X = start
	const chunk = 0.005
	for res.Time < haltLimit && res.Counts < haltSteps {
		stats, err := sim.Advance(chunk, nil)
		if err != nil {
			return res, err
		}
		res.Time = stats.Time
		if len(sim.particles) != 1 {
			return res, fmt.Errorf("halt marker count %d", len(sim.particles))
		}
		res.X = sim.particles[0].x
		res.Counts = crossed(res.X, stations)
	}
	res.Halted = res.Counts >= haltSteps
	return res, nil
}

// fillJet writes a horizontal streaming acceleration that depends only on
// the distance from the midline, so its curl is nonzero and pressure cannot
// cancel it.
func fillJet(s *Sim, au []float32, mid, sigma, accel float32) {
	nx := s.nx
	dx := s.dx
	inv := 1 / (2 * sigma * sigma)
	for j := 0; j < s.ny; j++ {
		y := (float32(j) + 0.5) * dx
		dy := y - mid
		a := accel * float32(math.Exp(float64(-dy*dy*inv)))
		row := j * (nx + 1)
		for i := 0; i <= nx; i++ {
			au[row+i] = a
		}
	}
}

func crossed(x float32, stations []float32) int {
	n := 0
	for _, s := range stations {
		if x >= s {
			n++
		}
	}
	return n
}

func runEasyHalt(workers int) error {
	res, err := RunEasyHalt(workers)
	if err != nil {
		return err
	}
	fmt.Printf("easy program: x := 0; x++; x++; x++; halt\n")
	fmt.Printf("marker x=%.2f mm after %.3f s, increments %d/%d\n",
		res.X*1000, res.Time, res.Counts, haltSteps)
	if res.Halted {
		fmt.Printf("halts\n")
		return nil
	}
	return fmt.Errorf("no halt witness within %.2f s (x=%.2f mm, increments %d)",
		haltLimit, res.X*1000, res.Counts)
}
