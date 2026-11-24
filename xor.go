// Copyright 2026 The Fluid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package main

import (
	"fmt"
	"math"
	"strings"
)

// One sealed body of water, no internal walls. Two ultrasonic pistons sit on
// the left wall and aim across the body. They are wired in antiphase, so when
// both are driven the acoustic particle velocity cancels in the overlap and
// the Eckart streaming force collapses. Exactly one driven piston leaves a jet
// along the midline. The logic output is that jet: fast means 1.

const (
	xorNX     = 48
	xorNY     = 32
	xorWidth  = 0.016  // m, the body is this wide
	xorNu     = 1e-6   // m²/s, water
	xorFreq   = 1.0e6  // Hz
	xorC      = 1480.0 // m/s, sound speed in water
	xorAlpha  = 40.0   // 1/m, Eckart coefficient of the beams
	xorAmp    = 0.20   // m/s, piston particle velocity when the input is on
	xorSigma  = 0.0025 // m, Gaussian beam width
	xorOffset = 0.0006 // m, pistons sit this far above and below the midline
)

// XORDrive is the ultrasonic XOR gate.
type XORDrive struct {
	sim       *Sim
	au, av    []float32
	medium    acousticMedium
	elemA     transducer
	elemB     transducer
	seconds   float32
	threshold float32 // centerline jet speed that separates 0 from 1
}

// NewXORDrive builds the sealed body and sets the output comparator halfway
// between rest and the jet from transducer A alone. seconds is how long each
// evaluation drives the pistons.
func NewXORDrive(workers int, seconds float32) (*XORDrive, error) {
	if seconds <= 0 {
		return nil, fmt.Errorf("xor drive time must be positive")
	}
	cfg := DefaultConfig(xorNX, xorNY)
	cfg.Workers = workers
	cfg.Width = xorWidth
	cfg.Gravity = 0
	cfg.Nu = xorNu
	cfg.DTMax = 1.0 / 400.0
	cfg.PCGTol = 1e-4
	cfg.PCGMax = 300
	cfg.ParticlesPerEdge = 1
	sim, err := New(cfg)
	if err != nil {
		return nil, err
	}
	sim.FillBody()
	height := sim.Height()
	mid := height * 0.5
	g := &XORDrive{
		sim:     sim,
		au:      make([]float32, len(sim.u)),
		av:      make([]float32, len(sim.v)),
		seconds: seconds,
		medium: acousticMedium{
			WaveNumber:  float32(2 * math.Pi * xorFreq / xorC),
			Attenuation: xorAlpha,
		},
		elemA: transducer{
			X: sim.dx, Y: mid - xorOffset,
			DirX: 1, DirY: 0,
			Sigma: xorSigma, Phase: 0,
		},
		elemB: transducer{
			X: sim.dx, Y: mid + xorOffset,
			DirX: 1, DirY: 0,
			Sigma: xorSigma, Phase: math.Pi,
		},
	}
	if err := sim.SetStreaming(g.au, g.av); err != nil {
		sim.Close()
		return nil, err
	}
	speed, err := g.Run(true, false)
	if err != nil {
		sim.Close()
		return nil, err
	}
	if speed <= 1e-4 {
		sim.Close()
		return nil, fmt.Errorf("transducer A produced no centerline jet (speed %g)", speed)
	}
	g.threshold = 0.5 * speed
	return g, nil
}

// Close stops the simulator's goroutine pool.
func (g *XORDrive) Close() { g.sim.Close() }

// Threshold is the centerline speed, in m/s, above which the output reads 1.
func (g *XORDrive) Threshold() float32 { return g.threshold }

// Run drives the two inputs from rest and returns the centerline jet speed.
// Positive speed points away from the pistons.
func (g *XORDrive) Run(a, b bool) (float32, error) {
	g.sim.resetFields()
	g.sim.Classify()
	g.drive(a, b)
	if _, err := g.sim.Advance(g.seconds, nil); err != nil {
		return 0, err
	}
	return g.jetSpeed(), nil
}

// Eval drives the inputs and reads the logic output.
func (g *XORDrive) Eval(a, b bool) (bool, float32, error) {
	speed, err := g.Run(a, b)
	if err != nil {
		return false, 0, err
	}
	return speed > g.threshold, speed, nil
}

func (g *XORDrive) drive(a, b bool) {
	ta, tb := g.elemA, g.elemB
	if a {
		ta.Amplitude = xorAmp
	}
	if b {
		tb.Amplitude = xorAmp
	}
	fillStreaming(g.sim, []transducer{ta, tb}, g.medium, g.au, g.av)
}

// jetSpeed is the mean horizontal velocity in the midline window.
func (g *XORDrive) jetSpeed() float32 {
	s := g.sim
	nx, ny := s.nx, s.ny
	jMid := ny / 2
	half := 4
	if half > jMid-1 {
		half = jMid - 1
	}
	i0 := nx / 4
	i1 := 3 * nx / 4
	if i0 < 1 {
		i0 = 1
	}
	if i1 > nx {
		i1 = nx
	}
	var sum float64
	n := 0
	ust := nx + 1
	for j := jMid - half; j <= jMid+half; j++ {
		row := j * ust
		for i := i0; i < i1; i++ {
			sum += float64(s.u[row+i])
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return float32(sum / float64(n))
}

// Sketch draws the velocity in the single body. '#' is the wall, '.' is slow
// fluid, and the arrows follow the local velocity.
func (g *XORDrive) Sketch(cols, rows int) string {
	s := g.sim
	if cols < 2 {
		cols = 2
	}
	if rows < 2 {
		rows = 2
	}
	type sample struct{ u, v float32 }
	pts := make([]sample, cols*rows)
	maxS := float32(0)
	for row := 0; row < rows; row++ {
		// Row 0 is the top of the body, matching the other ASCII views.
		y := s.Height() * (1 - (float32(row)+0.5)/float32(rows))
		for col := 0; col < cols; col++ {
			x := s.Width() * (float32(col) + 0.5) / float32(cols)
			u := s.sampleU(x, y)
			v := s.sampleV(x, y)
			pts[row*cols+col] = sample{u, v}
			if sp := u*u + v*v; sp > maxS {
				maxS = sp
			}
		}
	}
	maxS = float32(math.Sqrt(float64(maxS)))
	var b strings.Builder
	wall := strings.Repeat("#", cols+2)
	b.WriteString(wall)
	b.WriteByte('\n')
	cut := maxS * 0.25
	for row := 0; row < rows; row++ {
		b.WriteByte('#')
		for col := 0; col < cols; col++ {
			p := pts[row*cols+col]
			sp := float32(math.Hypot(float64(p.u), float64(p.v)))
			if maxS == 0 || sp < cut {
				b.WriteByte('.')
				continue
			}
			if math.Abs(float64(p.u)) >= math.Abs(float64(p.v)) {
				if p.u >= 0 {
					b.WriteByte('>')
				} else {
					b.WriteByte('<')
				}
			} else if p.v >= 0 {
				b.WriteByte('^')
			} else {
				b.WriteByte('v')
			}
		}
		b.WriteString("#\n")
	}
	b.WriteString(wall)
	b.WriteByte('\n')
	return b.String()
}

// runXOR prints the gate. If a and b are both 0 or 1, it evaluates that
// input. Otherwise it prints the truth table.
func runXOR(workers int, seconds float32, a, b int) error {
	single := a >= 0 || b >= 0
	if single && (a < 0 || b < 0 || a > 1 || b > 1) {
		return fmt.Errorf("xor inputs -a and -b must both be 0 or 1")
	}
	g, err := NewXORDrive(workers, seconds)
	if err != nil {
		return err
	}
	defer g.Close()
	simdKind := "hardware"
	if g.sim.SIMDEmulated() {
		simdKind = "emulated"
	}
	fmt.Printf("ultrasonic XOR  one sealed body  %.1f mm × %.1f mm  dx=%.3f mm\n",
		g.sim.Width()*1000, g.sim.Height()*1000, g.sim.DX()*1000)
	fmt.Printf("no channels  f=%.2f MHz  antiphase pistons  α=%.0f /m  simd %s  lanes=%d  workers=%d\n",
		xorFreq/1e6, xorAlpha, simdKind, g.sim.SIMDLanes(), g.sim.Workers())
	fmt.Printf("comparator  jet > %.1f mm/s\n", g.Threshold()*1000)

	eval := func(inA, inB bool) error {
		on, speed, err := g.Eval(inA, inB)
		if err != nil {
			return err
		}
		bit := 0
		if on {
			bit = 1
		}
		ia, ib := 0, 0
		if inA {
			ia = 1
		}
		if inB {
			ib = 1
		}
		fmt.Printf("A=%d B=%d  jet=%7.2f mm/s  XOR=%d\n", ia, ib, speed*1000, bit)
		fmt.Print(g.Sketch(42, 16))
		return nil
	}
	if single {
		return eval(a == 1, b == 1)
	}
	for _, in := range [][2]bool{{false, false}, {true, false}, {false, true}, {true, true}} {
		if err := eval(in[0], in[1]); err != nil {
			return err
		}
	}
	return nil
}
