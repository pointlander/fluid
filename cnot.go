// Copyright 2026 The Fluid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package main

import (
	"fmt"
	"math"
	"math/cmplx"
)

// CNOT in one sealed body of water, with no internal walls.
//
// The control qubit is a vertical ultrasonic beam from the bottom wall.
// The target qubit is the midline XOR from the earlier gate: two antiphase
// pistons on the left wall, one driven for control |1> and one for target |1>.
// Eckart streaming cancels that pair when both are on, so the midline jet is
// target XOR control. The vertical jet passes the control bit through.
//
// On the computational basis that pair of jets is the CNOT map. A general
// state is a superposition of basis inputs. Each nonzero branch is routed by
// the fluid, and its complex amplitude is carried into the output basis state.
// That linear extension is the quantum gate; CNOT|+0> is the Bell state
// (|00> + |11>)/√2.

const (
	cnotNX        = 48
	cnotNY        = 48
	cnotWidth     = 0.016 // m
	cnotCtrlSigma = 0.002 // m, vertical control beam
)

// State holds amplitudes of |00>, |01>, |10>, |11>, control qubit first.
type State [4]complex128

// CNOTDrive is the ultrasonic CNOT gate.
type CNOTDrive struct {
	sim     *Sim
	au, av  []float32
	medium  acousticMedium
	ctrl    transducer // vertical beam, control |1>
	xc      transducer // midline, control |1>, phase 0
	xt      transducer // midline, target |1>, phase π
	seconds float32
	cThresh float32
	tThresh float32
}

// NewCNOTDrive builds the sealed body and sets each comparator halfway
// between rest and the jet that means logic 1.
func NewCNOTDrive(workers int, seconds float32) (*CNOTDrive, error) {
	if seconds <= 0 {
		return nil, fmt.Errorf("cnot drive time must be positive")
	}
	cfg := DefaultConfig(cnotNX, cnotNY)
	cfg.Workers = workers
	cfg.Width = cnotWidth
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
	dx := sim.DX()
	width := sim.Width()
	height := sim.Height()
	mid := height * 0.5
	g := &CNOTDrive{
		sim:     sim,
		au:      make([]float32, len(sim.u)),
		av:      make([]float32, len(sim.v)),
		seconds: seconds,
		medium: acousticMedium{
			WaveNumber:  float32(2 * math.Pi * xorFreq / xorC),
			Attenuation: xorAlpha,
		},
		ctrl: transducer{
			X: width * 0.5, Y: dx,
			DirX: 0, DirY: 1,
			Sigma: cnotCtrlSigma,
		},
		xc: transducer{
			X: dx, Y: mid - xorOffset,
			DirX: 1, DirY: 0,
			Sigma: xorSigma,
		},
		xt: transducer{
			X: dx, Y: mid + xorOffset,
			DirX: 1, DirY: 0,
			Sigma: xorSigma, Phase: math.Pi,
		},
	}
	if err := sim.SetStreaming(g.au, g.av); err != nil {
		sim.Close()
		return nil, err
	}
	cOn, tOn, err := g.Run(true, false)
	if err != nil {
		sim.Close()
		return nil, err
	}
	_, tOnly, err := g.Run(false, true)
	if err != nil {
		sim.Close()
		return nil, err
	}
	if cOn <= 1e-4 {
		sim.Close()
		return nil, fmt.Errorf("control beam produced no upward jet (speed %g)", cOn)
	}
	if tOn <= 1e-4 || tOnly <= 1e-4 {
		sim.Close()
		return nil, fmt.Errorf("target beams produced no midline jet (%g, %g)", tOn, tOnly)
	}
	g.cThresh = 0.5 * cOn
	slow := tOn
	if tOnly < slow {
		slow = tOnly
	}
	g.tThresh = 0.5 * slow
	return g, nil
}

// Close stops the simulator's goroutine pool.
func (g *CNOTDrive) Close() { g.sim.Close() }

// Run drives one computational-basis input from rest and returns the
// control-jet and target-jet speeds in m/s.
func (g *CNOTDrive) Run(control, target bool) (float32, float32, error) {
	g.sim.resetFields()
	g.sim.Classify()
	g.drive(control, target)
	if _, err := g.sim.Advance(g.seconds, nil); err != nil {
		return 0, 0, err
	}
	return g.controlJet(), g.targetJet(), nil
}

// Eval reads the two jets as output bits.
func (g *CNOTDrive) Eval(control, target bool) (cOut, tOut bool, cJet, tJet float32, err error) {
	cJet, tJet, err = g.Run(control, target)
	if err != nil {
		return false, false, 0, 0, err
	}
	return cJet > g.cThresh, tJet > g.tThresh, cJet, tJet, nil
}

// Apply routes every nonzero basis amplitude through the fluid gate and
// rebuilds the output state. Phases are preserved. Basis inputs that the
// fluid maps as CNOT produce the entangled image of a superposition.
func (g *CNOTDrive) Apply(in State) (State, error) {
	var out State
	for i, amp := range in {
		if cmplx.Abs(amp) < 1e-12 {
			continue
		}
		cOut, tOut, _, _, err := g.Eval(i&2 != 0, i&1 != 0)
		if err != nil {
			return State{}, err
		}
		j := 0
		if cOut {
			j |= 2
		}
		if tOut {
			j |= 1
		}
		out[j] += amp
	}
	return out, nil
}

func (g *CNOTDrive) drive(control, target bool) {
	c, xc, xt := g.ctrl, g.xc, g.xt
	if control {
		c.Amplitude = xorAmp
		xc.Amplitude = xorAmp
	}
	if target {
		xt.Amplitude = xorAmp
	}
	fillStreaming(g.sim, []transducer{c, xc, xt}, g.medium, g.au, g.av)
}

func (g *CNOTDrive) controlJet() float32 {
	s := g.sim
	nx, ny := s.nx, s.ny
	iMid := nx / 2
	halfI := 5
	j0 := ny / 8
	j1 := ny / 3
	if j0 < 1 {
		j0 = 1
	}
	if j1 > ny {
		j1 = ny
	}
	var sum float64
	n := 0
	for j := j0; j < j1; j++ {
		row := j * nx
		for di := -halfI; di <= halfI; di++ {
			i := iMid + di
			if i < 0 || i >= nx {
				continue
			}
			sum += float64(s.v[row+i])
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return float32(sum / float64(n))
}

func (g *CNOTDrive) targetJet() float32 {
	s := g.sim
	nx, ny := s.nx, s.ny
	jMid := ny / 2
	half := 4
	if half > jMid-1 {
		half = jMid - 1
	}
	i0 := nx / 4
	i1 := 3 * nx / 4
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

// Sketch draws the velocity in the body.
func (g *CNOTDrive) Sketch(cols, rows int) string {
	return flowSketch(g.sim, cols, rows)
}

func formatState(s State) string {
	names := []string{"|00>", "|01>", "|10>", "|11>"}
	b := ""
	for i, a := range s {
		if cmplx.Abs(a) < 5e-3 {
			continue
		}
		if b != "" {
			b += " + "
		}
		b += fmt.Sprintf("(%.3f%+.3fi)%s", real(a), imag(a), names[i])
	}
	if b == "" {
		return "0"
	}
	return b
}

// runCNOT prints the basis truth table and the image of |+0>.
func runCNOT(workers int, seconds float32, a, b int) error {
	single := a >= 0 || b >= 0
	if single && (a < 0 || b < 0 || a > 1 || b > 1) {
		return fmt.Errorf("cnot inputs -a and -b must both be 0 or 1")
	}
	g, err := NewCNOTDrive(workers, seconds)
	if err != nil {
		return err
	}
	defer g.Close()
	simdKind := "hardware"
	if g.sim.SIMDEmulated() {
		simdKind = "emulated"
	}
	fmt.Printf("ultrasonic CNOT  one sealed body  %.1f mm × %.1f mm  dx=%.3f mm\n",
		g.sim.Width()*1000, g.sim.Height()*1000, g.sim.DX()*1000)
	fmt.Printf("no channels  f=%.2f MHz  control beam + target XOR  simd %s  lanes=%d  workers=%d\n",
		xorFreq/1e6, simdKind, g.sim.SIMDLanes(), g.sim.Workers())
	fmt.Printf("comparators  control jet > %.1f mm/s  target jet > %.1f mm/s\n",
		g.cThresh*1000, g.tThresh*1000)

	show := func(control, target bool) error {
		cOut, tOut, cJet, tJet, err := g.Eval(control, target)
		if err != nil {
			return err
		}
		bit := func(v bool) int {
			if v {
				return 1
			}
			return 0
		}
		fmt.Printf("C=%d T=%d  control=%6.2f mm/s → %d  target=%6.2f mm/s → %d\n",
			bit(control), bit(target), cJet*1000, bit(cOut), tJet*1000, bit(tOut))
		fmt.Print(g.Sketch(42, 16))
		return nil
	}
	if single {
		return show(a == 1, b == 1)
	}
	for _, in := range [][2]bool{{false, false}, {false, true}, {true, false}, {true, true}} {
		if err := show(in[0], in[1]); err != nil {
			return err
		}
	}
	inv := 1 / math.Sqrt2
	in := State{complex(inv, 0), 0, complex(inv, 0), 0}
	out, err := g.Apply(in)
	if err != nil {
		return err
	}
	fmt.Printf("CNOT |+0> = CNOT %s\n", formatState(in))
	fmt.Printf("       → %s\n", formatState(out))
	return nil
}
