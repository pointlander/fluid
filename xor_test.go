// Copyright 2026 The Fluid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package main

import (
	"math"
	"testing"
)

func TestAntiphaseBeamsCancel(t *testing.T) {
	m := acousticMedium{WaveNumber: float32(2 * math.Pi * xorFreq / xorC), Attenuation: xorAlpha}
	base := transducer{X: 0, Y: 0, DirX: 1, DirY: 0, Sigma: xorSigma, Amplitude: xorAmp}
	a := base
	b := base
	b.Y = xorOffset
	a.Y = -xorOffset
	b.Phase = math.Pi

	// On the midline, equal path lengths and opposite phase cancel.
	_, _ = fieldAccel([]transducer{a}, m, 0.008, 0)
	one, _ := fieldAccel([]transducer{a}, m, 0.008, 0)
	both, _ := fieldAccel([]transducer{a, b}, m, 0.008, 0)
	if one <= 0 {
		t.Fatalf("single piston acceleration %g, want a jet in +x", one)
	}
	if math.Abs(float64(both)) > 0.05*float64(one) {
		t.Fatalf("midline both-on acceleration %g, single %g", both, one)
	}

	// A point in the overlap still carries little force compared with one piston.
	oneSide, _ := fieldAccel([]transducer{a}, m, 0.008, xorOffset)
	bothSide, _ := fieldAccel([]transducer{a, b}, m, 0.008, xorOffset)
	if math.Abs(float64(bothSide)) > 0.25*float64(oneSide) {
		t.Fatalf("overlap both-on %g, single %g", bothSide, oneSide)
	}
}

func TestXORTruthTable(t *testing.T) {
	g, err := NewXORDrive(2, 0.05)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Close)

	for j := 1; j < g.sim.ny-1; j++ {
		for i := 1; i < g.sim.nx-1; i++ {
			if g.sim.solid[j*g.sim.nx+i] == 1 {
				t.Fatalf("interior solid at %d,%d", i, j)
			}
		}
	}
	if g.sim.fluidCount != (g.sim.nx-2)*(g.sim.ny-2) {
		t.Fatalf("fluid cells %d, want one sealed body", g.sim.fluidCount)
	}

	type row struct {
		a, b bool
		want bool
	}
	rows := []row{{false, false, false}, {true, false, true}, {false, true, true}, {true, true, false}}
	var speeds [4]float32
	for i, row := range rows {
		got, speed, err := g.Eval(row.a, row.b)
		if err != nil {
			t.Fatal(err)
		}
		speeds[i] = speed
		if got != row.want {
			t.Fatalf("A=%v B=%v jet=%g mm/s threshold=%g mm/s got %v want %v\n%s",
				row.a, row.b, speed*1000, g.Threshold()*1000, got, row.want, g.Sketch(42, 16))
		}
	}
	// The two single-piston jets agree, and rest is quiet next to them.
	if speeds[1] <= 0 || speeds[2] <= 0 {
		t.Fatalf("jets should leave the pistons: %v mm/s", speeds)
	}
	hi := speeds[1]
	if speeds[2] < hi {
		hi = speeds[2]
	}
	lo := speeds[1]
	if speeds[2] > lo {
		lo = speeds[2]
	}
	if lo < 0.5*hi {
		t.Fatalf("single-piston jets differ: %g and %g mm/s", speeds[1]*1000, speeds[2]*1000)
	}
	if math.Abs(float64(speeds[0])) > 0.15*float64(lo) || math.Abs(float64(speeds[3])) > 0.35*float64(lo) {
		t.Fatalf("cancelled drives are not quiet: off %g both %g single %g mm/s\n%s",
			speeds[0]*1000, speeds[3]*1000, lo*1000, g.Sketch(42, 16))
	}
	t.Logf("jet mm/s  00=%g  10=%g  01=%g  11=%g  threshold=%g",
		speeds[0]*1000, speeds[1]*1000, speeds[2]*1000, speeds[3]*1000, g.Threshold()*1000)
}
