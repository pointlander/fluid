// Copyright 2026 The Fluid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package main

import (
	"math"
	"math/cmplx"
	"testing"
)

func TestCNOTBasis(t *testing.T) {
	g, err := NewCNOTDrive(2, 0.05)
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

	// CNOT |c t> = |c, t XOR c>
	want := [][2]bool{
		{false, false},
		{false, true},
		{true, true},
		{true, false},
	}
	var cJet, tJet [4]float32
	for i, w := range want {
		cOut, tOut, cj, tj, err := g.Eval(i&2 != 0, i&1 != 0)
		if err != nil {
			t.Fatal(err)
		}
		cJet[i], tJet[i] = cj, tj
		if cOut != w[0] || tOut != w[1] {
			t.Fatalf("|%d%d> → control %v (%.2f mm/s) target %v (%.2f mm/s), want %v %v\n%s",
				i>>1, i&1, cOut, cj*1000, tOut, tj*1000, w[0], w[1], g.Sketch(42, 16))
		}
	}
	// Single-input jets stay clear of the cancelled and idle cases.
	if tJet[1] < 0 || tJet[2] < 0 || cJet[2] < 0 {
		t.Fatalf("logic-1 jets should be positive: control %v target %v mm/s", cJet, tJet)
	}
	if math.Abs(float64(cJet[0])) > 0.35*float64(cJet[2]) || math.Abs(float64(cJet[1])) > 0.35*float64(cJet[2]) {
		t.Fatalf("control jet leaked: %v mm/s", cJet)
	}
	if math.Abs(float64(tJet[0])) > 0.35*float64(tJet[1]) || math.Abs(float64(tJet[3])) > 0.35*float64(tJet[1]) {
		t.Fatalf("target jet leaked: %v mm/s", tJet)
	}
	t.Logf("control mm/s  %.2f %.2f %.2f %.2f  target mm/s  %.2f %.2f %.2f %.2f",
		cJet[0]*1000, cJet[1]*1000, cJet[2]*1000, cJet[3]*1000,
		tJet[0]*1000, tJet[1]*1000, tJet[2]*1000, tJet[3]*1000)
}

func TestCNOTBell(t *testing.T) {
	g, err := NewCNOTDrive(2, 0.05)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Close)

	inv := 1 / math.Sqrt2
	in := State{complex(inv, 0), 0, complex(inv, 0), 0}
	out, err := g.Apply(in)
	if err != nil {
		t.Fatal(err)
	}
	want := State{complex(inv, 0), 0, 0, complex(inv, 0)}
	for i := range want {
		if cmplx.Abs(out[i]-want[i]) > 1e-9 {
			t.Fatalf("CNOT|+0> = %s, want %s", formatState(out), formatState(want))
		}
	}

	// A phase on |10> is carried onto |11>.
	phased := State{0, 0, 1i, 0}
	out, err = g.Apply(phased)
	if err != nil {
		t.Fatal(err)
	}
	if cmplx.Abs(out[3]-1i) > 1e-9 || cmplx.Abs(out[0])+cmplx.Abs(out[1])+cmplx.Abs(out[2]) > 1e-9 {
		t.Fatalf("CNOT i|10> = %s", formatState(out))
	}

	var n2 float64
	for _, a := range out {
		n2 += cmplx.Abs(a) * cmplx.Abs(a)
	}
	if math.Abs(n2-1) > 1e-9 {
		t.Fatalf("norm %g", n2)
	}
}
