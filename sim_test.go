// Copyright 2026 The Fluid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package main

import (
	"math"
	"testing"
)

func testSim(t *testing.T, cfg Config) *Sim {
	t.Helper()
	sim, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(sim.Close)
	return sim
}

func TestPoolCoversRange(t *testing.T) {
	p := NewPool(3)
	defer p.Close()
	const n = 10
	got := make([]int, n)
	p.Run(func(w int) {
		a := n * w / 3
		b := n * (w + 1) / 3
		for i := a; i < b; i++ {
			got[i]++
		}
	})
	for i, c := range got {
		if c != 1 {
			t.Fatalf("index %d covered %d times", i, c)
		}
	}
}

func TestSampleLinear(t *testing.T) {
	cfg := DefaultConfig(16, 12)
	cfg.Workers = 2
	sim := testSim(t, cfg)
	for j := 0; j < sim.ny; j++ {
		for i := 0; i <= sim.nx; i++ {
			sim.u[j*(sim.nx+1)+i] = float32(i)
		}
	}
	for j := 0; j <= sim.ny; j++ {
		for i := 0; i < sim.nx; i++ {
			sim.v[j*sim.nx+i] = float32(j) * 0.5
		}
	}
	x := 2.3 * sim.dx
	y := 4.2 * sim.dx
	if got := sim.sampleU(x, y); math.Abs(float64(got-2.3)) > 1e-4 {
		t.Fatalf("sampleU = %g, want 2.3", got)
	}
	// v is stored at y = j*dx with value 0.5*j, so v(y) = 0.5*y/dx.
	if got := sim.sampleV(x, y); math.Abs(float64(got-0.5*4.2)) > 1e-4 {
		t.Fatalf("sampleV = %g, want %g", got, 0.5*4.2)
	}
}

func TestDivergenceOfLinearField(t *testing.T) {
	cfg := DefaultConfig(18, 14)
	cfg.Workers = 3
	sim := testSim(t, cfg)
	for j := 0; j < sim.ny; j++ {
		for i := 0; i <= sim.nx; i++ {
			// u(x) = x ⇒ ∂u/∂x = 1.
			sim.u[j*(sim.nx+1)+i] = float32(i) * sim.dx
		}
	}
	for j := 1; j < sim.ny-1; j++ {
		for i := 1; i < sim.nx-1; i++ {
			sim.fluid[j*sim.nx+i] = 1
		}
	}
	divergence(sim, sim.rhs, 1/sim.dx)
	for j := 1; j < sim.ny-1; j++ {
		for i := 1; i < sim.nx-1; i++ {
			got := sim.rhs[j*sim.nx+i]
			if math.Abs(float64(got-1)) > 1e-4 {
				t.Fatalf("div at %d,%d = %g, want 1", i, j, got)
			}
		}
	}
	if sim.rhs[0] != 0 {
		t.Fatalf("solid cell divergence = %g", sim.rhs[0])
	}
}

func TestSpMVMatchesScalar(t *testing.T) {
	cfg := DefaultConfig(22, 16)
	cfg.Workers = 4
	cfg.Nu = 0
	sim := testSim(t, cfg)
	sim.SeedPool()
	sim.buildCoefficients()
	sim.ensureDirichlet()
	for j := 1; j < sim.ny-1; j++ {
		for i := 1; i < sim.nx-1; i++ {
			id := j*sim.nx + i
			if sim.fluid[id] == 0 {
				continue
			}
			sim.p[id] = float32((i*3+j*5)%7) + 0.25
		}
	}
	spmv(sim, sim.p, sim.q)
	for j := 0; j < sim.ny; j++ {
		for i := 0; i < sim.nx; i++ {
			id := j*sim.nx + i
			var want float32
			if i > 0 && j > 0 && i < sim.nx-1 && j < sim.ny-1 {
				want = sim.apAt(i, j, sim.p)
			}
			got := sim.q[id]
			if math.Abs(float64(got-want)) > 1e-4 {
				t.Fatalf("Ap at %d,%d = %g, want %g", i, j, got, want)
			}
		}
	}
}

func TestGradientMatchesLaplacian(t *testing.T) {
	cfg := DefaultConfig(20, 16)
	cfg.Workers = 3
	cfg.Nu = 0
	cfg.Gravity = 0
	sim := testSim(t, cfg)
	sim.SeedPool()
	sim.buildCoefficients()
	sim.ensureDirichlet()
	for j := 1; j < sim.ny-1; j++ {
		for i := 1; i < sim.nx-1; i++ {
			id := j*sim.nx + i
			if sim.fluid[id] == 0 {
				sim.p[id] = 0
				continue
			}
			sim.p[id] = float32(i+2*j) * 0.5
		}
	}
	dt := float32(1.0 / 60.0)
	spmv(sim, sim.p, sim.q)
	applyGradient(sim, dt)
	divergence(sim, sim.rhs, 1/sim.dx)
	scale := dt / (sim.rho * sim.dx * sim.dx)
	worst := 0.0
	for j := 1; j < sim.ny-1; j++ {
		for i := 1; i < sim.nx-1; i++ {
			id := j*sim.nx + i
			if sim.fluid[id] == 0 {
				continue
			}
			want := scale * sim.q[id]
			got := sim.rhs[id]
			err := math.Abs(float64(got - want))
			if err > worst {
				worst = err
			}
			if err > 1e-2 {
				t.Fatalf("div at %d,%d = %g, want %g (Ap=%g)", i, j, got, want, sim.q[id])
			}
		}
	}
	t.Logf("max |div − (dt/ρh²) Ap| = %g", worst)
}

func TestHydrostaticPool(t *testing.T) {
	for _, workers := range []int{1, 5} {
		cfg := DefaultConfig(36, 28)
		cfg.Workers = workers
		cfg.Nu = 0
		cfg.PCGTol = 1e-6
		cfg.PCGMax = 400
		sim := testSim(t, cfg)
		sim.SeedPool()
		_, _, _, j1 := sim.poolRect()
		jf := j1 - 1
		dt := float32(1.0 / 60.0)
		st := sim.Step(dt)
		if sim.pinned {
			t.Fatalf("workers=%d: still pool should have a free surface", workers)
		}
		if st.PCGIters == 0 {
			t.Fatal("pressure solve did not run")
		}
		i := sim.nx / 2
		for j := 1; j <= jf; j++ {
			want := sim.rho * sim.g * sim.dx * float32(jf-j+1)
			got := sim.p[j*sim.nx+i]
			rel := math.Abs(float64((got - want) / want))
			if rel > 1e-2 {
				t.Fatalf("workers=%d: p[%d] = %g, want %g (rel %g), iters=%d resid=%g",
					workers, j, got, want, rel, st.PCGIters, st.Residual)
			}
		}
		if st.MaxSpeed > 0.02 {
			t.Fatalf("workers=%d: max speed %g after one step, want ~0 (g dt = %g)",
				workers, st.MaxSpeed, sim.g*dt)
		}
		if st.MaxDiv > 0.05 {
			t.Fatalf("workers=%d: max |div| %g", workers, st.MaxDiv)
		}
		// The body force was removed by pressure, so markers stay in their cells.
		if st.FluidCells == 0 {
			t.Fatal("pool vanished")
		}
		t.Logf("workers=%d iters=%d resid=%g max|u|=%g max|div|=%g",
			workers, st.PCGIters, st.Residual, st.MaxSpeed, st.MaxDiv)
	}
}

func TestWallVelocityStaysZero(t *testing.T) {
	cfg := DefaultConfig(24, 18)
	cfg.Workers = 2
	cfg.Nu = 0
	sim := testSim(t, cfg)
	sim.SeedDam()
	sim.Step(1.0 / 120)
	for j := 0; j < sim.ny; j++ {
		if sim.u[j*(sim.nx+1)] != 0 || sim.u[j*(sim.nx+1)+sim.nx] != 0 {
			t.Fatalf("horizontal wall velocity leaked at row %d", j)
		}
	}
	for i := 0; i < sim.nx; i++ {
		if sim.v[i] != 0 || sim.v[sim.ny*sim.nx+i] != 0 {
			t.Fatalf("vertical wall velocity leaked at column %d", i)
		}
	}
}

func TestViscositySpreads(t *testing.T) {
	cfg := DefaultConfig(16, 16)
	cfg.Workers = 2
	cfg.Nu = 0.05
	cfg.Gravity = 0
	sim := testSim(t, cfg)
	sim.SeedPool()
	sim.buildCoefficients()
	// A single liquid face, away from the walls.
	id := 4*(sim.nx+1) + 4
	if sim.uFluid[id] == 0 {
		t.Fatal("expected a liquid face")
	}
	sim.u[id] = 1
	sim.viscosity(1.0 / 200)
	if sim.u[id] >= 1 {
		t.Fatalf("center velocity %g did not fall", sim.u[id])
	}
	if sim.u[id-1] <= 0 || sim.u[id+1] <= 0 {
		t.Fatalf("neighbors did not rise: left %g right %g", sim.u[id-1], sim.u[id+1])
	}
}

func TestDamBreakFlowsRight(t *testing.T) {
	cfg := DefaultConfig(40, 28)
	cfg.Workers = 4
	cfg.Nu = 0
	cfg.ParticlesPerEdge = 2
	cfg.PCGTol = 1e-3
	cfg.PCGMax = 80
	sim := testSim(t, cfg)
	sim.SeedDam()
	n0 := len(sim.particles)
	x0, y0 := sim.CenterOfMass()
	st, err := sim.Advance(0.35, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(sim.particles) != n0 {
		t.Fatalf("particles %d, want %d", len(sim.particles), n0)
	}
	for i := range sim.particles {
		p := sim.particles[i]
		if p.x <= sim.dx || p.y <= sim.dx || p.x >= float32(sim.nx-1)*sim.dx || p.y >= float32(sim.ny-1)*sim.dx {
			t.Fatalf("particle %d left the tank at (%g, %g)", i, p.x, p.y)
		}
	}
	x1, y1 := sim.CenterOfMass()
	if x1 < x0+0.02*sim.Width() {
		t.Fatalf("center of mass x %g -> %g, want the column to fall to the right", x0, x1)
	}
	if y1 > y0 {
		t.Fatalf("center of mass rose: %g -> %g", y0, y1)
	}
	t.Logf("com (%.3f, %.3f) -> (%.3f, %.3f)  %s", x0, y0, x1, y1, st)
}

func TestRenderHasWater(t *testing.T) {
	cfg := DefaultConfig(20, 14)
	cfg.Workers = 2
	sim := testSim(t, cfg)
	sim.SeedDam()
	w, h, rgb := sim.Image(2)
	if w != 40 || h != 28 || len(rgb) != w*h*3 {
		t.Fatalf("image %dx%d len %d", w, h, len(rgb))
	}
	water := 0
	for i := 0; i < len(rgb); i += 3 {
		if rgb[i+2] > rgb[i]+20 && rgb[i+2] > 40 {
			water++
		}
	}
	if water < 20 {
		t.Fatalf("only %d water-colored pixels", water)
	}
	ascii := sim.ASCII(20, 14)
	if !containsByte(ascii, '~') || !containsByte(ascii, '#') {
		t.Fatalf("ASCII missing water or walls:\n%s", ascii)
	}
}

func containsByte(s string, c byte) bool {
	for i := 0; i < len(s); i++ {
		if s[i] == c {
			return true
		}
	}
	return false
}

func BenchmarkStep(b *testing.B) {
	cfg := DefaultConfig(64, 40)
	cfg.Nu = 0
	cfg.PCGTol = 1e-3
	cfg.PCGMax = 40
	sim, err := New(cfg)
	if err != nil {
		b.Fatal(err)
	}
	defer sim.Close()
	sim.SeedDam()
	sim.Step(1.0 / 120)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sim.Step(1.0 / 120)
	}
}
