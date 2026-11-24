// Copyright 2026 The Fluid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

// Command fluid simulates free-surface water with the incompressible
// Navier–Stokes equations on a staggered MAC grid.
//
//	∂u/∂t + (u·∇)u = −(1/ρ)∇p + ν∇²u + g
//	∇·u = 0
//
// Each substep is an operator split: semi-Lagrangian advection, gravity,
// explicit viscosity, then a pressure projection. Marker particles carry
// the liquid. Air cells are a free surface (p = 0) and the tank walls are
// solid. The Poisson problem is solved with Jacobi-preconditioned conjugate
// gradient. Row stencils and element-wise updates run on a goroutine pool
// and use simd.Float32s.
package main

import (
	"fmt"
	"runtime"
	"simd"
)

const (
	cellAir byte = iota
	cellFluid
	cellSolid
)

// Config is the physical and numerical setup of a simulation.
// Use DefaultConfig and override the fields you care about.
type Config struct {
	NX, NY           int     // grid cells, including the one-cell solid wall
	Workers          int     // goroutines in the pool; 0 uses GOMAXPROCS
	Width            float32 // tank width in meters (square cells)
	Rho              float32 // density, kg/m^3 (water is 1000)
	Gravity          float32 // m/s^2, applied in the −y direction
	Nu               float32 // kinematic viscosity, m^2/s (water is about 1e-6)
	CFL              float32 // Courant limit on the advective timestep
	DTMax            float32 // largest substep, seconds
	PCGTol           float32 // relative residual tolerance of the pressure solve
	PCGMax           int     // iteration cap of the pressure solve
	ParticlesPerEdge int     // marker particles along each edge of a seeded cell
	Layers           int     // velocity-extrapolation layers into the air
}

// DefaultConfig returns a water tank Width meters wide.
func DefaultConfig(nx, ny int) Config {
	return Config{
		NX:               nx,
		NY:               ny,
		Width:            1,
		Rho:              1000,
		Gravity:          9.81,
		Nu:               1e-6,
		CFL:              0.5,
		DTMax:            1.0 / 60.0,
		PCGTol:           1e-4,
		PCGMax:           200,
		ParticlesPerEdge: 3,
		Layers:           6,
	}
}

// Stats is one substep of the running liquid.
type Stats struct {
	Time       float32
	DT         float32
	FluidCells int
	Particles  int
	MaxSpeed   float32
	MaxDiv     float32
	PCGIters   int
	Residual   float32
}

func (st Stats) String() string {
	return fmt.Sprintf("t=%6.3f  dt=%.4f  fluid=%5d  particles=%6d  max|u|=%6.3f  max|div|=%.3e  pcg=%3d  resid=%.3e",
		st.Time, st.DT, st.FluidCells, st.Particles, st.MaxSpeed, st.MaxDiv, st.PCGIters, st.Residual)
}

type particle struct {
	x, y float32
}

// Sim is an incompressible free-surface water simulation.
// Close releases its goroutine pool.
type Sim struct {
	nx, ny int
	dx     float32
	rho    float32
	g      float32
	nu     float32
	cfl    float32
	dtMax  float32
	pcgTol float32
	pcgMax int
	ppc    int
	layers int

	t     float32
	steps int

	cell  []byte
	solid []byte

	// Staggered velocity. u is on vertical faces, size (nx+1)*ny.
	// v is on horizontal faces, size nx*(ny+1). Positive v points up.
	u, v       []float32
	uTmp, vTmp []float32

	p []float32

	// Pressure-matrix coefficients on cell centers, size nx*ny.
	// aL is 1 when the west neighbor is not solid, and so on.
	// fluid is 1 on liquid cells. diag = aL+aR+aB+aT.
	aL, aR, aB, aT []float32
	diag, invDiag  []float32
	fluid          []float32
	rhs            []float32

	// Conjugate-gradient workspace, size nx*ny.
	r, z, d, q []float32

	// Face masks, 1 or 0. OK faces do not touch a solid.
	// Fluid faces touch the liquid and do not touch a solid.
	uOK, vOK       []float32
	uFluid, vFluid []float32

	uValid, uValidTmp []byte
	vValid, vValidTmp []byte

	particles []particle
	marks     [][]byte

	pool    *Pool
	lanes   int
	scratch [][]float32
	partial []float64

	fluidCount int
	pcgIters   int
	residual   float32
	maxDiv     float32
	pinned     bool
	pinAt      int
}

// New builds a simulator. The tank walls are the outer cell layer.
func New(cfg Config) (*Sim, error) {
	if cfg.Workers == 0 {
		cfg.Workers = runtime.GOMAXPROCS(0)
	}
	if cfg.Workers < 1 || cfg.Workers > 1024 {
		return nil, fmt.Errorf("workers must be in 1..1024, got %d", cfg.Workers)
	}
	if cfg.NX < 4 || cfg.NY < 4 {
		return nil, fmt.Errorf("grid must be at least 4x4, got %dx%d", cfg.NX, cfg.NY)
	}
	if cfg.Width <= 0 || cfg.Rho <= 0 || cfg.CFL <= 0 || cfg.DTMax <= 0 {
		return nil, fmt.Errorf("width, density, CFL, and dt max must be positive")
	}
	if cfg.Nu < 0 || cfg.PCGTol <= 0 || cfg.PCGMax < 1 || cfg.ParticlesPerEdge < 1 || cfg.Layers < 0 {
		return nil, fmt.Errorf("invalid viscosity, solver tolerance, particle count, or extrapolation layers")
	}

	nx, ny := cfg.NX, cfg.NY
	s := &Sim{
		nx: nx, ny: ny,
		dx:     cfg.Width / float32(nx),
		rho:    cfg.Rho,
		g:      cfg.Gravity,
		nu:     cfg.Nu,
		cfl:    cfg.CFL,
		dtMax:  cfg.DTMax,
		pcgTol: cfg.PCGTol,
		pcgMax: cfg.PCGMax,
		ppc:    cfg.ParticlesPerEdge,
		layers: cfg.Layers,

		cell:  make([]byte, nx*ny),
		solid: make([]byte, nx*ny),

		u:    make([]float32, (nx+1)*ny),
		v:    make([]float32, nx*(ny+1)),
		uTmp: make([]float32, (nx+1)*ny),
		vTmp: make([]float32, nx*(ny+1)),
		p:    make([]float32, nx*ny),

		aL: make([]float32, nx*ny), aR: make([]float32, nx*ny),
		aB: make([]float32, nx*ny), aT: make([]float32, nx*ny),
		diag: make([]float32, nx*ny), invDiag: make([]float32, nx*ny),
		fluid: make([]float32, nx*ny),
		rhs:   make([]float32, nx*ny),
		r:     make([]float32, nx*ny), z: make([]float32, nx*ny),
		d: make([]float32, nx*ny), q: make([]float32, nx*ny),

		uOK: make([]float32, (nx+1)*ny), vOK: make([]float32, nx*(ny+1)),
		uFluid: make([]float32, (nx+1)*ny), vFluid: make([]float32, nx*(ny+1)),
		uValid: make([]byte, (nx+1)*ny), uValidTmp: make([]byte, (nx+1)*ny),
		vValid: make([]byte, nx*(ny+1)), vValidTmp: make([]byte, nx*(ny+1)),

		pool:    NewPool(cfg.Workers),
		scratch: make([][]float32, cfg.Workers),
		partial: make([]float64, cfg.Workers),
		marks:   make([][]byte, cfg.Workers),
	}
	// Lane count is a property of this process. It must not be read from
	// a package-level initializer: those run before SIMD is ready.
	s.lanes = simd.BroadcastFloat32s(0).Len()
	if s.lanes < 1 {
		s.pool.Close()
		return nil, fmt.Errorf("simd reported %d lanes", s.lanes)
	}
	for i := range s.scratch {
		s.scratch[i] = make([]float32, s.lanes)
		s.marks[i] = make([]byte, nx*ny)
	}
	for j := 0; j < ny; j++ {
		for i := 0; i < nx; i++ {
			if i == 0 || j == 0 || i == nx-1 || j == ny-1 {
				s.solid[j*nx+i] = 1
				s.cell[j*nx+i] = cellSolid
			}
		}
	}
	return s, nil
}

// Close stops the goroutine pool. It is safe to call more than once.
func (s *Sim) Close() { s.pool.Close() }

// SIMDLanes is the Float32s lane count selected for this process.
func (s *Sim) SIMDLanes() int { return s.lanes }

// SIMDEmulated reports whether the simd package is running in software.
func (s *Sim) SIMDEmulated() bool { return simd.Emulated() }

// Width is the tank width in meters.
func (s *Sim) Width() float32 { return s.dx * float32(s.nx) }

// Height is the tank height in meters.
func (s *Sim) Height() float32 { return s.dx * float32(s.ny) }

// DX is the cell size in meters.
func (s *Sim) DX() float32 { return s.dx }

// Workers is the size of the goroutine pool.
func (s *Sim) Workers() int { return s.pool.n }

func (s *Sim) resetFields() {
	clear(s.u)
	clear(s.v)
	clear(s.uTmp)
	clear(s.vTmp)
	clear(s.p)
	s.particles = s.particles[:0]
	s.t = 0
	s.steps = 0
	s.maxDiv = 0
	s.residual = 0
	s.pcgIters = 0
}

// damRect is the initial column for a dam break: the left 30% of the
// interior, filled to 70% of the interior height.
func (s *Sim) damRect() (i0, i1, j0, j1 int) {
	innerW := s.nx - 2
	innerH := s.ny - 2
	w := innerW * 30 / 100
	h := innerH * 70 / 100
	if w < 1 {
		w = 1
	}
	if h < 1 {
		h = 1
	}
	if h > innerH-1 {
		h = innerH - 1
	}
	return 1, 1 + w, 1, 1 + h
}

// poolRect fills the bottom 55% of the interior and leaves air above it,
// so the free-surface pressure condition is active.
func (s *Sim) poolRect() (i0, i1, j0, j1 int) {
	innerH := s.ny - 2
	h := innerH * 55 / 100
	if h < 1 {
		h = 1
	}
	if h > innerH-1 {
		h = innerH - 1
	}
	return 1, s.nx - 1, 1, 1 + h
}

func (s *Sim) seedRect(i0, i1, j0, j1 int) {
	n := s.ppc
	inv := 1 / float32(n)
	for j := j0; j < j1; j++ {
		for i := i0; i < i1; i++ {
			for sy := 0; sy < n; sy++ {
				for sx := 0; sx < n; sx++ {
					s.particles = append(s.particles, particle{
						x: (float32(i) + (float32(sx)+0.5)*inv) * s.dx,
						y: (float32(j) + (float32(sy)+0.5)*inv) * s.dx,
					})
				}
			}
		}
	}
}

// SeedDam drops a column of water against the left wall.
func (s *Sim) SeedDam() {
	s.resetFields()
	i0, i1, j0, j1 := s.damRect()
	s.seedRect(i0, i1, j0, j1)
	s.Classify()
}

// SeedPool fills a still pool. One pressure projection should hold it up.
func (s *Sim) SeedPool() {
	s.resetFields()
	i0, i1, j0, j1 := s.poolRect()
	s.seedRect(i0, i1, j0, j1)
	s.Classify()
}

// CenterOfMass is the average marker position, in meters.
func (s *Sim) CenterOfMass() (x, y float32) {
	n := len(s.particles)
	if n == 0 {
		return 0, 0
	}
	var sx, sy float32
	for i := range s.particles {
		sx += s.particles[i].x
		sy += s.particles[i].y
	}
	inv := 1 / float32(n)
	return sx * inv, sy * inv
}

// Classify marks every cell that contains a marker as fluid.
// Wall cells stay solid. Cells with no marker become air.
func (s *Sim) Classify() {
	nx, ny := s.nx, s.ny
	for id, wall := range s.solid {
		if wall == 1 {
			s.cell[id] = cellSolid
		} else {
			s.cell[id] = cellAir
		}
	}
	n := len(s.particles)
	nw := s.pool.n
	s.pool.Run(func(w int) {
		mark := s.marks[w]
		clear(mark)
		if n == 0 {
			return
		}
		a := n * w / nw
		b := n * (w + 1) / nw
		dx := s.dx
		for i := a; i < b; i++ {
			p := s.particles[i]
			ix := int(p.x / dx)
			iy := int(p.y / dx)
			if ix < 1 || iy < 1 || ix >= nx-1 || iy >= ny-1 {
				continue
			}
			mark[iy*nx+ix] = 1
		}
	})
	count := 0
	for id, wall := range s.solid {
		if wall == 1 {
			continue
		}
		for w := 0; w < nw; w++ {
			if s.marks[w][id] == 1 {
				s.cell[id] = cellFluid
				count++
				break
			}
		}
	}
	s.fluidCount = count
}

// Step advances the liquid by one substep of dt seconds.
func (s *Sim) Step(dt float32) Stats {
	s.Classify()
	s.buildCoefficients()
	s.ensureDirichlet()
	s.extrapolate()
	s.enforceSolid()
	s.advectVelocity(dt)
	s.enforceSolid()
	addGravity(s, dt)
	if s.nu > 0 {
		s.viscosity(dt)
		s.enforceSolid()
	}
	s.project(dt)
	s.extrapolate()
	s.enforceSolid()
	s.advectParticles(dt)
	s.Classify()
	s.steps++
	return s.snapshot(dt)
}

// Advance integrates the liquid forward by total seconds, splitting the
// interval into CFL-sized substeps. hook, when not nil, runs after each
// substep.
func (s *Sim) Advance(total float32, hook func(Stats)) (Stats, error) {
	if total < 0 {
		return Stats{}, fmt.Errorf("negative duration %g", total)
	}
	var st Stats
	target := s.t + total
	for s.t < target-1e-8 {
		dt := s.cflDt()
		if remain := target - s.t; dt > remain {
			dt = remain
		}
		st = s.Step(dt)
		s.t += dt
		st.Time = s.t
		if hook != nil {
			hook(st)
		}
		if s.steps > 100000 {
			return st, fmt.Errorf("exceeded 100000 substeps at t=%.4f", s.t)
		}
	}
	st.Time = s.t
	return st, nil
}

func (s *Sim) snapshot(dt float32) Stats {
	maxU := maxAbs(s, s.u)
	if maxV := maxAbs(s, s.v); maxV > maxU {
		maxU = maxV
	}
	return Stats{
		Time:       s.t,
		DT:         dt,
		FluidCells: s.fluidCount,
		Particles:  len(s.particles),
		MaxSpeed:   maxU,
		MaxDiv:     s.maxDiv,
		PCGIters:   s.pcgIters,
		Residual:   s.residual,
	}
}

func (s *Sim) cflDt() float32 {
	m := maxAbs(s, s.u)
	if v := maxAbs(s, s.v); v > m {
		m = v
	}
	dt := s.dtMax
	if m > 1e-8 {
		if c := s.cfl * s.dx / m; c < dt {
			dt = c
		}
	}
	if s.nu > 0 {
		// Explicit 5-point diffusion in 2D wants dt < dx² / (4 ν).
		if vd := 0.2 * s.dx * s.dx / s.nu; vd < dt {
			dt = vd
		}
	}
	if dt < 1e-6 {
		dt = 1e-6
	}
	return dt
}

// parallel splits [0, n) into contiguous bands, one per worker.
// It does not nest: fn must not call parallel.
func (s *Sim) parallel(n int, fn func(w, a, b int)) {
	if n <= 0 {
		return
	}
	workers := s.pool.n
	if workers == 1 {
		fn(0, 0, n)
		return
	}
	s.pool.Run(func(w int) {
		a := n * w / workers
		b := n * (w + 1) / workers
		if a < b {
			fn(w, a, b)
		}
	})
}

func (s *Sim) collide(p *particle) {
	eps := s.dx * 1e-4
	min := s.dx + eps
	maxX := float32(s.nx-1)*s.dx - eps
	maxY := float32(s.ny-1)*s.dx - eps
	if p.x < min {
		p.x = min
	} else if p.x > maxX {
		p.x = maxX
	}
	if p.y < min {
		p.y = min
	} else if p.y > maxY {
		p.y = maxY
	}
}

func (s *Sim) sampleU(x, y float32) float32 {
	return bilerp(s.u, s.nx+1, s.nx, s.ny-1, x/s.dx, y/s.dx-0.5)
}

func (s *Sim) sampleV(x, y float32) float32 {
	return bilerp(s.v, s.nx, s.nx-1, s.ny, x/s.dx-0.5, y/s.dx)
}

func bilerp(field []float32, stride, maxI, maxJ int, gx, gy float32) float32 {
	if gx < 0 {
		gx = 0
	} else if gx > float32(maxI) {
		gx = float32(maxI)
	}
	if gy < 0 {
		gy = 0
	} else if gy > float32(maxJ) {
		gy = float32(maxJ)
	}
	i0 := int(gx)
	j0 := int(gy)
	if i0 > maxI {
		i0 = maxI
	}
	if j0 > maxJ {
		j0 = maxJ
	}
	i1 := i0 + 1
	j1 := j0 + 1
	tx := gx - float32(i0)
	ty := gy - float32(j0)
	if i1 > maxI {
		i1 = maxI
		tx = 0
	}
	if j1 > maxJ {
		j1 = maxJ
		ty = 0
	}
	f00 := field[j0*stride+i0]
	f10 := field[j0*stride+i1]
	f01 := field[j1*stride+i0]
	f11 := field[j1*stride+i1]
	a := f00*(1-tx) + f10*tx
	b := f01*(1-tx) + f11*tx
	return a*(1-ty) + b*ty
}

func (s *Sim) clampDomain(x, y float32) (float32, float32) {
	maxX := float32(s.nx) * s.dx
	maxY := float32(s.ny) * s.dx
	if x < 0 {
		x = 0
	} else if x > maxX {
		x = maxX
	}
	if y < 0 {
		y = 0
	} else if y > maxY {
		y = maxY
	}
	return x, y
}

// backtrace follows the velocity backward one step (semi-Lagrangian).
func (s *Sim) backtrace(x, y, dt float32) (float32, float32) {
	u1 := s.sampleU(x, y)
	v1 := s.sampleV(x, y)
	mx := x - 0.5*dt*u1
	my := y - 0.5*dt*v1
	u2 := s.sampleU(mx, my)
	v2 := s.sampleV(mx, my)
	return s.clampDomain(x-dt*u2, y-dt*v2)
}

// traceForward moves a marker with RK2.
func (s *Sim) traceForward(x, y, dt float32) (float32, float32) {
	u1 := s.sampleU(x, y)
	v1 := s.sampleV(x, y)
	mx := x + 0.5*dt*u1
	my := y + 0.5*dt*v1
	u2 := s.sampleU(mx, my)
	v2 := s.sampleV(mx, my)
	x2, y2 := x+dt*u2, y+dt*v2
	return x2, y2
}

func (s *Sim) advectVelocity(dt float32) {
	nx, ny := s.nx, s.ny
	ust := nx + 1
	s.parallel(ny, func(w, j0, j1 int) {
		for j := j0; j < j1; j++ {
			for i := 0; i < ust; i++ {
				id := j*ust + i
				if s.uFluid[id] == 0 {
					s.uTmp[id] = s.u[id]
					continue
				}
				x := float32(i) * s.dx
				y := (float32(j) + 0.5) * s.dx
				xb, yb := s.backtrace(x, y, dt)
				s.uTmp[id] = s.sampleU(xb, yb)
			}
		}
	})
	s.parallel(ny+1, func(w, j0, j1 int) {
		for j := j0; j < j1; j++ {
			for i := 0; i < nx; i++ {
				id := j*nx + i
				if s.vFluid[id] == 0 {
					s.vTmp[id] = s.v[id]
					continue
				}
				x := (float32(i) + 0.5) * s.dx
				y := float32(j) * s.dx
				xb, yb := s.backtrace(x, y, dt)
				s.vTmp[id] = s.sampleV(xb, yb)
			}
		}
	})
	s.u, s.uTmp = s.uTmp, s.u
	s.v, s.vTmp = s.vTmp, s.v
}

func (s *Sim) advectParticles(dt float32) {
	n := len(s.particles)
	s.parallel(n, func(w, a, b int) {
		for i := a; i < b; i++ {
			p := &s.particles[i]
			x, y := s.traceForward(p.x, p.y, dt)
			p.x, p.y = x, y
			s.collide(p)
		}
	})
}

// extrapolate copies liquid-face velocities a few cells into the air so a
// semi-Lagrangian trace that leaves the water still finds a velocity.
// Solid faces stay zero.
func (s *Sim) extrapolate() {
	s.seedFaceValid(s.u, s.uFluid, s.uValid)
	s.seedFaceValid(s.v, s.vFluid, s.vValid)
	for layer := 0; layer < s.layers; layer++ {
		s.sweepU()
		s.sweepV()
	}
}

func (s *Sim) seedFaceValid(vel, fluid []float32, valid []byte) {
	s.parallel(len(vel), func(w, a, b int) {
		for i := a; i < b; i++ {
			if fluid[i] > 0 {
				valid[i] = 1
			} else {
				valid[i] = 0
				vel[i] = 0
			}
		}
	})
}

func (s *Sim) sweepU() {
	nx, ny := s.nx, s.ny
	stride := nx + 1
	s.parallel(ny, func(w, j0, j1 int) {
		for j := j0; j < j1; j++ {
			for i := 0; i < stride; i++ {
				id := j*stride + i
				if s.uOK[id] == 0 {
					s.uTmp[id] = 0
					s.uValidTmp[id] = 0
					continue
				}
				if s.uValid[id] == 1 {
					s.uTmp[id] = s.u[id]
					s.uValidTmp[id] = 1
					continue
				}
				sum, cnt := float32(0), float32(0)
				if i > 0 && s.uValid[id-1] == 1 {
					sum += s.u[id-1]
					cnt++
				}
				if i+1 < stride && s.uValid[id+1] == 1 {
					sum += s.u[id+1]
					cnt++
				}
				if j > 0 && s.uValid[id-stride] == 1 {
					sum += s.u[id-stride]
					cnt++
				}
				if j+1 < ny && s.uValid[id+stride] == 1 {
					sum += s.u[id+stride]
					cnt++
				}
				if cnt > 0 {
					s.uTmp[id] = sum / cnt
					s.uValidTmp[id] = 1
				} else {
					s.uTmp[id] = 0
					s.uValidTmp[id] = 0
				}
			}
		}
	})
	s.u, s.uTmp = s.uTmp, s.u
	s.uValid, s.uValidTmp = s.uValidTmp, s.uValid
}

func (s *Sim) sweepV() {
	nx, ny := s.nx, s.ny
	rows := ny + 1
	s.parallel(rows, func(w, j0, j1 int) {
		for j := j0; j < j1; j++ {
			for i := 0; i < nx; i++ {
				id := j*nx + i
				if s.vOK[id] == 0 {
					s.vTmp[id] = 0
					s.vValidTmp[id] = 0
					continue
				}
				if s.vValid[id] == 1 {
					s.vTmp[id] = s.v[id]
					s.vValidTmp[id] = 1
					continue
				}
				sum, cnt := float32(0), float32(0)
				if i > 0 && s.vValid[id-1] == 1 {
					sum += s.v[id-1]
					cnt++
				}
				if i+1 < nx && s.vValid[id+1] == 1 {
					sum += s.v[id+1]
					cnt++
				}
				if j > 0 && s.vValid[id-nx] == 1 {
					sum += s.v[id-nx]
					cnt++
				}
				if j+1 < rows && s.vValid[id+nx] == 1 {
					sum += s.v[id+nx]
					cnt++
				}
				if cnt > 0 {
					s.vTmp[id] = sum / cnt
					s.vValidTmp[id] = 1
				} else {
					s.vTmp[id] = 0
					s.vValidTmp[id] = 0
				}
			}
		}
	})
	s.v, s.vTmp = s.vTmp, s.v
	s.vValid, s.vValidTmp = s.vValidTmp, s.vValid
}

// hsum adds the lanes of v. scratch must hold at least one vector.
func hsum(v simd.Float32s, scratch []float32) float64 {
	n := v.Len()
	v.Store(scratch[:n])
	sum := 0.0
	for i := 0; i < n; i++ {
		sum += float64(scratch[i])
	}
	return sum
}

// hmax returns the largest lane of v.
func hmax(v simd.Float32s, scratch []float32) float32 {
	n := v.Len()
	v.Store(scratch[:n])
	m := scratch[0]
	for i := 1; i < n; i++ {
		if scratch[i] > m {
			m = scratch[i]
		}
	}
	return m
}
