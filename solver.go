// Copyright 2026 The Fluid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package main

import "math"

// buildCoefficients assembles the MAC pressure matrix and the face masks
// from the current cell labels.
//
// The discrete projection wants
//
//	div(u) − (dt /(ρ h²)) L(p) = 0
//	L(p) = Σ_n a_n (p_n − p)
//
// with a_n = 1 for a fluid or air neighbor and a_n = 0 for a solid
// neighbor (air pressure is zero). That is A p = b with
//
//	(Ap)_i = (Σ a_n) p_i − Σ a_n p_n
//	b_i = −(ρ h / dt) (u_east − u_west + v_north − v_south)
//
// and the velocity update
//
//	u_face −= (dt /(ρ h)) (p_right − p_left).
func (s *Sim) buildCoefficients() {
	clear(s.aL)
	clear(s.aR)
	clear(s.aB)
	clear(s.aT)
	clear(s.diag)
	clear(s.invDiag)
	clear(s.fluid)
	clear(s.uOK)
	clear(s.vOK)
	clear(s.uFluid)
	clear(s.vFluid)
	s.pinned = false

	nx, ny := s.nx, s.ny
	s.parallel(ny, func(w, j0, j1 int) {
		ust := nx + 1
		for j := j0; j < j1; j++ {
			if j > 0 && j < ny-1 {
				for i := 1; i < nx-1; i++ {
					id := j*nx + i
					if s.cell[id] != cellFluid {
						continue
					}
					s.fluid[id] = 1
					if s.cell[id-1] != cellSolid {
						s.aL[id] = 1
					}
					if s.cell[id+1] != cellSolid {
						s.aR[id] = 1
					}
					if s.cell[id-nx] != cellSolid {
						s.aB[id] = 1
					}
					if s.cell[id+nx] != cellSolid {
						s.aT[id] = 1
					}
					d := s.aL[id] + s.aR[id] + s.aB[id] + s.aT[id]
					if d == 0 {
						// Isolated liquid cell: keep the row nonsingular.
						d = 1
					}
					s.diag[id] = d
					s.invDiag[id] = 1 / d
				}
			}
			for i := 0; i <= nx; i++ {
				left, right := cellSolid, cellSolid
				if i > 0 {
					left = s.cell[j*nx+i-1]
				}
				if i < nx {
					right = s.cell[j*nx+i]
				}
				uid := j*ust + i
				if left != cellSolid && right != cellSolid {
					s.uOK[uid] = 1
					if left == cellFluid || right == cellFluid {
						s.uFluid[uid] = 1
					}
				}
			}
		}
	})
	s.parallel(ny+1, func(w, j0, j1 int) {
		for j := j0; j < j1; j++ {
			for i := 0; i < nx; i++ {
				below, above := cellSolid, cellSolid
				if j > 0 {
					below = s.cell[(j-1)*nx+i]
				}
				if j < ny {
					above = s.cell[j*nx+i]
				}
				vid := j*nx + i
				if below != cellSolid && above != cellSolid {
					s.vOK[vid] = 1
					if below == cellFluid || above == cellFluid {
						s.vFluid[vid] = 1
					}
				}
			}
		}
	})
}

// ensureDirichlet pins one liquid cell when the tank has no air, so the
// all-Neumann pressure system is not singular. A free surface already
// supplies the Dirichlet condition and no pin is used.
func (s *Sim) ensureDirichlet() {
	s.pinned = false
	nx, ny := s.nx, s.ny
	pin := -1
	for j := 1; j < ny-1; j++ {
		for i := 1; i < nx-1; i++ {
			id := j*nx + i
			if s.fluid[id] == 0 {
				continue
			}
			if pin < 0 {
				pin = id
			}
			if s.aL[id] == 1 && s.fluid[id-1] == 0 ||
				s.aR[id] == 1 && s.fluid[id+1] == 0 ||
				s.aB[id] == 1 && s.fluid[id-nx] == 0 ||
				s.aT[id] == 1 && s.fluid[id+nx] == 0 {
				return
			}
		}
	}
	if pin < 0 {
		return
	}
	s.aL[pin], s.aR[pin], s.aB[pin], s.aT[pin] = 0, 0, 0, 0
	s.diag[pin] = 1
	s.invDiag[pin] = 1
	s.pinned = true
	s.pinAt = pin
}

// project makes the velocity divergence-free inside the liquid.
func (s *Sim) project(dt float32) {
	if dt <= 0 {
		return
	}
	// b = −(ρ h / dt) * (u_east − u_west + v_north − v_south), liquid only.
	divergence(s, s.rhs, -s.rho*s.dx/dt)
	if s.pinned {
		s.rhs[s.pinAt] = 0
	}
	s.solve()
	applyGradient(s, dt)
	s.enforceSolid()
	divergence(s, s.q, 1/s.dx)
	s.maxDiv = maxAbs(s, s.q)
}

// solve runs Jacobi-preconditioned conjugate gradient on A p = rhs.
// Pressure outside the liquid is kept at zero (the free-surface value).
func (s *Sim) solve() {
	mulInto(s, s.p, s.p, s.fluid)
	spmv(s, s.p, s.q)
	axpy(s, s.r, s.rhs, s.q, -1)
	mulInto(s, s.z, s.r, s.invDiag)
	copyFlat(s, s.d, s.z)

	rho := dot(s, s.r, s.z)
	b2 := dot(s, s.rhs, s.rhs)
	tol := float64(s.pcgTol) * math.Sqrt(b2)
	if tol < 1e-10 {
		tol = 1e-10
	}
	s.pcgIters = 0
	if math.Sqrt(dot(s, s.r, s.r)) <= tol {
		s.residual = float32(math.Sqrt(dot(s, s.r, s.r)))
		return
	}

	for iter := 0; iter < s.pcgMax; iter++ {
		s.pcgIters = iter + 1
		if iter > 0 && iter%40 == 0 {
			spmv(s, s.p, s.q)
			axpy(s, s.r, s.rhs, s.q, -1)
			mulInto(s, s.z, s.r, s.invDiag)
			copyFlat(s, s.d, s.z)
			rho = dot(s, s.r, s.z)
		}
		spmv(s, s.d, s.q)
		dq := dot(s, s.d, s.q)
		if dq <= 1e-30 {
			break
		}
		alpha := rho / dq
		axpy(s, s.p, s.p, s.d, float32(alpha))
		axpy(s, s.r, s.r, s.q, float32(-alpha))
		r2 := dot(s, s.r, s.r)
		if math.Sqrt(r2) <= tol {
			break
		}
		mulInto(s, s.z, s.r, s.invDiag)
		rhoNew := dot(s, s.r, s.z)
		if rho == 0 {
			break
		}
		beta := rhoNew / rho
		rho = rhoNew
		axpy(s, s.d, s.z, s.d, float32(beta))
	}
	spmv(s, s.p, s.q)
	axpy(s, s.r, s.rhs, s.q, -1)
	s.residual = float32(math.Sqrt(dot(s, s.r, s.r)))
}
