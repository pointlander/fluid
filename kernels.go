// Copyright 2026 The Fluid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package main

import "simd"

// copyFlat copies src into dst. Lengths must match.
func copyFlat(s *Sim, dst, src []float32) {
	lanes := s.lanes
	s.parallel(len(dst), func(w, i0, i1 int) {
		i := i0
		for ; i+lanes <= i1; i += lanes {
			simd.LoadFloat32s(src[i:]).Store(dst[i:])
		}
		for ; i < i1; i++ {
			dst[i] = src[i]
		}
	})
}

// axpy sets dst = y + alpha*x. The slices share a length.
// dst may alias x or y; each vector is fully computed before it is stored.
func axpy(s *Sim, dst, y, x []float32, alpha float32) {
	lanes := s.lanes
	alphaV := simd.BroadcastFloat32s(alpha)
	s.parallel(len(dst), func(w, i0, i1 int) {
		i := i0
		for ; i+lanes <= i1; i += lanes {
			simd.LoadFloat32s(x[i:]).MulAdd(alphaV, simd.LoadFloat32s(y[i:])).Store(dst[i:])
		}
		for ; i < i1; i++ {
			dst[i] = y[i] + alpha*x[i]
		}
	})
}

// mulInto sets dst = a*b.
func mulInto(s *Sim, dst, a, b []float32) {
	lanes := s.lanes
	s.parallel(len(dst), func(w, i0, i1 int) {
		i := i0
		for ; i+lanes <= i1; i += lanes {
			simd.LoadFloat32s(a[i:]).Mul(simd.LoadFloat32s(b[i:])).Store(dst[i:])
		}
		for ; i < i1; i++ {
			dst[i] = a[i] * b[i]
		}
	})
}

// enforceSolid zeros every velocity that touches a wall.
func (s *Sim) enforceSolid() {
	mulInto(s, s.u, s.u, s.uOK)
	mulInto(s, s.v, s.v, s.vOK)
}

// addGravity applies ∂v/∂t = −g on faces that touch the liquid.
func addGravity(s *Sim, dt float32) {
	lanes := s.lanes
	gdt := simd.BroadcastFloat32s(-s.g * dt)
	v := s.v
	mask := s.vFluid
	s.parallel(len(v), func(w, i0, i1 int) {
		i := i0
		for ; i+lanes <= i1; i += lanes {
			// v += (−g dt) on liquid faces, unchanged elsewhere.
			simd.LoadFloat32s(mask[i:]).MulAdd(gdt, simd.LoadFloat32s(v[i:])).Store(v[i:])
		}
		g := -s.g * dt
		for ; i < i1; i++ {
			v[i] += g * mask[i]
		}
	})
}

// dot is the sum of a[i]*b[i] over the shared length, accumulated in float64.
func dot(s *Sim, a, b []float32) float64 {
	for i := range s.partial {
		s.partial[i] = 0
	}
	lanes := s.lanes
	s.parallel(len(a), func(w, i0, i1 int) {
		sum := 0.0
		scratch := s.scratch[w]
		i := i0
		for ; i+lanes <= i1; i += lanes {
			sum += hsum(simd.LoadFloat32s(a[i:]).Mul(simd.LoadFloat32s(b[i:])), scratch)
		}
		for ; i < i1; i++ {
			sum += float64(a[i]) * float64(b[i])
		}
		s.partial[w] = sum
	})
	total := 0.0
	for _, p := range s.partial {
		total += p
	}
	return total
}

// maxAbs is the maximum of |a[i]|.
func maxAbs(s *Sim, a []float32) float32 {
	for i := range s.partial {
		s.partial[i] = 0
	}
	lanes := s.lanes
	s.parallel(len(a), func(w, i0, i1 int) {
		m := float32(0)
		scratch := s.scratch[w]
		i := i0
		for ; i+lanes <= i1; i += lanes {
			if h := hmax(simd.LoadFloat32s(a[i:]).Abs(), scratch); h > m {
				m = h
			}
		}
		for ; i < i1; i++ {
			x := a[i]
			if x < 0 {
				x = -x
			}
			if x > m {
				m = x
			}
		}
		s.partial[w] = float64(m)
	})
	m := float32(0)
	for _, p := range s.partial {
		if x := float32(p); x > m {
			m = x
		}
	}
	return m
}

// divergence writes scale * (∂u/∂x + ∂v/∂y) * h into dst, masked to liquid
// cells. Passing scale = 1/h stores the physical divergence. The face
// difference (u_east − u_west + v_north − v_south) is already h times
// the divergence, so the physical divergence uses scale 1/h.
func divergence(s *Sim, dst []float32, scale float32) {
	nx, ny := s.nx, s.ny
	lanes := s.lanes
	ust := nx + 1
	scaleV := simd.BroadcastFloat32s(scale)
	s.parallel(ny, func(w, j0, j1 int) {
		for j := j0; j < j1; j++ {
			uRow := s.u[j*ust:]
			vRow := s.v[j*nx:]
			vAbove := s.v[(j+1)*nx:]
			out := dst[j*nx:]
			fl := s.fluid[j*nx:]
			i := 0
			for ; i+lanes <= nx; i += lanes {
				du := simd.LoadFloat32s(uRow[i+1:]).Sub(simd.LoadFloat32s(uRow[i:]))
				dv := simd.LoadFloat32s(vAbove[i:]).Sub(simd.LoadFloat32s(vRow[i:]))
				du.Add(dv).Mul(scaleV).Mul(simd.LoadFloat32s(fl[i:])).Store(out[i:])
			}
			for ; i < nx; i++ {
				d := uRow[i+1] - uRow[i] + vAbove[i] - vRow[i]
				out[i] = scale * d * fl[i]
			}
		}
	})
}

// spmv computes dst = A p for the variable-coefficient Laplacian.
//
//	(Ap)_i = diag_i p_i − aL p_west − aR p_east − aB p_south − aT p_north
//
// Air pressure is stored as zero, which is the free-surface condition.
// Solid neighbors have a zero coefficient, which is the wall condition.
func spmv(s *Sim, p, dst []float32) {
	nx, ny := s.nx, s.ny
	lanes := s.lanes
	s.parallel(ny, func(w, j0, j1 int) {
		for j := j0; j < j1; j++ {
			row := dst[j*nx : (j+1)*nx]
			if j == 0 || j == ny-1 {
				for i := range row {
					row[i] = 0
				}
				continue
			}
			row[0] = 0
			row[nx-1] = 0
			base := j * nx
			i := 1
			end := nx - 1
			for ; i+lanes <= end; i += lanes {
				id := base + i
				center := simd.LoadFloat32s(p[id:])
				left := simd.LoadFloat32s(p[id-1:])
				right := simd.LoadFloat32s(p[id+1:])
				below := simd.LoadFloat32s(p[id-nx:])
				above := simd.LoadFloat32s(p[id+nx:])
				acc := simd.LoadFloat32s(s.diag[id:]).Mul(center)
				acc = acc.Sub(simd.LoadFloat32s(s.aL[id:]).Mul(left))
				acc = acc.Sub(simd.LoadFloat32s(s.aR[id:]).Mul(right))
				acc = acc.Sub(simd.LoadFloat32s(s.aB[id:]).Mul(below))
				acc = acc.Sub(simd.LoadFloat32s(s.aT[id:]).Mul(above))
				acc.Mul(simd.LoadFloat32s(s.fluid[id:])).Store(dst[id:])
			}
			for ; i < end; i++ {
				dst[base+i] = s.apAt(i, j, p)
			}
		}
	})
}

// apAt is the scalar Laplacian used on vector tails and as a reference.
func (s *Sim) apAt(i, j int, p []float32) float32 {
	id := j*s.nx + i
	if s.fluid[id] == 0 {
		return 0
	}
	return s.diag[id]*p[id] -
		s.aL[id]*p[id-1] -
		s.aR[id]*p[id+1] -
		s.aB[id]*p[id-s.nx] -
		s.aT[id]*p[id+s.nx]
}

// applyGradient subtracts (dt/ρ) ∇p from the velocity.
// A face that touches a solid is left alone (uOK / vOK is zero there).
func applyGradient(s *Sim, dt float32) {
	nx, ny := s.nx, s.ny
	lanes := s.lanes
	scale := dt / (s.rho * s.dx)
	scaleV := simd.BroadcastFloat32s(scale)
	ust := nx + 1

	s.parallel(ny, func(w, j0, j1 int) {
		for j := j0; j < j1; j++ {
			uRow := s.u[j*ust:]
			ok := s.uOK[j*ust:]
			pRow := s.p[j*nx:]
			// Faces 1..nx-1 sit between two real cells. The wall faces
			// at 0 and nx stay at the solid velocity.
			i := 1
			end := nx
			for ; i+lanes <= end; i += lanes {
				pL := simd.LoadFloat32s(pRow[i-1:])
				pR := simd.LoadFloat32s(pRow[i:])
				delta := pR.Sub(pL).Mul(scaleV).Mul(simd.LoadFloat32s(ok[i:]))
				simd.LoadFloat32s(uRow[i:]).Sub(delta).Store(uRow[i:])
			}
			for ; i < end; i++ {
				uRow[i] -= scale * (pRow[i] - pRow[i-1]) * ok[i]
			}
		}
	})

	s.parallel(ny-1, func(w, j0, j1 int) {
		// Bands cover output rows [1, ny). parallel's range is shifted.
		for j := j0 + 1; j < j1+1; j++ {
			vRow := s.v[j*nx:]
			ok := s.vOK[j*nx:]
			pA := s.p[j*nx:]
			pB := s.p[(j-1)*nx:]
			i := 0
			for ; i+lanes <= nx; i += lanes {
				delta := simd.LoadFloat32s(pA[i:]).Sub(simd.LoadFloat32s(pB[i:])).Mul(scaleV).Mul(simd.LoadFloat32s(ok[i:]))
				simd.LoadFloat32s(vRow[i:]).Sub(delta).Store(vRow[i:])
			}
			for ; i < nx; i++ {
				vRow[i] -= scale * (pA[i] - pB[i]) * ok[i]
			}
		}
	})
}

// viscosity applies the explicit term ∂u/∂t = ν ∇²u on liquid faces.
func (s *Sim) viscosity(dt float32) {
	viscousU(s, dt)
	viscousV(s, dt)
	s.u, s.uTmp = s.uTmp, s.u
	s.v, s.vTmp = s.vTmp, s.v
}

func viscousU(s *Sim, dt float32) {
	nx, ny := s.nx, s.ny
	lanes := s.lanes
	ust := nx + 1
	invH2 := 1 / (s.dx * s.dx)
	dtNu := s.nu * dt
	dtNuV := simd.BroadcastFloat32s(dtNu * invH2)
	four := simd.BroadcastFloat32s(4)
	zero := simd.BroadcastFloat32s(0)
	copyFlat(s, s.uTmp, s.u)
	s.parallel(ny-2, func(w, j0, j1 int) {
		for j := j0 + 1; j < j1+1; j++ {
			uRow := s.u[j*ust:]
			below := s.u[(j-1)*ust:]
			above := s.u[(j+1)*ust:]
			out := s.uTmp[j*ust:]
			mask := s.uFluid[j*ust:]
			i := 1
			end := nx
			for ; i+lanes <= end; i += lanes {
				center := simd.LoadFloat32s(uRow[i:])
				left := simd.LoadFloat32s(uRow[i-1:])
				right := simd.LoadFloat32s(uRow[i+1:])
				lap := left.Add(right).Add(simd.LoadFloat32s(below[i:])).Add(simd.LoadFloat32s(above[i:]))
				lap = lap.Sub(center.Mul(four))
				updated := lap.MulAdd(dtNuV, center)
				keep := simd.LoadFloat32s(mask[i:]).Greater(zero)
				updated.IfElse(keep, center).Store(out[i:])
			}
			for ; i < end; i++ {
				if mask[i] == 0 {
					out[i] = uRow[i]
					continue
				}
				lap := uRow[i-1] + uRow[i+1] + below[i] + above[i] - 4*uRow[i]
				out[i] = uRow[i] + dtNu*invH2*lap
			}
		}
	})
}

func viscousV(s *Sim, dt float32) {
	nx, ny := s.nx, s.ny
	lanes := s.lanes
	invH2 := 1 / (s.dx * s.dx)
	dtNu := s.nu * dt
	dtNuV := simd.BroadcastFloat32s(dtNu * invH2)
	four := simd.BroadcastFloat32s(4)
	zero := simd.BroadcastFloat32s(0)
	copyFlat(s, s.vTmp, s.v)
	// Faces j = 1..ny-1 have both vertical neighbors inside the v grid
	// (rows 0..ny). Skip j = 0 and j = ny, which are the outer walls.
	s.parallel(ny-1, func(w, j0, j1 int) {
		for j := j0 + 1; j < j1+1; j++ {
			vRow := s.v[j*nx:]
			below := s.v[(j-1)*nx:]
			above := s.v[(j+1)*nx:]
			out := s.vTmp[j*nx:]
			mask := s.vFluid[j*nx:]
			i := 1
			end := nx - 1
			for ; i+lanes <= end; i += lanes {
				center := simd.LoadFloat32s(vRow[i:])
				left := simd.LoadFloat32s(vRow[i-1:])
				right := simd.LoadFloat32s(vRow[i+1:])
				lap := left.Add(right).Add(simd.LoadFloat32s(below[i:])).Add(simd.LoadFloat32s(above[i:]))
				lap = lap.Sub(center.Mul(four))
				updated := lap.MulAdd(dtNuV, center)
				keep := simd.LoadFloat32s(mask[i:]).Greater(zero)
				updated.IfElse(keep, center).Store(out[i:])
			}
			for ; i < end; i++ {
				if mask[i] == 0 {
					out[i] = vRow[i]
					continue
				}
				lap := vRow[i-1] + vRow[i+1] + below[i] + above[i] - 4*vRow[i]
				out[i] = vRow[i] + dtNu*invH2*lap
			}
			out[0] = vRow[0]
			out[nx-1] = vRow[nx-1]
		}
	})
}
