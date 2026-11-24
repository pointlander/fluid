// Copyright 2026 The Fluid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package main

import "math"

// acousticMedium is the water the transducers radiate into.
//
// Eckart streaming is the time average of a progressive ultrasonic wave.
// With complex particle velocity V and complex amplitude sum S = Σ U_i
// (each progressive wave has pressure ρ c U_i), the streaming acceleration is
//
//	a = α Re(S conj(V))
//
// which is α |U|² along the beam for a single wave. Superposition is linear
// in U, so antiphase beams cancel before the square.
type acousticMedium struct {
	WaveNumber  float32 // rad/m
	Attenuation float32 // 1/m, streaming coefficient α
}

// transducer is one ultrasonic piston on the boundary of the fluid body.
// The beam is a collimated Gaussian in the piston near field: amplitude
// A exp(−α s) exp(−n² / 2σ²) exp(i (k s + φ)), aimed along Dir.
type transducer struct {
	X, Y       float32
	DirX, DirY float32
	Sigma      float32
	Amplitude  float32 // acoustic particle velocity, m/s; 0 is off
	Phase      float32 // radians
}

// complexAmplitude is the progressive-wave particle-velocity amplitude at (x, y).
func (t transducer) complexAmplitude(m acousticMedium, x, y float32) (re, im float32) {
	if t.Amplitude == 0 || t.Sigma <= 0 {
		return 0, 0
	}
	sx := x - t.X
	sy := y - t.Y
	s := sx*t.DirX + sy*t.DirY
	if s <= 0 {
		return 0, 0
	}
	n := -sx*t.DirY + sy*t.DirX
	env := t.Amplitude * float32(math.Exp(float64(-m.Attenuation*s-n*n/(2*t.Sigma*t.Sigma))))
	th := float64(m.WaveNumber*s + t.Phase)
	return env * float32(math.Cos(th)), env * float32(math.Sin(th))
}

// fieldAccel is the Eckart streaming acceleration, in m/s², at one point.
func fieldAccel(trs []transducer, m acousticMedium, x, y float32) (ax, ay float32) {
	var sumRe, sumIm float32
	var vxRe, vxIm, vyRe, vyIm float32
	for i := range trs {
		re, im := trs[i].complexAmplitude(m, x, y)
		sumRe += re
		sumIm += im
		vxRe += re * trs[i].DirX
		vxIm += im * trs[i].DirX
		vyRe += re * trs[i].DirY
		vyIm += im * trs[i].DirY
	}
	// Re(S conj(V)) = sumRe*vRe + sumIm*vIm.
	ax = m.Attenuation * (sumRe*vxRe + sumIm*vxIm)
	ay = m.Attenuation * (sumRe*vyRe + sumIm*vyIm)
	return ax, ay
}

// fillStreaming writes the streaming acceleration onto the staggered faces.
func fillStreaming(s *Sim, trs []transducer, m acousticMedium, au, av []float32) {
	nx, ny := s.nx, s.ny
	dx := s.dx
	s.parallel(ny, func(w, j0, j1 int) {
		ust := nx + 1
		for j := j0; j < j1; j++ {
			y := (float32(j) + 0.5) * dx
			row := j * ust
			for i := 0; i <= nx; i++ {
				ax, _ := fieldAccel(trs, m, float32(i)*dx, y)
				au[row+i] = ax
			}
		}
	})
	s.parallel(ny+1, func(w, j0, j1 int) {
		for j := j0; j < j1; j++ {
			y := float32(j) * dx
			row := j * nx
			for i := 0; i < nx; i++ {
				_, ay := fieldAccel(trs, m, (float32(i)+0.5)*dx, y)
				av[row+i] = ay
			}
		}
	})
}
