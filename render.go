// Copyright 2026 The Fluid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package main

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"strings"
)

// surfaces returns, for each column, the highest liquid cell, or −1.
func (s *Sim) surfaces() []int {
	nx, ny := s.nx, s.ny
	out := make([]int, nx)
	for i := range out {
		out[i] = -1
		for j := ny - 1; j >= 0; j-- {
			if s.cell[j*nx+i] == cellFluid {
				out[i] = j
				break
			}
		}
	}
	return out
}

func (s *Sim) cellColor(i, j, surface int) (byte, byte, byte) {
	switch s.cell[j*s.nx+i] {
	case cellSolid:
		return 68, 72, 84
	case cellAir:
		return 7, 10, 22
	default:
		uc := 0.5 * (s.u[j*(s.nx+1)+i] + s.u[j*(s.nx+1)+i+1])
		vc := 0.5 * (s.v[j*s.nx+i] + s.v[(j+1)*s.nx+i])
		speed := float32(math.Sqrt(float64(uc*uc + vc*vc)))
		foam := speed / 1.2
		if foam > 1 {
			foam = 1
		}
		depth := float32(0)
		if surface > j {
			depth = float32(surface-j) / float32(surface)
			if depth > 1 {
				depth = 1
			}
		}
		sr, sg, sb := float32(64), float32(168), float32(214)
		dr, dg, db := float32(6), float32(36), float32(92)
		fr, fg, fb := float32(232), float32(244), float32(252)
		r := dr*depth + sr*(1-depth)
		g := dg*depth + sg*(1-depth)
		b := db*depth + sb*(1-depth)
		r = r*(1-foam) + fr*foam
		g = g*(1-foam) + fg*foam
		b = b*(1-foam) + fb*foam
		return byte(r), byte(g), byte(b)
	}
}

// Image renders the tank as packed RGB. The first row is the top of the tank.
// scale is the number of pixels on each side of a cell.
func (s *Sim) Image(scale int) (w, h int, rgb []byte) {
	if scale < 1 {
		scale = 1
	}
	w = s.nx * scale
	h = s.ny * scale
	rgb = make([]byte, w*h*3)
	surf := s.surfaces()
	for py := 0; py < h; py++ {
		j := (h - 1 - py) / scale
		for px := 0; px < w; px++ {
			i := px / scale
			r, g, b := s.cellColor(i, j, surf[i])
			o := (py*w + px) * 3
			rgb[o] = r
			rgb[o+1] = g
			rgb[o+2] = b
		}
	}
	return w, h, rgb
}

// WritePPM writes a binary PPM of the tank.
func (s *Sim) WritePPM(path string, scale int) error {
	w, h, rgb := s.Image(scale)
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "P6\n%d %d\n255\n", w, h)
	buf.Write(rgb)
	return os.WriteFile(path, buf.Bytes(), 0o644)
}

// ASCII draws the tank with '#' for walls, '~' for water, and spaces for air.
// cols and rows downsample the grid; values above the grid size are clamped.
func (s *Sim) ASCII(cols, rows int) string {
	if cols < 1 || cols > s.nx {
		cols = s.nx
	}
	if rows < 1 || rows > s.ny {
		rows = s.ny
	}
	var b strings.Builder
	b.Grow((cols + 1) * rows)
	for r := 0; r < rows; r++ {
		jLo := (rows - r - 1) * s.ny / rows
		jHi := (rows - r) * s.ny / rows
		if jHi <= jLo {
			jHi = jLo + 1
		}
		for c := 0; c < cols; c++ {
			iLo := c * s.nx / cols
			iHi := (c + 1) * s.nx / cols
			if iHi <= iLo {
				iHi = iLo + 1
			}
			solid, fluid := 0, 0
			for j := jLo; j < jHi; j++ {
				for i := iLo; i < iHi; i++ {
					switch s.cell[j*s.nx+i] {
					case cellSolid:
						solid++
					case cellFluid:
						fluid++
					}
				}
			}
			switch {
			case solid > 0 && solid >= fluid:
				b.WriteByte('#')
			case fluid > 0:
				b.WriteByte('~')
			default:
				b.WriteByte(' ')
			}
		}
		b.WriteByte('\n')
	}
	return b.String()
}

// ANSI draws the tank with truecolor blocks.
func (s *Sim) ANSI(cols, rows int) string {
	if cols < 1 || cols > s.nx {
		cols = s.nx
	}
	if rows < 1 || rows > s.ny {
		rows = s.ny
	}
	surf := s.surfaces()
	var b strings.Builder
	for r := 0; r < rows; r++ {
		jLo := (rows - r - 1) * s.ny / rows
		jHi := (rows - r) * s.ny / rows
		if jHi <= jLo {
			jHi = jLo + 1
		}
		for c := 0; c < cols; c++ {
			iLo := c * s.nx / cols
			iHi := (c + 1) * s.nx / cols
			if iHi <= iLo {
				iHi = iLo + 1
			}
			// Prefer a liquid sample so a splash is not hidden by air.
			si, sj := iLo, jLo
			found := byte(cellAir)
			for j := jLo; j < jHi && found != cellFluid; j++ {
				for i := iLo; i < iHi; i++ {
					k := s.cell[j*s.nx+i]
					if k == cellFluid || (k == cellSolid && found == cellAir) {
						si, sj, found = i, j, k
						if k == cellFluid {
							break
						}
					}
				}
			}
			cr, cg, cb := s.cellColor(si, sj, surf[si])
			fmt.Fprintf(&b, "\x1b[38;2;%d;%d;%dm█", cr, cg, cb)
		}
		b.WriteString("\x1b[0m\n")
	}
	return b.String()
}
