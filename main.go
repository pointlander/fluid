// Copyright 2026 The Fluid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
)

func main() {
	nx := flag.Int("nx", 80, "cells across the tank, including the solid wall")
	ny := flag.Int("ny", 48, "cells up the tank, including the solid wall")
	timeSec := flag.Float64("time", 1, "seconds of simulated time")
	outDir := flag.String("out", "", "directory for PPM frames; empty skips writing")
	frameDt := flag.Float64("frame-dt", 0.05, "seconds of simulation between PPM frames")
	preview := flag.Bool("preview", false, "draw a truecolor frame on stdout as the liquid runs")
	workers := flag.Int("workers", 0, "goroutine workers (0 uses GOMAXPROCS)")
	scene := flag.String("scene", "dam", "initial water: dam (column collapse) or pool (still water)")
	nu := flag.Float64("nu", 1e-6, "kinematic viscosity in m^2/s; water at 20°C is about 1e-6")
	g := flag.Float64("g", 9.81, "gravity in m/s^2, applied downward")
	rho := flag.Float64("rho", 1000, "density in kg/m^3")
	scale := flag.Int("scale", 4, "PPM pixels on each side of a cell")
	ppc := flag.Int("ppc", 3, "marker particles along each edge of a seeded cell")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: fluid [flags]\n\n")
		fmt.Fprintf(os.Stderr, "Free-surface water, from the incompressible Navier-Stokes equations\n")
		fmt.Fprintf(os.Stderr, "on a staggered grid. Build with Go 1.27+ and GOEXPERIMENT=simd.\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	cfg := DefaultConfig(*nx, *ny)
	cfg.Workers = *workers
	cfg.Nu = float32(*nu)
	cfg.Gravity = float32(*g)
	cfg.Rho = float32(*rho)
	cfg.ParticlesPerEdge = *ppc
	sim, err := New(cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer sim.Close()

	switch *scene {
	case "dam":
		sim.SeedDam()
	case "pool":
		sim.SeedPool()
	default:
		fmt.Fprintf(os.Stderr, "unknown scene %q (want dam or pool)\n", *scene)
		os.Exit(1)
	}

	simdKind := "hardware"
	if sim.SIMDEmulated() {
		simdKind = "emulated"
	}
	fmt.Printf("incompressible water  %dx%d  dx=%.4f m  domain %.3f m × %.3f m\n",
		*nx, *ny, sim.DX(), sim.Width(), sim.Height())
	fmt.Printf("simd %s  lanes=%d (%d-bit)  workers=%d  scene=%s  ρ=%.0f  g=%.2f  ν=%.3g\n",
		simdKind, sim.SIMDLanes(), sim.SIMDLanes()*32, sim.Workers(), *scene, *rho, *g, *nu)
	cx, cy := sim.CenterOfMass()
	fmt.Printf("t=0  particles=%d  center of mass (%.3f, %.3f) m\n", len(sim.particles), cx, cy)
	fmt.Print(sim.ASCII(72, 28))

	if *outDir != "" {
		if err := os.MkdirAll(*outDir, 0o755); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		if err := sim.WritePPM(filepath.Join(*outDir, "frame-0000.ppm"), *scale); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}

	nextPrint := 0.1
	nextFrame := *frameDt
	frame := 1
	var writeErr error
	last, err := sim.Advance(float32(*timeSec), func(st Stats) {
		if writeErr != nil {
			return
		}
		if *preview {
			fmt.Print("\033[H\033[2J")
			fmt.Print(sim.ANSI(72, 28))
			fmt.Println(st)
		} else if float64(st.Time) >= nextPrint {
			fmt.Println(st)
			for float64(st.Time) >= nextPrint {
				nextPrint += 0.1
			}
		}
		if *outDir != "" && float64(st.Time) >= nextFrame {
			name := filepath.Join(*outDir, fmt.Sprintf("frame-%04d.ppm", frame))
			if werr := sim.WritePPM(name, *scale); werr != nil {
				writeErr = werr
				return
			}
			frame++
			for float64(st.Time) >= nextFrame {
				nextFrame += *frameDt
			}
		}
	})
	if writeErr != nil {
		err = writeErr
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	cx, cy = sim.CenterOfMass()
	fmt.Println(last)
	fmt.Printf("center of mass (%.3f, %.3f) m\n", cx, cy)
	fmt.Print(sim.ASCII(72, 28))
}
