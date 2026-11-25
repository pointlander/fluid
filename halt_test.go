// Copyright 2026 The Fluid Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package main

import "testing"

func TestEasyHalt(t *testing.T) {
	res, err := RunEasyHalt(2)
	if err != nil {
		t.Fatal(err)
	}
	if !res.Halted {
		t.Fatalf("program did not halt: t=%g s x=%g mm increments %d",
			res.Time, res.X*1000, res.Counts)
	}
	if res.Counts != haltSteps {
		t.Fatalf("increments %d, want %d", res.Counts, haltSteps)
	}
	t.Logf("halts at t=%g s, marker x=%g mm", res.Time, res.X*1000)
}
