# fluid

Incompressible water on a staggered grid, in Go. The free surface is a dam break or a still pool. The same solver drives ultrasonic logic in one sealed body of water: an XOR gate and a CNOT gate, with no internal channels.

The momentum equation is

```text
∂u/∂t + (u·∇)u = −(1/ρ)∇p + ν∇²u + g + a
∇·u = 0
```

`a` is the Eckart streaming acceleration from the ultrasonic pistons. It is zero for the dam and the pool. Each substep advects the velocity, adds gravity and streaming, applies viscosity, then projects the pressure with Jacobi-preconditioned conjugate gradient. Marker particles carry a free surface. A sealed body keeps every interior cell liquid. Row stencils and element-wise updates run on a goroutine pool and on `simd.Float32s`.

## Build

Go 1.27 or newer, with the experimental SIMD package enabled:

```bash
export GOEXPERIMENT=simd
go test ./...
go build -o fluid .
```

## Run

```bash
go run .                         # dam break, 1 s, 80×48
go run . -scene pool             # still water
go run . -preview                # truecolor frames on the terminal
go run . -out frames             # PPM sequence
go run . -scene xor              # ultrasonic XOR truth table
go run . -scene xor -a 1 -b 0    # one XOR case
go run . -scene cnot             # ultrasonic CNOT, then CNOT|+0>
go run . -scene cnot -a 1 -b 0   # one CNOT basis state
```

`-time` is simulated seconds. For `xor` and `cnot` it is the piston drive time, and it defaults to 0.05 s when the flag is omitted. `-workers` defaults to `GOMAXPROCS`. `-nx` and `-ny` set the tank for `dam` and `pool`.

Water defaults are ρ = 1000 kg/m³, g = 9.81 m/s², and ν = 1e-6 m²/s.

## Gates

Both gates use one sealed square of water and wall-mounted ultrasonic pistons at 1 MHz. Streaming is the time average of the progressive beams, `a = α Re(S conj(V))`. Antiphase beams cancel before that square, so two pistons aimed the same way leave a jet only when exactly one is on.

**XOR.** Two pistons on the left wall. The output is the midline jet, read against half the single-piston speed.

**CNOT.** A vertical beam from the bottom wall passes the control bit through. An antiphase pair on the midline sets the target jet to control XOR target. A general state is split into computational-basis branches; the fluid routes each branch and the complex amplitude is carried along. `CNOT|+0⟩` is the Bell state `(|00⟩ + |11⟩)/√2`.

## License

BSD, see [LICENSE](LICENSE). Copyright 2026 The Fluid Authors.
