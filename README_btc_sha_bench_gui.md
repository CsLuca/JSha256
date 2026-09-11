# Bitcoin SHA256 Benchmark GUI (V1..V15 + G1..G5)

This project provides a single Windows executable in C++ that:

- detects supported SIMD instruction sets on the host CPU,
- runs benchmark versions V1 to V5,
- shows a graphical scoreboard with per-version metrics.

## Implemented versions

- `V1`: Midstate + generic schedule for first SHA second block + generic second SHA.
- `V2`: Midstate + specialized message schedule for first SHA second block.
- `V3`: V2 + fully unrolled 64 rounds for first SHA second block.
- `V4`: V3 + specialized second SHA (32-byte fixed input block).
- `V5`: Multi-lane benchmark mode over V4.
  - On `AVX2` builds (`/arch:AVX2` or `-mavx2`), it uses a real 8-lane AVX2 backend for the first SHA block in the V4 path.
  - On other tiers it uses lane-model fallback.

## SIMD detection

At startup, benchmark run detects and reports:

- `SSE2`
- `SSE4.1`
- `AVX`
- `AVX2`
- `AVX512F`
- `SHA-NI`

Detection uses CPUID plus OS state checks (`XGETBV`) for AVX/AVX-512 enablement.

## Build (Visual Studio Developer Command Prompt)

```bat
cl /std:c++17 /O2 /EHsc /arch:AVX2 btc_sha_bench_gui.cpp user32.lib gdi32.lib
```

## Build (MinGW-w64 g++)

```bat
g++ -std=c++17 -O3 -mavx2 -municode btc_sha_bench_gui.cpp -o btc_sha_bench_gui.exe -lgdi32 -luser32
```

If your toolchain does not need `-municode`, remove it.

## Run

Start `btc_sha_bench_gui.exe`, then click `Run Benchmark V1..V15 + G1..G5`.

Headless self-test mode (for CI):

```bat
btc_sha_bench_gui.exe --selftest
```

The GUI prints:

- detected SIMD features,
- detected GPU section (phase-0 scaffold),
- correctness report (known nonces and backend consistency checks),
- ranked table columns: `Engine`, `SIMD`, `Ver`, `Backend`, `Lanes`, `Hash/s`, `Std H/s`, `Cycles/hash`, `Std Cyc`.
- text bar chart for top throughput rows.

After each run, results are also exported in:

- `benchmark-output/results-<timestamp>.csv`
- `benchmark-output/results-<timestamp>.json`

## Notes

- `SHA-NI` is detected and, when compiled with SHA intrinsics support, V4/V5 on SHA-NI tier use a real SHA backend for the second SHA block compression.
- This is a benchmarking/education project, not a production miner.
- For strict cross-version comparability, run on an idle system and repeat several times.
