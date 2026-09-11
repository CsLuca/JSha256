# JSha256

Current release: **v1.1.0**

Release status: build and `--selftest` validated locally on Windows (MSYS2 UCRT64).

Release notes draft: `RELEASE_NOTES_v1.0.0.md`

![Platform](https://img.shields.io/badge/platform-Windows-blue)
![Language](https://img.shields.io/badge/language-C%2B%2B17-00599C)
![Build](https://img.shields.io/badge/build-MSYS2%20UCRT64-brightgreen)
![SIMD](https://img.shields.io/badge/SIMD-SSE2%2FSSE4.1%2FAVX%2FAVX2%2FSHA--NI-orange)

Win32 GUI benchmark for Bitcoin-style `double-SHA256` optimization stages `V1..V15` plus GPU `G1..G6` scaffold runs.

It detects CPU SIMD capabilities, runs all versions, and shows a ranked scoreboard with:

- `Hash/s`
- `Cycles/hash`
- backend used (`scalar-core`, `avx2-8lane-real`, `sha-ni-real`, ...)
- built-in correctness checks (known nonces + backend cross-check)
- repeat-run statistics (`Std H/s`, `Std Cyc`) with warmup
- automatic export in `benchmark-output/results-*.csv` and `benchmark-output/results-*.json`
- phase-0 GPU integration scaffold (`Engine` column + GPU device info section)
- phase-1 entry points with `G1..G6` benchmark rows when CUDA driver is detected
- optional external CUDA runner support via `gpu_cuda_bench.exe` for real `G1..G6` rows
- optional external OpenCL runner support via `gpu_opencl_bench.exe` for real `G1..G4` rows

CUDA build example for external runner:

```bash
nvcc -O3 gpu_cuda_bench.cu -o gpu_cuda_bench.exe
```

OpenCL build example for external runner:

```bash
g++ -std=c++17 -O3 gpu_opencl_bench.cpp -o gpu_opencl_bench.exe -lOpenCL
```

When `gpu_cuda_bench.exe` and/or `gpu_opencl_bench.exe` are present next to the GUI executable, parsed GPU rows from available external runners are merged into the benchmark output.

## Project Layout

- `btc_sha_bench_gui.cpp` - GUI + benchmark engine
- `README_btc_sha_bench_gui.md` - technical implementation notes
- `CHANGELOG.md` - release notes
- `ROADMAP_GPU.md` - plan for GPU extension (`G1..G6`)

## Optimization Versions

| Version | Description |
|---|---|
| `V1` | Midstate optimization |
| `V2` | Specialized schedule for first SHA block #2 |
| `V3` | Unrolled rounds for first SHA block #2 |
| `V4` | Specialized second SHA block |
| `V5` | Multi-lane benchmark mode (real AVX2 backend when available) |
| `V6` | Multi-threaded V4 on all logical cores |
| `V7` | SHA-NI full path (block #2 + second SHA with SHA intrinsics) |
| `V8` | AVX2 pipelined dual-batch mode (2x8 lanes in flight) |
| `V9` | Hybrid multi-thread + AVX2 pipelined mode |
| `V10` | Hybrid multi-thread + SHA-NI full mode |
| `V11` | SHA block #2 via SHA-NI + second SHA scalar specialized |
| `V12` | AVX2 batch path with SHA-NI assisted tail path |
| `V13` | Multi-thread AVX2 batch + SHA-NI tail |
| `V14` | Multi-thread AVX2 dual-batch pipeline + SHA-NI tail |
| `V15` | Multi-thread affinity-pinned AVX2 dual-batch + SHA-NI tail |

## SIMD Features Reported

- `SSE2`
- `SSE4.1`
- `AVX`
- `AVX2`
- `AVX512F`
- `SHA-NI`

## Build (MSYS2 UCRT64)

```bash
g++ -std=c++17 -O3 -mavx2 -msha -municode btc_sha_bench_gui.cpp -o btc_sha_bench_gui.exe -lgdi32 -luser32
```

## Run

```bash
./btc_sha_bench_gui.exe
```

Self-test mode (no GUI, exit code based):

```bash
./btc_sha_bench_gui.exe --selftest
```

GPU mode switches:

```bash
./btc_sha_bench_gui.exe --gpu-external-only
./btc_sha_bench_gui.exe --gpu-external-disable
```

Then click `Run Benchmark V1..V15`.

## Example Scoreboard

```text
SIMD       Ver   Backend          Lanes   Hash/s          Cycles/hash
AVX2       V5    avx2-8lane-real  8       1234567.89      210.34
SHA-NI     V4    sha-ni-real      1       845000.12       305.77
SCALAR     V4    scalar-core      1       412000.55       622.11
```

## Screenshot

![JSha256 GUI](docs/screenshot-main-window.png)
