# Changelog

All notable changes to this project are documented in this file.

## [1.0.0] - 2026-09-10

### Added

- Win32 GUI benchmark app for Bitcoin-style double-SHA256.
- Optimization path progression from `V1` to `V14`.
- SIMD detection (`SSE2`, `SSE4.1`, `AVX`, `AVX2`, `AVX512F`, `SHA-NI`).
- CPU info block in GUI output (vendor, model, arch, logical cores).
- Built-in correctness validation with known nonce vectors and backend consistency checks.
- Repeat-run benchmark statistics with warmup (`Std H/s`, `Std Cyc`, min/max).
- Automatic export to CSV/JSON under `benchmark-output/`.
- Text bar chart for top-throughput rows.
- Headless `--selftest` mode for automation.
- GitHub Actions workflow for Windows build, self-test, and artifact upload.

### Notes

- This project is intended for benchmarking/education and is not a production miner.

## [1.0.1] - 2026-09-11

### Added

- `V15` CPU benchmark path with thread-affinity pinning plus AVX2 dual-batch and SHA-NI tail.
- GUI range updated to `V1..V15` and title/version updated to `v1.0.1`.

### Validation

- Windows build: PASS
- `--selftest`: PASS

## [1.1.0] - 2026-09-11

### Added

- GPU phase-1 entry row `G1` in benchmark output when CUDA driver is detected.
- GPU phase-1 row `G2` (midstate host + block #2 model) in benchmark output when CUDA driver is detected.
- GPU phase-1 row `G3` (schedule-specialized model) in benchmark output when CUDA driver is detected.
- GPU device detection improved to check CUDA driver presence (`nvcuda.dll`).
- GUI labels updated to `V1..V15 + G1/G2/G3`.

### Validation

- Windows build: PASS
- `--selftest`: PASS
