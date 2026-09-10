# JSha256 v1.0.0

First stable release of JSha256 benchmark GUI.

## Highlights

- Win32 GUI benchmark for Bitcoin-style `double-SHA256`.
- Optimization pipeline from `V1` to `V14`.
- SIMD detection (`SSE2`, `SSE4.1`, `AVX`, `AVX2`, `AVX512F`, `SHA-NI`).
- CPU info panel (vendor, model, architecture, logical cores).
- Built-in correctness validation with known vectors.
- Repeat-run statistics (warmup + stddev/min/max).
- Export to CSV/JSON in `benchmark-output/`.
- Headless CI mode: `--selftest`.

## Validation

Local Windows validation completed:

- Build: PASS
- `--selftest`: PASS

## Assets

- `btc_sha_bench_gui.exe` (versioned in branch and available in CI artifacts)
