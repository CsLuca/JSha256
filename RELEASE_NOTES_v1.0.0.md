# JSha256 v1.0.0

First stable release of JSha256 benchmark GUI.

## Highlights

- Win32 GUI benchmark for Bitcoin-style `double-SHA256`.
- Optimization pipeline from `V1` to `V14`.
- SIMD detection: `SSE2`, `SSE4.1`, `AVX`, `AVX2`, `AVX512F`, `SHA-NI`.
- CPU info panel in GUI (vendor, model, architecture, logical cores).
- Built-in correctness validation with known nonce vectors.
- Repeat-run benchmark statistics:
  - warmup
  - multi-run aggregation
  - standard deviation (`Std H/s`, `Std Cyc`)
  - min/max throughput
- Automatic export after each run:
  - `benchmark-output/results-<timestamp>.csv`
  - `benchmark-output/results-<timestamp>.json`
- Top-throughput text bar chart in GUI output.
- Headless CI mode:
  - `btc_sha_bench_gui.exe --selftest`
- GitHub Actions workflow (`windows-build`) with:
  - Windows build
  - self-test
  - artifact upload

## Included Asset

- `btc_sha_bench_gui.exe`

## Validation

Local Windows validation completed:

- Build: PASS
- `--selftest`: PASS

## Notes

- Project is intended for benchmarking/education and is not a production miner.
- `.exe` is intentionally versioned in branch.
