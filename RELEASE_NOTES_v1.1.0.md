# JSha256 v1.1.0

Release introducing first GPU benchmark entry integration.

## Highlights

- Added GPU benchmark row `G1` (phase-1 entry path) in the benchmark pipeline.
- Added CUDA driver detection (`nvcuda.dll`) to enable GPU-row availability checks.
- Updated GUI labels and title to `V1..V15 + G1`.
- Kept CPU pipeline and validation stable (`V1..V15`).

## Validation

- Build: PASS
- `--selftest`: PASS

## Included Asset

- `btc_sha_bench_gui.exe`
