# JSha256 v1.1.0

Release introducing first GPU benchmark entry integrations.

## Highlights

- Added GPU benchmark rows `G1`, `G2`, and `G3` (phase-1 entry paths) in the benchmark pipeline.
- Added CUDA driver detection (`nvcuda.dll`) to enable GPU-row availability checks.
- Updated GUI labels and title to `V1..V15 + G1/G2/G3`.
- Kept CPU pipeline and validation stable (`V1..V15`).

## Validation

- Build: PASS
- `--selftest`: PASS

## Included Asset

- `btc_sha_bench_gui.exe`
