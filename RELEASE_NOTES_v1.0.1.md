# JSha256 v1.0.1

Release focused on further CPU tuning and roadmap expansion.

## Highlights

- Added `V15` CPU benchmark path:
  - multi-threaded
  - thread-affinity pinned workers
  - AVX2 dual-batch processing
  - SHA-NI tail integration
- Updated app/version labeling to `v1.0.1`.
- Updated GUI range to `V1..V15`.
- Added `ROADMAP_GPU.md` with staged `G1..G6` plan.

## Validation

Local Windows validation completed:

- Build: PASS
- `--selftest`: PASS

## Included Asset

- `btc_sha_bench_gui.exe`

## Notes

- This project is for benchmarking/education and is not a production miner.
- `.exe` is intentionally versioned in branch.
