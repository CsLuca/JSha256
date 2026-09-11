# GPU Roadmap (CUDA/OpenCL)

This roadmap extends JSha256 from CPU-centric versions (`V1..V15`) to GPU-centric versions (`G1..G6`).

## Guiding Principle

- `V` versions optimize CPU execution.
- `G` versions optimize GPU kernels and host-device orchestration.

## Planned GPU Versions

### G1 - Naive Double-SHA256 Kernel

- Single kernel computes `SHA256(SHA256(header80))` per nonce.
- Baseline throughput and correctness reference.

### G2 - Midstate Host Precompute

- Precompute first block midstate on host.
- Kernel processes only block #2 + second SHA path.

### G3 - Specialized Schedule in Kernel

- Specialize `W` schedule for fixed/predictable words.
- Reduce per-thread instruction cost.

### G4 - Massive Nonce Batching

- Increase nonce batches per launch.
- Reduce launch overhead and improve occupancy.

### G5 - Multi-Stream Overlap

- Use multiple streams to overlap transfers and compute.
- Pipeline: H2D copy, kernel execution, D2H copy.

### G6 - Device Tuning

- Tune block size, register pressure, occupancy.
- Per-device presets and auto-tuning option.

## Stack Options

### NVIDIA Path (Recommended)

- CUDA C++ kernels
- Nsight Compute/Systems for tuning

### Cross-Vendor Path

- OpenCL kernels
- Portable but usually harder to tune for peak throughput

## GUI Integration Plan

- Keep separate CPU and GPU tables.
- Shared metrics:
  - `Hash/s`
  - latency per hash (`Cycles/hash` for CPU, `ns/hash` for GPU)
- Optional power metrics (`J/hash`) when telemetry is available.
- Add CPU vs GPU percentage comparison block.

## Milestones

1. Implement `G1` + correctness checks against CPU V4.
2. Add `G2` and report speedup vs `G1`.
3. Implement `G3-G4` and stabilize benchmark export format.
4. Add `G5-G6` tuning and expose presets in GUI.

## Current Implementation Status

- `G1..G6` are available as integrated scaffold rows in main GUI benchmark flow.
- External CUDA runner hook added (`gpu_cuda_bench.exe`) for real `G1..G6` ingestion when available.
