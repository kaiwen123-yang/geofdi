# Onboard-proxy benchmark on the NUC13 (Sprint 10 Block N)

Purpose: the timing numbers in the manuscript are currently measured on a laptop CPU
(`Intel Core Ultra 9 275HX`, one core), which is a *proxy* for a robot's onboard computer. Re-running the identical
script on the NUC13 (Ubuntu 22.04, no GPU) gives a second, more representative point. The script writes into the same
CSV schema, keyed by hostname, so the two hosts can be compared row by row without any post-processing.

## Three steps

**1. Clone and set up (no CUDA needed).**
```sh
git clone <this repo> ~/geofdi && cd ~/geofdi
make setup                      # creates the venv and installs the package with dev extras
# the benchmark path needs numpy / scipy / pandas / pyyaml only; if pip tries to pull a CUDA torch wheel, use:
#   .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```
MuJoCo is required only for the `--inekf` stage on a *legged* robot with URDF forward kinematics; the Go2 corpus uses
the robot's own foot positions (`use_provided_feet`), so the benchmark runs without it.

**2. Copy one session across.** From the laptop:
```sh
scp -r /path/to/geofdi-data/data/raw/go2/2026-03-06/by2      nuc:~/geofdi-data/data/raw/go2/2026-03-06/
scp -r /path/to/geofdi-data/data/raw/m1/nominal/m1_walk_20260810_173028  nuc:~/geofdi-data/data/raw/m1/nominal/
```
`by2` is ~150 MB and `m1_walk_20260810_173028` ~5.5 GB; if the M1 session is too large to copy, benchmark the Go2 one
alone and say so — the Go2 row is the one with the InEKF stage.

**3. Run, pinned to one core.**
```sh
export GEOFDI_DATA_ROOT=~/geofdi-data
taskset -c 0 .venv/bin/python scripts/bench_pipeline.py \
    $GEOFDI_DATA_ROOT/data/raw/go2/2026-03-06/by2 --robot go2 --inekf --repeats 5
taskset -c 0 .venv/bin/python scripts/bench_pipeline.py \
    $GEOFDI_DATA_ROOT/data/raw/m1/nominal/m1_walk_20260810_173028 --robot m1 --repeats 5
```
The first Go2 run parses the ~90 MB transcript and caches it as parquet, which takes ~30–40 s; the script warms that
cache before timing, so the reported `load` figure is the steady-state one.

## What comes back
`$GEOFDI_DATA_ROOT/results/bench/<hostname>/` gains `bench_table.csv` (per stage), `bench_summary.csv` (one row per
run) and a markdown table per run. Copy that directory back and it drops straight into the results tree beside the
laptop's. The two numbers the manuscript quotes are the summary's `realtime_factor` (detector work per gait cycle
against the cycle period) and `inekf_sustainable_hz` (against the 250 Hz telemetry rate).

## Laptop reference (2026-08-16, `Intel Core Ultra 9 275HX`, 1 core, BLAS 1 thread, median of 3)

| robot | session | detector ms per cycle/block | budget | real-time factor | InEKF | sustainable |
|---|---|---:|---:|---:|---:|---:|
| Go2 | `by2` | 3.38 ms/cycle | 443.7 ms | **131×** | 0.163 ms/sample | **6137 Hz** vs 249 Hz telemetry |
| M1 | `m1_walk_20260810_173028` | 2.25 ms/block | 1000 ms | **444×** | not run | — |

Per-stage on the Go2: segment 0.07, phase 1.47, element 0.04, permutation test ($M{=}512$) 0.06, e-process $<0.001$,
$H_0'$ 1.75 ms per cycle. **The phase estimate and the $H_0'$ permutation dominate; the flip test itself is nearly
free.** If the NUC comes in an order of magnitude slower, the pipeline still fits inside a gait cycle by a wide margin,
which is the claim the manuscript needs.
