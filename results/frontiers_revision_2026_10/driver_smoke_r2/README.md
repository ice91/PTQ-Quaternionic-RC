# REV26-P1F-R2 evidence

ENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE.

Only `execution_v1/` contains actual optimizer execution: one original P1A start each for ptq-screen and nfw1p, one accepted iteration each. `controls_v1/` and `controls_v2/` contain synthetic stubs; their callback messages and vectors are not real optimizer output.

The new CLI is `candidate/scripts/run_repaired_gradient_smoke.py`. The old P1F/R1 driver remains disabled. All scientific and dependency-gradient module bytes equal R1. Exact original theta, input/data/code hashes, frozen protocol, dependency receipt and environment are supplied. Full R1 parity is reused by exact identity; every fresh initial/terminal numerical check is charged in the single actual smoke ledger.

Executed command, with single-thread BLAS and no bytecode:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/rev26-p1a-env/bin/python -u candidate/scripts/run_repaired_gradient_smoke.py --bundle /Users/rocky/ptq-mond/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/driver_smoke_r2
```

The numerical root and `execution_v1` already exist and must not be overwritten or rerun. A separately authorized replay requires a fresh root, regenerated absolute-path tested/input manifests and the same frozen scientific code/protocol. This is evidence, not a canonical installer.

`analysis_v1/VALIDATED_RESULTS.json` and `DECISION.json` give engineering verdicts. Per-request gradient/objective files, all trial and accepted/terminal vectors, independent precision audits, native FD endpoints, work ledger and bounded warning summaries are complete. Both SciPy calls stop at maxiter=1; neither establishes stationarity. `INACTIVE_P1G_PLAN.json` is a proposal, not executed fitting.
