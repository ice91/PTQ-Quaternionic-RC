# P1C isolated covariance reference

STATISTICAL_SPECIFICATION=ACCEPTABLE_WITH_DECLARED_LIMITATIONS; REFERENCE_IMPLEMENTATION=PASS; REFIT_READINESS=READY_FOR_PRODUCTION_PATCH_REVIEW (review only, not fit launch).

STATISTICAL_SPECIFICATION.json / PREDECLARED_TEST_CONTRACT.json were frozen before tests. reference.py independently implements mean/law equations, full D/i mean derivatives, B0 normalized Gaussian surrogate, B0/B1 explicit nuisance integrals and matched-truncation linear conditional integral. It does not import production modules or optimizers.

Actual environment is local_tests_v1/environment.json and environment_freeze.txt. Commands/console paths are in EXECUTION_LOG.json. Run order is test_reference.py -> run_effects.py -> verify_outputs.py; every output path refuses overwrite. The complete original run is preserved. To rerun without altering it, copy only source/spec/input-manifest files to a NEW sibling run-id directory under results/frontiers_revision_2026_10 (same depth), then run with PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/rev26-p1a-env/bin/python. Prior physical-source absolute paths remain the authority; do not regenerate prior bundles. If the temporary environment is unavailable, build a separate temporary environment from the frozen requirements and document it; no production install.

11 local hard checks passed before637 full fixed-point evaluations; original likelihood components/C are authenticated P1A inputs. All seven parameterizations retain their k/domains. Native mapped-point checks and all records are saved. No corrected MLE or AIC/BIC ranking was produced.

nuisance_v1 retains all585 integral rows and195 convergence cases, T3/4/5 and64/128/256 quadrature orders; B0/B1 floor and same-prior linear integrand are distinct. Edge-on and floor-coordinate limitations preclude blanket all-sample nuisance-approximation validation.

Reviewer primary passages and PDF availability/hash are recorded as new P1C sources. Prior P1B reports are unchanged. Complete assets are indexed by docs/ASSET_MANIFEST.json; each root has SHA256SUMS, and SOURCE_HASHES_BEFORE/SOURCE_PRESERVATION provide input integrity.
