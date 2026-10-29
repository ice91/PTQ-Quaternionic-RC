# P1D isolated patch and boundary audit

Verdicts: PATCH_PARITY PASS; BOUNDARY_APPROXIMATION MATERIAL_SENSITIVITY_REQUIRES_NEW_SPEC; FLOOR_SENSITIVITY MATERIAL_MODEL_UNCERTAINTY; REFIT_LAUNCH REQUIRES_STATISTICAL_SPECIFICATION_DECISION.

Canonical source remains unchanged. isolated_candidate/src/ptquat contains candidate source; production_candidate.diff is complete six-file patch incl verified experiments/test API consumers. PATCH_CALL_GRAPH maps12 direct source C callers. PREDECLARED_PROTOCOL was frozen before computations.

parity_v1:637 fixed points,364 mappings,all paths/tests. neighborhood_patch_parity_v1:the existing66 fixed perturbations plus actual sourceAST callback checks, no new integrations. boundary_v1:3300 raw integrals,825 attempted grids,22 diagnostic galaxies,all7real edge-on×7models,4geometry controls,66 neighborhood points. All results fixed historical or declared perturbations, not correctedMLE.

IMPORTANT: boundary_v1/neighborhood_effect_changes.csv is a PRESERVED FAILED DERIVED ARTIFACT (NaN due pandas .T column bug); v1 decomposition convergence flags also used incorrect linkage. Raw integrals/convergence numerical values were unaffected. Use postprocessing_v2 ONLY for decomposition/convergence/neighbor/decision summaries. postprocess_v2.py rebuilds from raw without new quadrature or contract change; V1_DERIVED_FAILURE.json records the failure.

Run order: bootstrap.py -> freeze_protocol.py -> build_patch.py -> test_patch.py -> run_boundary.py -> postprocess_v2.py -> verify_neighborhood_patch.py -> write_reports.py -> finalize.py. All output roots/run IDs reject overwrites; scripts are preserved, including failed original postprocessor. For exact reproduction copy only scripts plus candidate/spec/protocol authority to a NEW sibling run-id at same depth, preserving absolute original-input paths; do not copy completed output folders. Do not rerun original prepare/freeze against already frozen files. Use temporary environment in parity_v1/environment_freeze.txt; command prefix PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/rev26-p1a-env/bin/python. No optimization or canonical installation.

Complete logs: parity_execution.log, boundary_execution.log, postprocessing_v2.log, neighborhood_patch_parity.log; execution metadata/authority/spec hashes are machine-readable. INDEPENDENT_PATCH_REVIEW records separate read-only critique and the actual derived failure. UNIFORM_ALTERNATIVE_SPECIFICATION is a proposal, not implemented target; INACTIVE_SEVEN_MODEL_LAUNCH_DESIGN can_run=false.

Allnew assets/hashes are indexed in docs ASSET_MANIFEST; each root has SHA256SUMS. Source preservation validation covers all1866 preexisting files. Never use fixed-point values as optimized maxima or IC rankings.
