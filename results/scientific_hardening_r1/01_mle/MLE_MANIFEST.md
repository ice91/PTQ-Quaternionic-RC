# MLE_MANIFEST

True full-covariance MLE (R1B). Priors excluded.

## Models
- `baryon`: max_logL=-8826.776935, AIC=17837.554, BIC_NPOINTS=18342.219, k=92
- `mond`: max_logL=-6663.422141, AIC=13512.844, BIC_NPOINTS=14022.995, k=93
- `ptq`: max_logL=-7138.535286, AIC=14463.071, BIC_NPOINTS=14973.221, k=93
- `ptq-nu`: max_logL=-6663.422141, AIC=13512.844, BIC_NPOINTS=14022.995, k=93
- `ptq-screen`: max_logL=-6663.002103, AIC=13514.004, BIC_NPOINTS=14029.640, k=94
- `mond-screen`: max_logL=-6663.002103, AIC=13514.004, BIC_NPOINTS=14029.640, k=94
- `nfw1p`: max_logL=-6508.169660, AIC=13382.339, BIC_NPOINTS=14386.184, k=183

## Equivalence
- `ptq-nu_vs_mond`: abs(delta_logL)=4.676e-08, pass_1e-5=True
- `ptq-screen_vs_mond-screen`: abs(delta_logL)=4.547e-12, pass_1e-5=True
