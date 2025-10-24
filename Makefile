.PHONY: data-h data-merge data-all test

data-h:
\tpython -m ptq.data.s4g_h_pipeline build-h \\
\t  --src dataset/geometry/s4g_tablea1.tsv \\
\t  --out dataset/geometry/h_catalog.csv \\
\t  --outliers dataset/geometry/h_z_outliers.csv

data-merge:
\tpython -m ptq.data.s4g_h_pipeline merge-sparc-h \\
\t  --sparc dataset/sparc_tidy.csv \\
\t  --h dataset/geometry/h_catalog.csv \\
\t  --out dataset/geometry/sparc_with_h.csv \\
\t  --alias dataset/geometry/aliases.csv \\
\t  --unmatched dataset/geometry/unmatched_galaxies.csv

data-all: data-h data-merge

test:
\tpytest -q