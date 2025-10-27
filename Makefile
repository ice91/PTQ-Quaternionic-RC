.PHONY: data-h data-merge data-all data-h-cli data-h-audit test

# Paths
SPARC    := dataset/sparc_tidy.csv
H_CAT    := dataset/geometry/h_catalog.csv
SPARC_H  := dataset/geometry/sparc_with_h.csv
ALIASES  := dataset/geometry/aliases.csv
UNMATCH  := dataset/geometry/unmatched_galaxies.csv

# --------------------------------------------------------------------
# Preferred route (used in the paper): one-command CLI
# --------------------------------------------------------------------
data-h-cli: $(H_CAT)
	@echo "[OK] Built $(H_CAT) via CLI."

$(H_CAT): $(SPARC)
	ptquat geom s4g-hcat --sparc $(SPARC) --out $(H_CAT) --prefer thin

data-merge: $(SPARC_H)
	@echo "[OK] Built $(SPARC_H) via CLI."

$(SPARC_H): $(H_CAT)
	ptquat geom s4g-join --sparc $(SPARC) --h $(H_CAT) --out $(SPARC_H) \
	    --alias $(ALIASES) --unmatched $(UNMATCH)

# Back-compat aliases: keep old targets but point to preferred route
data-h: data-h-cli

data-all: data-h data-merge

# --------------------------------------------------------------------
# Auditable fallback route (explicit TSV -> python ETL module)
# --------------------------------------------------------------------
data-h-audit:
	python -m ptq.data.s4g_h_pipeline build-h \
	  --src dataset/geometry/s4g_tablea1.tsv \
	  --out $(H_CAT) \
	  --outliers dataset/geometry/h_z_outliers.csv
	python -m ptq.data.s4g_h_pipeline merge-sparc-h \
	  --sparc $(SPARC) \
	  --h $(H_CAT) \
	  --out $(SPARC_H) \
	  --alias $(ALIASES) \
	  --unmatched $(UNMATCH)
	@echo "[OK] Audit route completed."

# --------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------
test:
	pytest -q
