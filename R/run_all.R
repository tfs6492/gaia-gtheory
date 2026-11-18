# =============================================================================
# run_all.R  --  reproduce every table and figure from data/gaia_runs_public.csv
# -----------------------------------------------------------------------------
# From the repo root:   Rscript R/run_all.R
# 00_prepare_data.R is skipped automatically unless the private run file exists
# (it rebuilds the public CSV from the GAIA-containing private data).
# 01 fits four brms models (cached in outputs/01_bayes_glmm/*.rds, roughly
# 30-60 min on first run). 06 runs a 500-refit parametric bootstrap (cached).
# =============================================================================

source("R/00_config.R")

steps <- c("00_prepare_data", "01_bayes_glmm", "02_cell_models", "03_within_cell",
           "04_answer_nearmiss", "05_answer_dispersion", "06_gstudy",
           "07_sensitivity", "08_figures")

for (s in steps) {
  if (s == "00_prepare_data" && !file.exists(PRIVATE_RUNS)) {
    message("== skipping ", s, " (no private data; using ", PUBLIC_RUNS, ")")
    next
  }
  message("\n== ", s, " ==")
  source(file.path("R", paste0(s, ".R")), echo = FALSE, local = new.env())
}
