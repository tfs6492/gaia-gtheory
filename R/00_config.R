# =============================================================================
# 00_config.R  --  shared paths, data loading, analysis subsets and plot theme
# -----------------------------------------------------------------------------
# Sourced by every other script. Run all scripts with the working directory set
# to the REPO ROOT (e.g. open the folder in RStudio, or `Rscript R/run_all.R`).
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
})

if (!file.exists("R/00_config.R")) {
  stop("Set the working directory to the repository root before sourcing R/ scripts.")
}

# --- paths --------------------------------------------------------------------
PUBLIC_RUNS  <- "data/gaia_runs_public.csv"          # shareable, committed
PRIVATE_RUNS <- "data/private/gaia_runs.csv"         # contains GAIA text, git-ignored
OUT_DIR      <- "outputs"
FIG_DIR      <- "figs"

out_path <- function(sub, file) {
  d <- file.path(OUT_DIR, sub)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  file.path(d, file)
}

# --- data ---------------------------------------------------------------------
# One row per agent run. Facets are returned as factors with fixed level order,
# plus a `cell` id for the (task x effort x scaffold) design cell.
load_runs <- function(path = PUBLIC_RUNS) {
  if (!file.exists(path)) stop("Missing ", path, ". Run R/00_prepare_data.R first.")
  read_csv(path, show_col_types = FALSE) |>
    mutate(
      task_id   = factor(task_id),
      effort    = factor(effort,   levels = c("low", "high")),
      scaffold  = factor(scaffold, levels = c("tool", "code")),
      condition = factor(paste(effort, scaffold, sep = " / "),
                         levels = c("low / tool", "low / code",
                                    "high / tool", "high / code")),
      cell      = interaction(task_id, effort, scaffold, drop = TRUE, sep = ":"),
      horizon_steps_z = as.numeric(scale(horizon_steps))
    )
}

# --- sensitivity subsets --------------------------------------------------------
# Each element maps the full run table to the runs kept in that analysis.
#   timeouts_missing : the 240 s timeout was a harness artifact (see README), so
#                      treat those runs as missing instead of as failures.
#   no_attachments   : drop the 11 tasks whose attached file was never supplied.
#   no_leak_flagged  : drop runs whose trajectory mentions GAIA / an answer key.
#                      This conditions on a post-treatment variable correlated
#                      with success, so read it as a bound, not an estimate.
SUBSETS <- list(
  all_runs         = function(d) d,
  timeouts_missing = function(d) filter(d, timed_out == 0),
  no_attachments   = function(d) filter(d, has_attachment == 0),
  no_leak_flagged  = function(d) filter(d, leak_flag == 0),
  all_three        = function(d) filter(d, timed_out == 0, has_attachment == 0, leak_flag == 0)
)
SUBSET_LABELS <- c(
  all_runs         = "All runs (as collected)",
  timeouts_missing = "Timeouts treated as missing",
  no_attachments   = "Attachment tasks removed",
  no_leak_flagged  = "Leak-flagged runs removed",
  all_three        = "All three restrictions"
)

# --- plotting -------------------------------------------------------------------
# ggplot2 + Okabe-Ito colours from {see}. Colour-blind safe.
suppressPackageStartupMessages(library(see))

theme_gaia <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title         = element_text(face = "bold", size = rel(1.1)),
      plot.subtitle      = element_text(colour = "grey35", size = rel(0.85)),
      plot.caption       = element_text(colour = "grey45", size = rel(0.75), hjust = 0),
      plot.title.position = "plot",
      plot.caption.position = "plot",
      panel.grid.minor   = element_blank(),
      strip.text         = element_text(face = "bold"),
      legend.position    = "top",
      legend.justification = "left"
    )
}

# Fixed condition -> colour mapping so every figure agrees.
# Okabe-Ito: orange, sky blue, bluish green, vermillion.
COND_COLOURS <- setNames(unname(okabeito_colors(c("orange", "sky blue",
                                                  "bluish green", "vermillion"))),
                         c("low / tool", "low / code", "high / tool", "high / code"))

save_fig <- function(plot, name, width = 8, height = 5, dpi = 200) {
  dir.create(FIG_DIR, showWarnings = FALSE)
  ggsave(file.path(FIG_DIR, paste0(name, ".png")), plot,
         width = width, height = height, dpi = dpi, bg = "white")
}
