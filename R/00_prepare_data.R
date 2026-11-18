# =============================================================================
# 00_prepare_data.R  --  private run CSV  ->  shareable public run CSV
# -----------------------------------------------------------------------------
# The private CSV (harness/aggregate_to_csv.py output) holds GAIA question text,
# ground-truth answers and model answers. None of that may be republished, so
# this script derives everything the analyses need from those strings and then
# drops them:
#
#   answer_category : how wrong a run's answer is (classification below)
#   answer_id       : salted hash of the normalized model answer. Equal answers
#                     get equal ids (for answer dispersion, 05), but the answer
#                     text cannot be read back. The salt stays in data/private/.
#
# Answer categories (the GAIA scorer is exact-match, so "wrong" is very broad):
#   exact_correct      scored correct by the official GAIA scorer
#   no_answer          crash, timeout or empty answer
#   units_scale_error  both numeric, ratio is a clean power of ten (17000 vs 17)
#   numeric_near_miss  both numeric, relative error <= 5%, and at least one of
#                      them non-integer (a rounding/precision slip). Two different
#                      integers (a count of 519 vs 531, a year of 2015 vs 2016)
#                      are simply wrong, not "near"
#   string_near_miss   not both numeric, normalized edit distance <= 0.20
#   wrong              everything else
#
# A stratified sample of classified misses is written to a private audit sheet
# (outputs/04_answer_nearmiss/near_miss_audit_PRIVATE.csv, git-ignored) for
# manual checking.
#
# Only needed when you have the private data. Everyone else can start from
# data/gaia_runs_public.csv.
# Packages: tidyverse, stringdist, digest
# =============================================================================

source("R/00_config.R")
library(stringdist)
library(digest)

if (!file.exists(PRIVATE_RUNS)) stop("Missing ", PRIVATE_RUNS, " (see README > Reproducing).")

raw <- read_csv(PRIVATE_RUNS, show_col_types = FALSE,
                col_types = cols(true_answer = col_character(),
                                 model_answer = col_character(),
                                 .default = col_guess()))

# --- answer helpers -------------------------------------------------------------
clean_number <- function(x) {
  x <- str_replace_all(x, "[\\$%,]", "")
  suppressWarnings(as.numeric(str_trim(x)))
}
norm_str <- function(x) {       # GAIA-style: lowercase, no punctuation/whitespace
  x <- str_to_lower(coalesce(as.character(x), ""))
  x <- str_replace_all(x, "[[:punct:]]", "")
  str_replace_all(x, "\\s+", "")
}

REL_ERR_TOL <- 0.05
EDIT_TOL    <- 0.20

classified <- raw |>
  mutate(
    m_raw    = str_trim(coalesce(model_answer, "")),
    t_raw    = str_trim(coalesce(true_answer, "")),
    m_num    = clean_number(m_raw),
    t_num    = clean_number(t_raw),
    both_num = !is.na(m_num) & !is.na(t_num) & is.finite(m_num) & is.finite(t_num),
    year_like_truth = both_num & t_num == round(t_num) & t_num >= 1000 & t_num <= 2100,
    both_integer    = both_num & t_num == round(t_num) & m_num == round(m_num),
    rel_err  = if_else(both_num & t_num != 0, abs(m_num - t_num) / abs(t_num), NA_real_),
    log10ratio = if_else(both_num & t_num != 0 & m_num != 0,
                         log10(abs(m_num / t_num)), NA_real_),
    is_scale = !is.na(log10ratio) &
               abs(round(log10ratio) - log10ratio) < 0.02 & round(log10ratio) != 0,
    norm_edit = stringdist(norm_str(m_raw), norm_str(t_raw), method = "lv") /
                pmax(nchar(norm_str(m_raw)), nchar(norm_str(t_raw)), 1),
    answer_category = case_when(
      correct == 1                                    ~ "exact_correct",
      crashed == 1 | m_raw == ""                      ~ "no_answer",
      both_num & is_scale & !year_like_truth          ~ "units_scale_error",
      both_num & !both_integer & !year_like_truth & !is.na(rel_err) &
        rel_err <= REL_ERR_TOL                        ~ "numeric_near_miss",
      !both_num & norm_edit <= EDIT_TOL               ~ "string_near_miss",
      TRUE                                            ~ "wrong"
    )
  )

# --- salted answer ids -----------------------------------------------------------
salt_file <- "data/private/answer_salt.txt"
if (!file.exists(salt_file)) {
  writeLines(digest(paste(Sys.time(), runif(1)), algo = "sha256"), salt_file)
}
salt <- readLines(salt_file, n = 1)
hash_answer <- function(a) {
  vapply(a, \(x) if (x == "") NA_character_
                 else substr(digest(paste0(salt, x), algo = "sha256", serialize = FALSE), 1, 12),
         character(1), USE.NAMES = FALSE)
}

public <- classified |>
  mutate(answer_id = hash_answer(norm_str(m_raw))) |>
  transmute(
    task_id, run_index, model, effort, scaffold = prompt_variant, temperature,
    search_backend, task_type, level, horizon_steps, horizon_tools, has_attachment,
    correct, answer_category, answer_id,
    wall_time_s, n_steps, n_tool_calls, n_step_errors,
    empty_answer, no_tool_use, tool_or_exec_error, no_final_answer, harness_exception,
    crashed, timed_out, crash_type, leak_flag, timestamp_utc
  )

stopifnot(nrow(public) == nrow(raw),
          !any(c("question", "true_answer", "model_answer", "crash_msg") %in% names(public)))
write_csv(public, PUBLIC_RUNS)
message("Wrote ", PUBLIC_RUNS, " (", nrow(public), " runs, no GAIA text).")

# --- private audit sheet (git-ignored) ---------------------------------------------
set.seed(2026)
audit <- classified |>
  filter(answer_category %in% c("units_scale_error", "numeric_near_miss",
                                "string_near_miss", "wrong")) |>
  group_by(answer_category) |>
  slice_sample(n = 15) |>
  ungroup() |>
  select(answer_category, task_id, effort, scaffold = prompt_variant, run_index,
         true_answer, model_answer, rel_err, norm_edit) |>
  mutate(audit_ok = NA)
write_csv(audit, out_path("04_answer_nearmiss", "near_miss_audit_PRIVATE.csv"))
message("Wrote private audit sample (", nrow(audit), " rows) for manual checking.")
print(count(public, answer_category, sort = TRUE))
