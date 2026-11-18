# =============================================================================
# 03_within_cell.R  --  do process metrics predict WHICH replications succeed?
# -----------------------------------------------------------------------------
# Among MIXED cells (a task x effort x scaffold cell whose completed runs are
# neither all right nor all wrong), does a run's process (manager steps,
# manager tool calls, wall time) predict its outcome, holding the cell fixed?
#
# Fixes relative to the first version:
#  * Predictors are CENTRED WITHIN CELL (run value minus its cell mean) and
#    scaled by the pooled within-cell SD. Globally standardized predictors mix
#    between-cell differences (slow tasks are hard) into the "within" estimate.
#  * Wall time is log10-transformed (heavy right tail).
#  * Discrimination is reported as WITHIN-CELL concordance of the fixed-effect
#    predictor (pairs of runs from the same cell). The earlier in-sample AUC of
#    0.86 used the cell random intercepts, so it mostly measured how cells
#    differ in base rate, not what process metrics add.
#  * The within-cell concordance is also cross-validated (10 folds of cells).
#  * A model without wall time is reported too. Wall time is partly a
#    consequence of failing (slow runs time out or flounder), not just a cause.
#
# Crashed runs (incl. timeouts) carry no process metrics and are excluded.
# Packages: lme4, tidyverse
# =============================================================================

source("R/00_config.R")
library(lme4)

d <- load_runs() |>
  filter(crashed == 0, !is.na(n_steps), !is.na(n_tool_calls), !is.na(wall_time_s))

md <- d |>
  group_by(cell) |>
  filter(any(correct == 1) & any(correct == 0)) |>
  mutate(across(c(n_steps, n_tool_calls), \(x) x - mean(x), .names = "w_{.col}"),
         w_log_wall = log10(wall_time_s) - mean(log10(wall_time_s))) |>
  ungroup() |>
  mutate(across(starts_with("w_"), \(x) x / sd(x)), cell = droplevels(cell))

cat(sprintf("Mixed cells: %d | completed runs in them: %d\n",
            n_distinct(md$cell), nrow(md)))

ctrl <- glmerControl(optimizer = "bobyqa")
f_null <- correct ~ 1 + (1 | cell)
f_full <- correct ~ w_n_steps + w_n_tool_calls + w_log_wall + (1 | cell)
f_nowall <- correct ~ w_n_steps + w_n_tool_calls + (1 | cell)

m_null   <- glmer(f_null,   data = md, family = binomial, control = ctrl)
m_full   <- glmer(f_full,   data = md, family = binomial, control = ctrl)
m_nowall <- glmer(f_nowall, data = md, family = binomial, control = ctrl)

print(summary(m_full))
lrt <- bind_rows(
  as_tibble(anova(m_null, m_nowall))[2, ] |> mutate(comparison = "null -> steps + tool calls"),
  as_tibble(anova(m_nowall, m_full))[2, ] |> mutate(comparison = "+ log wall time")
) |> select(comparison, Chisq, Df, p = `Pr(>Chisq)`)
print(lrt)

# --- within-cell concordance -------------------------------------------------------
# Share of (success, failure) pairs from the SAME cell in which the success has
# the higher fixed-effect linear predictor. 0.5 = no within-cell signal.
within_c <- function(eta, y, cell) {
  num <- den <- 0
  for (g in split(seq_along(y), cell)) {
    pos <- eta[g][y[g] == 1]; neg <- eta[g][y[g] == 0]
    if (!length(pos) || !length(neg)) next
    cmp <- outer(pos, neg, "-")
    num <- num + sum(cmp > 0) + 0.5 * sum(cmp == 0)
    den <- den + length(cmp)
  }
  num / den
}
eta_fixed <- function(m, newdata) {
  X <- model.matrix(delete.response(terms(lme4::nobars(formula(m)))), newdata)
  drop(X %*% fixef(m))
}

# in-sample
conc_in <- c(
  full   = within_c(eta_fixed(m_full, md),   md$correct, md$cell),
  nowall = within_c(eta_fixed(m_nowall, md), md$correct, md$cell)
)
# 10-fold CV over cells
set.seed(2026)
fold_of <- setNames(sample(rep(1:10, length.out = nlevels(md$cell))), levels(md$cell))
md$fold <- fold_of[as.character(md$cell)]
cv_eta <- map(c(full = "full", nowall = "nowall"), \(which) {
  f <- if (which == "full") f_full else f_nowall
  eta <- numeric(nrow(md))
  for (k in 1:10) {
    m <- glmer(f, data = filter(md, fold != k), family = binomial, control = ctrl)
    eta[md$fold == k] <- eta_fixed(m, filter(md, fold == k))
  }
  eta
})
conc_cv <- map_dbl(cv_eta, \(e) within_c(e, md$correct, md$cell))

# for reference: the old, inflated in-sample AUC that includes cell intercepts
auc_conditional <- if (requireNamespace("pROC", quietly = TRUE))
  as.numeric(pROC::auc(md$correct, fitted(m_full), quiet = TRUE)) else NA_real_

disc <- tibble(
  metric = c("within-cell concordance, in-sample (full)",
             "within-cell concordance, in-sample (no wall time)",
             "within-cell concordance, 10-fold CV (full)",
             "within-cell concordance, 10-fold CV (no wall time)",
             "pooled AUC incl. cell intercepts (old, inflated)"),
  value = c(conc_in, conc_cv, auc_conditional)
)
print(disc)

coefs <- map_dfr(list(full = m_full, no_wall_time = m_nowall), .id = "model", \(m) {
  s <- summary(m)$coefficients
  tibble(term = rownames(s), estimate = s[, 1], se = s[, 2], p = s[, 4])
})
write_csv(coefs, out_path("03_within_cell", "coefficients.csv"))
write_csv(lrt,   out_path("03_within_cell", "lrt.csv"))
write_csv(disc,  out_path("03_within_cell", "discrimination.csv"))
message("03_within_cell done.")
