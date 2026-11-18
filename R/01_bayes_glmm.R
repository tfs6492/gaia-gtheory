# =============================================================================
# 01_bayes_glmm.R  --  Bayesian mixed-effects logistic models of run success
# -----------------------------------------------------------------------------
# Outcome: `correct` (0/1) for each of the 4240 runs.
# Fixed:   effort * scaffold (the two crossed design facets) + annotator horizon.
# Random:  task intercept (+ task-specific facet slopes in the slopes model).
#
# Why Bayesian rather than glmer: 54 of 212 design cells are all-0 or all-1,
# which produces (quasi-)separation in ML logistic regression. Weakly
# informative priors with partial pooling keep those cells finite.
#
# Models
#   m_base      correct ~ effort*scaffold + horizon_z + (1 | task)
#   m_slopes    ... + (effort + scaffold | task)          task x facet heterogeneity
#   m_type      m_base + task_type                        coarse domain tag
#   m_base_nto  m_base on runs WITHOUT the 240 s timeouts (sensitivity: the
#               timeout was a harness artifact concentrated in high/code)
#
# Outputs (outputs/01_bayes_glmm/): fixed-effect summaries, LOO comparison,
# latent ICC, and population-level predicted success per condition.
# Fits (.rds) are cached and git-ignored. brms refits automatically when the
# formula or data change, but NOT when sampler settings (iter, warmup, ...)
# change: delete outputs/01_bayes_glmm/*.rds after editing those.
# Packages: brms (+ cmdstanr backend), posterior, tidyverse
# =============================================================================

source("R/00_config.R")
library(brms)
library(posterior)
options(brms.backend = "cmdstanr", mc.cores = 4)

d <- load_runs()

priors <- c(
  prior(normal(0, 2.5),       class = "b"),
  prior(student_t(3, 0, 2.5), class = "sd"),
  prior(normal(0, 5),         class = "Intercept")
)

fit_cached <- function(name, formula, data) {
  brm(formula, data = data, family = bernoulli("logit"), prior = priors,
      # 2000 iterations left the task-intercept SD with R-hat 1.04 / bulk ESS ~130
      # in the base model; 6000 (2000 warmup) gives 16000 post-warmup draws.
      chains = 4, iter = 6000, warmup = 2000, seed = 1234,
      control = list(adapt_delta = 0.95),
      file = out_path("01_bayes_glmm", name), file_refit = "on_change")
}

m_base     <- fit_cached("m_base",
                         correct ~ effort * scaffold + horizon_steps_z + (1 | task_id), d)
m_slopes   <- fit_cached("m_slopes",
                         correct ~ effort * scaffold + horizon_steps_z +
                           (effort + scaffold | task_id), d)
m_type     <- fit_cached("m_type",
                         correct ~ effort * scaffold + horizon_steps_z + task_type +
                           (1 | task_id), d)
m_base_nto <- fit_cached("m_base_nto",
                         correct ~ effort * scaffold + horizon_steps_z + (1 | task_id),
                         filter(d, timed_out == 0))

models <- list(base = m_base, slopes = m_slopes, type = m_type,
               base_timeouts_missing = m_base_nto)

# --- convergence -----------------------------------------------------------------
diag <- imap_dfr(models, \(m, nm) tibble(
  model = nm,
  max_rhat = max(rhat(m), na.rm = TRUE),
  min_bulk_ess = min(summarise_draws(m, ess_bulk)$ess_bulk, na.rm = TRUE),
  n_divergent = sum(nuts_params(m) |> filter(Parameter == "divergent__") |> pull(Value))
))
print(diag)
write_csv(diag, out_path("01_bayes_glmm", "convergence.csv"))

# --- fixed effects (log-odds) ---------------------------------------------------------
fixed <- imap_dfr(models, \(m, nm)
  as_tibble(fixef(m), rownames = "term") |> mutate(model = nm, .before = 1))
print(fixed, n = Inf)
write_csv(fixed, out_path("01_bayes_glmm", "fixed_effects.csv"))

# --- LOO comparison (same data only: excludes the timeout sensitivity fit) -----------
loo_tab <- loo_compare(loo(m_base), loo(m_slopes), loo(m_type))
print(loo_tab)
write_csv(as_tibble(loo_tab, rownames = "model"), out_path("01_bayes_glmm", "loo.csv"))

# --- latent-scale ICC of the base model: var_task / (var_task + pi^2/3) -------------
icc <- function(m) {
  s <- as_draws_df(m)$sd_task_id__Intercept
  x <- s^2 / (s^2 + pi^2 / 3)
  tibble(median = median(x), l95 = quantile(x, .025), u95 = quantile(x, .975))
}
icc_tab <- bind_rows(base = icc(m_base), base_timeouts_missing = icc(m_base_nto),
                     .id = "model")
print(icc_tab)
write_csv(icc_tab, out_path("01_bayes_glmm", "latent_icc.csv"))

# --- predicted success per condition for a typical task (random effects = 0) --------
newd <- expand_grid(effort = factor(c("low", "high"), levels = c("low", "high")),
                    scaffold = factor(c("tool", "code"), levels = c("tool", "code")),
                    horizon_steps_z = 0)
cond_pred <- imap_dfr(list(base = m_base, base_timeouts_missing = m_base_nto), \(m, nm) {
  ep <- posterior_epred(m, newdata = newd, re_formula = NA)
  newd |> mutate(model = nm,
                 p_median = apply(ep, 2, median),
                 p_l95 = apply(ep, 2, quantile, .025),
                 p_u95 = apply(ep, 2, quantile, .975))
})
print(cond_pred)
write_csv(cond_pred, out_path("01_bayes_glmm", "condition_predictions.csv"))
message("01_bayes_glmm done.")
