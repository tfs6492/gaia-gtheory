# =============================================================================
# 07_sensitivity.R  --  how much do the headline results depend on artifacts?
# -----------------------------------------------------------------------------
# Three problems with the collected data (see README > Limitations):
#   * timeouts      : the 240 s timeout did not abandon runs. 177 runs (163 in
#                     high/code) were scored 0 although the agent kept working.
#   * attachments   : 11 tasks need a file the harness never supplied.
#   * leakage       : some trajectories show the agent finding GAIA material or
#                     published answer keys online.
# For each analysis subset defined in R/00_config.R this script re-estimates:
#   - success per condition (raw)
#   - a frequentist GLMM:  correct ~ effort*scaffold + horizon_z + (1 | task)
#     (same structure as the Bayesian base model in 01, fit with glmer for speed)
#   - the G-study variance partition (linear model) and D-study dependability
# Packages: lme4, tidyverse
# =============================================================================

source("R/00_config.R")
library(lme4)

d_all <- load_runs()

run_subset <- function(name) {
  d <- SUBSETS[[name]](d_all) |>
    mutate(task_id = droplevels(task_id), cell = droplevels(cell),
           horizon_steps_z = as.numeric(scale(horizon_steps)),
           task_effort   = interaction(task_id, effort,   drop = TRUE),
           task_scaffold = interaction(task_id, scaffold, drop = TRUE))

  acc <- d |> group_by(condition) |>
    summarise(n = n(), success = mean(correct), .groups = "drop") |>
    mutate(subset = name)

  m <- glmer(correct ~ effort * scaffold + horizon_steps_z + (1 | task_id),
             data = d, family = binomial, control = glmerControl(optimizer = "bobyqa"))
  s <- summary(m)$coefficients
  glmm <- tibble(subset = name, term = rownames(s), estimate = s[, 1], se = s[, 2],
                 l95 = s[, 1] - 1.96 * s[, 2], u95 = s[, 1] + 1.96 * s[, 2], p = s[, 4])

  g <- lmer(correct ~ effort * scaffold + (1 | task_id) + (1 | task_effort) +
              (1 | task_scaffold) + (1 | cell),
            data = d, control = lmerControl(optimizer = "bobyqa"))
  vc <- as.data.frame(VarCorr(g)); v <- setNames(vc$vcov, vc$grp)
  v <- v[c("task_id", "task_effort", "task_scaffold", "cell", "Residual")]; v[is.na(v)] <- 0
  gst <- tibble(subset = name, n_runs = nrow(d), n_tasks = nlevels(d$task_id),
                success = mean(d$correct),
                pct_task = 100 * v[[1]] / sum(v),
                pct_task_x_facets = 100 * sum(v[2:4]) / sum(v),
                pct_residual = 100 * v[[5]] / sum(v),
                dep_one_condition_r1 = sum(v[1:4]) / (sum(v[1:4]) + v[[5]]),
                dep_one_condition_r20 = sum(v[1:4]) / (sum(v[1:4]) + v[[5]] / 20))
  list(acc = acc, glmm = glmm, gstudy = gst)
}

res <- map(names(SUBSETS), run_subset)
acc    <- map_dfr(res, "acc")
glmm   <- map_dfr(res, "glmm")
gstudy <- map_dfr(res, "gstudy")

print(gstudy |> mutate(across(where(is.double), \(x) round(x, 3))))
print(glmm |> filter(term %in% c("efforthigh", "scaffoldcode", "efforthigh:scaffoldcode")) |>
        mutate(across(where(is.double), \(x) round(x, 3))), n = Inf)
print(acc |> select(-n) |> pivot_wider(names_from = condition, values_from = success))

# leakage / attachment descriptives
leak <- bind_rows(
  d_all |> group_by(group = if_else(leak_flag == 1, "leak-flagged run", "other run")) |>
    summarise(n = n(), success = mean(correct), tasks = n_distinct(task_id), .groups = "drop"),
  d_all |> group_by(group = if_else(has_attachment == 1, "attachment task (file never supplied)",
                                    "no-attachment task")) |>
    summarise(n = n(), success = mean(correct), tasks = n_distinct(task_id), .groups = "drop")
)
print(leak)

write_csv(acc,    out_path("07_sensitivity", "success_by_condition.csv"))
write_csv(glmm,   out_path("07_sensitivity", "glmm_coefficients.csv"))
write_csv(gstudy, out_path("07_sensitivity", "gstudy_summary.csv"))
write_csv(leak,   out_path("07_sensitivity", "leak_attachment_descriptives.csv"))
message("07_sensitivity done.")
