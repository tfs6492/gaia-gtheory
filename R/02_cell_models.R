# =============================================================================
# 02_cell_models.R  --  how consistently is each task solved under each condition?
# -----------------------------------------------------------------------------
# Unit: the design CELL (task x effort x scaffold; 53 x 2 x 2 = 212 cells, each
# with 20 replications). Every task contributes FOUR cells, so all models here
# include a task random intercept. (The earlier version treated the 212 cells as
# independent observations, which overstates the evidence.)
#
#  (a) Cumulative-link mixed model of the consistency TIER
#      (never < mixed < always correct within the cell).
#  (b) Beta-binomial GLMM of the cell's success COUNT (n_correct of n). The
#      beta-binomial absorbs extra-binomial variation between cells, and unlike
#      a beta regression on the success rate it needs no squeeze for the cells
#      at exactly 0% or 100%.
#
# Packages: ordinal, glmmTMB, tidyverse
# =============================================================================

source("R/00_config.R")
library(ordinal)
library(glmmTMB)

d <- load_runs()

cells <- d |>
  group_by(task_id, cell, effort, scaffold, condition, horizon_steps, has_attachment) |>
  summarise(n = n(), n_correct = sum(correct), .groups = "drop") |>
  mutate(
    success_rate = n_correct / n,
    tier = ordered(case_when(n_correct == 0 ~ "never",
                             n_correct == n ~ "always",
                             TRUE           ~ "mixed"),
                   levels = c("never", "mixed", "always")),
    horizon_z = as.numeric(scale(horizon_steps))
  )
write_csv(select(cells, -cell), out_path("02_cell_models", "cell_level.csv"))

cat("Cells per tier:\n"); print(count(cells, tier))
cat("\nTier by condition:\n"); print(count(cells, condition, tier) |>
                                      pivot_wider(names_from = tier, values_from = n))

# ---- (a) ordinal tier model ------------------------------------------------------
om0 <- clmm(tier ~ horizon_z + (1 | task_id), data = cells, nAGQ = 10)
om1 <- clmm(tier ~ horizon_z + effort + scaffold + (1 | task_id), data = cells, nAGQ = 10)
cat("\n=== CLMM: tier ~ horizon + effort + scaffold + (1 | task) ===\n")
print(summary(om1))
lrt_tier <- anova(om0, om1)
print(lrt_tier)

# ---- (b) beta-binomial count model ---------------------------------------------------
bb <- glmmTMB(cbind(n_correct, n - n_correct) ~ horizon_z + effort * scaffold +
                (1 | task_id),
              family = betabinomial(link = "logit"), data = cells)
cat("\n=== Beta-binomial GLMM of cell success counts ===\n")
print(summary(bb))

tidy_coef <- function(est, se, model) {
  tibble(model = model, term = names(est), estimate = est, se = se,
         l95 = est - 1.96 * se, u95 = est + 1.96 * se,
         p = 2 * pnorm(-abs(est / se)))
}
coefs <- bind_rows(
  tidy_coef(coef(om1)[c("horizon_z", "efforthigh", "scaffoldcode")],
            sqrt(diag(vcov(om1)))[c("horizon_z", "efforthigh", "scaffoldcode")],
            "clmm_tier"),
  with(as.data.frame(summary(bb)$coefficients$cond),
       tidy_coef(setNames(Estimate, rownames(summary(bb)$coefficients$cond)),
                 `Std. Error`, "betabinomial_count"))
)
print(coefs)
write_csv(coefs, out_path("02_cell_models", "coefficients.csv"))
write_csv(tibble(test = "LRT: + effort + scaffold (tier CLMM)",
                 chisq = lrt_tier$LR.stat[2], df = lrt_tier$df[2],
                 p = lrt_tier$`Pr(>Chisq)`[2]),
          out_path("02_cell_models", "tier_lrt.csv"))
message("02_cell_models done.")
