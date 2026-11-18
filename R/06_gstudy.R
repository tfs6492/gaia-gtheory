# =============================================================================
# 06_gstudy.R  --  Generalizability-Theory variance decomposition + D-study
# -----------------------------------------------------------------------------
# Design: tasks (p) crossed with effort (e) and scaffold (s); replications (r)
# nested within each p x e x s cell:  (r : p x e x s).
#
#   Object of measurement : TASK (random; a sample of GAIA-like tasks)
#   Facets                : effort {low, high}, scaffold {tool, code}. Their
#                           levels were chosen deliberately, so they are FIXED,
#                           and their main effects enter the model as fixed effects.
#   Replications          : the residual = run-to-run (stochastic) variation.
#
# Variance components (random part of the mixed model):
#   s2_p    task            stable task difficulty
#   s2_pe   task x effort   does task difficulty reorder across effort levels?
#   s2_ps   task x scaffold does it reorder across scaffolds?
#   s2_pes  task x e x s    cell-specific task x condition interaction
#   s2_res  residual        run-to-run noise within a cell (confounded with any
#                           unmodelled run-level effects, as usual in G-theory)
#
# D-study: dependability of a task's mean success as a function of r, the
# number of replications PER CONDITION. Three universes of generalization:
#  (1) one fixed condition         tau = s2_p + s2_pe + s2_ps + s2_pes
#                                  err = s2_res / r
#  (2) average over the 4 fixed    tau = s2_p + s2_pe/2 + s2_ps/2 + s2_pes/4
#      conditions (fixed facets)   err = s2_res / (4 r)
#  (3) random facets: generalize   tau = s2_p
#      to OTHER effort levels /    err = s2_pe/2 + s2_ps/2 + s2_pes/4 + s2_res/(4 r)
#      scaffolds (sampled 2 x 2)
# The first version of this script called the facets fixed but used formula (3)
# for its "across-condition" column. (2) is the formula consistent with fixed
# facets. (3) is kept because it answers a different, useful question.
# Since all runs are nested in cells, relative and absolute error coincide here.
#
# Models: (A) linear mixed model on the 0/1 outcome (probability scale; the
# classic ANOVA-style G-study, with the usual linear-on-binary caveat), with
# parametric-bootstrap 95% intervals; (B) binomial GLMM on the latent logit
# scale (residual fixed at pi^2/3) as a cross-check.
# Packages: lme4, tidyverse
# =============================================================================

source("R/00_config.R")
library(lme4)

N_BOOT <- 500

d <- load_runs() |>
  mutate(task_effort   = interaction(task_id, effort,   drop = TRUE),
         task_scaffold = interaction(task_id, scaffold, drop = TRUE))

g_formula <- correct ~ effort * scaffold +
  (1 | task_id) + (1 | task_effort) + (1 | task_scaffold) + (1 | cell)

comp_names <- c(task_id = "task", task_effort = "task x effort",
                task_scaffold = "task x scaffold", cell = "task x effort x scaffold",
                Residual = "residual (run-to-run)")

vc_vector <- function(m) {
  vc <- as.data.frame(VarCorr(m))
  v <- setNames(vc$vcov, vc$grp)[names(comp_names)]
  names(v) <- names(comp_names)   # GLMMs have no Residual row -> keep the name
  v[is.na(v)] <- 0
  v
}

R_GRID <- c(1, 2, 3, 5, 10, 20, 30, 50)
dstudy <- function(v, r = R_GRID) {
  p <- v[["task_id"]]; pe <- v[["task_effort"]]; ps <- v[["task_scaffold"]]
  pes <- v[["cell"]]; res <- v[["Residual"]]
  tibble(
    r = r,
    one_condition    = (p + pe + ps + pes) / (p + pe + ps + pes + res / r),
    avg_fixed_facets = (p + pe/2 + ps/2 + pes/4) /
                       (p + pe/2 + ps/2 + pes/4 + res / (4 * r)),
    random_facets    = p / (p + pe/2 + ps/2 + pes/4 + res / (4 * r))
  )
}
gstats <- function(m) {
  v <- vc_vector(m)
  c(v, setNames(v / sum(v), paste0("pct_", names(v))),
    unlist(select(pivot_longer(dstudy(v), -r) |> mutate(k = paste(name, r, sep = "_r")),
                  k, value) |> deframe()))
}

# ---- (A) linear mixed model + parametric bootstrap ---------------------------------
gA <- lmer(g_formula, data = d, control = lmerControl(optimizer = "bobyqa"))
est <- gstats(gA)

cache <- out_path("06_gstudy", "bootstrap.rds")
if (file.exists(cache)) {
  boot <- readRDS(cache)
} else {
  message("Parametric bootstrap (", N_BOOT, " refits) ...")
  # Serial on purpose: gstats() relies on helpers in the global environment,
  # which snow workers would not see. ~1-2 s per refit.
  boot <- bootMer(gA, gstats, nsim = N_BOOT, seed = 2026, use.u = FALSE,
                  type = "parametric")
  saveRDS(boot, cache)
}
ci <- apply(boot$t, 2, quantile, c(.025, .975), na.rm = TRUE)

components <- tibble(
  grp = names(comp_names), component = unname(comp_names),
  variance = est[names(comp_names)],
  pct = 100 * est[paste0("pct_", names(comp_names))],
  pct_l95 = 100 * ci[1, paste0("pct_", names(comp_names))],
  pct_u95 = 100 * ci[2, paste0("pct_", names(comp_names))],
  scale = "probability (linear model)"
)
print(components)

dstudy_tab <- dstudy(est[names(comp_names)]) |>
  pivot_longer(-r, names_to = "universe", values_to = "dependability") |>
  mutate(k = paste(universe, r, sep = "_r"),
         l95 = ci[1, k], u95 = ci[2, k], total_runs_per_task = if_else(
           universe == "one_condition", r, 4 * r)) |>
  select(-k)
print(dstudy_tab |> filter(r %in% c(1, 5, 20)), n = Inf)

# ---- (B) latent-scale GLMM cross-check ---------------------------------------------
gB <- glmer(g_formula, data = d, family = binomial,
            control = glmerControl(optimizer = "bobyqa"))
vB <- vc_vector(gB); vB[["Residual"]] <- pi^2 / 3
latent <- tibble(grp = names(comp_names), component = unname(comp_names),
                 variance = vB, pct = 100 * vB / sum(vB),
                 pct_l95 = NA_real_, pct_u95 = NA_real_, scale = "latent logit (GLMM)")
print(latent)

write_csv(bind_rows(components, latent), out_path("06_gstudy", "variance_components.csv"))
write_csv(dstudy_tab, out_path("06_gstudy", "dstudy.csv"))
message("06_gstudy done.")
