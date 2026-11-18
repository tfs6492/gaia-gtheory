# =============================================================================
# 08_figures.R  --  all README figures (ggplot2 + Okabe-Ito colours from {see})
# -----------------------------------------------------------------------------
# Reads data/gaia_runs_public.csv and the CSV tables in outputs/, writes PNGs
# to figs/. Run after 01-07. Figure 6 needs the brms output of 01; it is
# skipped with a message if that table is missing.
# Packages: tidyverse, see, scales
# =============================================================================

source("R/00_config.R")
library(scales)

d <- load_runs()
oi <- function(...) unname(okabeito_colors(...))

wilson <- function(k, n, z = 1.96) {
  p <- k / n; den <- 1 + z^2 / n
  ctr <- (p + z^2 / (2 * n)) / den
  hw  <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / den
  tibble(p = p, lo = ctr - hw, hi = ctr + hw)
}

# ---- Fig 1: per-task success under each condition -------------------------------------
task_cond <- d |>
  group_by(task_id, condition, has_attachment) |>
  summarise(success = mean(correct), .groups = "drop")
task_meta <- d |>
  group_by(task_id) |>
  summarise(mean_success = mean(correct), leak_share = mean(leak_flag),
            has_attachment = first(has_attachment), .groups = "drop") |>
  mutate(label = paste0(substr(task_id, 1, 8),
                        if_else(has_attachment == 1, "  [file]", ""),
                        if_else(leak_share >= 0.10, "  *", "")),
         label = fct_reorder(label, mean_success))
task_cond <- left_join(task_cond, select(task_meta, task_id, label), by = "task_id")
task_range <- task_cond |> group_by(label) |>
  summarise(lo = min(success), hi = max(success), .groups = "drop")

p1 <- ggplot(task_cond, aes(y = label)) +
  geom_segment(data = task_range, aes(x = lo, xend = hi, yend = label),
               colour = "grey80", linewidth = 1.6, lineend = "round") +
  geom_point(aes(x = success, colour = condition), size = 2.4, alpha = 0.9) +
  scale_colour_manual(values = COND_COLOURS, name = "Effort / scaffold") +
  scale_x_continuous(labels = percent, breaks = seq(0, 1, 0.25),
                     expand = expansion(mult = 0.02)) +
  labs(title = "Success rate of each GAIA Level 1 task under the four conditions",
       subtitle = "20 runs per point. Grey bar = range across conditions. Tasks ordered by mean success.",
       caption = paste("[file] = task needs an attachment the harness never supplied.",
                       "* = at least 10% of the task's runs were leak-flagged (agent found GAIA material or an answer key)."),
       x = "Share of runs scored correct", y = NULL) +
  theme_gaia(11) +
  theme(axis.text.y = element_text(family = "mono", size = 7.5),
        panel.grid.major.y = element_blank())
save_fig(p1, "fig1_task_success", width = 8.5, height = 10.5)

# ---- Fig 2: answer outcome categories by condition --------------------------------------
cat_labels <- c(exact_correct = "Exact correct", units_scale_error = "Units / scale error",
                numeric_near_miss = "Numeric near miss", string_near_miss = "String near miss",
                wrong = "Wrong", no_answer = "No answer (crash / timeout)")
cat_cols <- c("Exact correct" = oi("bluish green"), "Units / scale error" = oi("sky blue"),
              "Numeric near miss" = oi("yellow"), "String near miss" = oi("blue"),
              "Wrong" = oi("vermillion"), "No answer (crash / timeout)" = oi("grey"))
cats <- read_csv(out_path("04_answer_nearmiss", "categories_by_condition.csv"),
                 show_col_types = FALSE) |>
  filter(n > 0) |>
  mutate(category = factor(cat_labels[answer_category], levels = rev(cat_labels)),
         condition = factor(condition, levels = rev(levels(d$condition))))

p2 <- ggplot(cats, aes(x = pct / 100, y = condition, fill = category)) +
  geom_col(width = 0.7, colour = "white", linewidth = 0.3) +
  geom_text(aes(label = if_else(pct >= 4, sprintf("%.0f%%", pct), "")),
            position = position_stack(vjust = 0.5), size = 3.2, colour = "white",
            fontface = "bold") +
  scale_fill_manual(values = cat_cols,
                    breaks = unname(cat_labels[cat_labels %in% cats$category]),
                    name = NULL) +
  scale_x_continuous(labels = percent, expand = expansion(mult = c(0, 0.03))) +
  guides(fill = guide_legend(nrow = 2)) +
  labs(title = "What the misses are made of",
       subtitle = "Share of runs per answer category. Near misses are classified from the answer strings (R/00_prepare_data.R).",
       x = NULL, y = "Effort / scaffold") +
  theme_gaia() + theme(panel.grid = element_blank())
save_fig(p2, "fig2_answer_categories", width = 9, height = 4)

# ---- Fig 3: G-study variance components ------------------------------------------------
vc <- read_csv(out_path("06_gstudy", "variance_components.csv"), show_col_types = FALSE) |>
  mutate(component = fct_rev(factor(component, levels = unique(component))),
         scale = factor(scale, levels = c("probability (linear model)", "latent logit (GLMM)"),
                        labels = c("Probability scale (linear mixed model)",
                                   "Latent logit scale (binomial GLMM)")))
p3 <- ggplot(vc, aes(x = pct / 100, y = component, fill = scale)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  geom_errorbar(aes(xmin = pct_l95 / 100, xmax = pct_u95 / 100),
                position = position_dodge(width = 0.75), width = 0.25,
                colour = "grey20", na.rm = TRUE) +
  geom_text(aes(label = sprintf("%.1f%%", pct),
                x = pmax(pct, coalesce(pct_u95, pct)) / 100 + 0.012),
            position = position_dodge(width = 0.75), hjust = 0, size = 3.1) +
  scale_fill_manual(values = oi(c("blue", "orange")), name = NULL) +
  scale_x_continuous(labels = percent, expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Where the variance in run success comes from",
       subtitle = "G-study variance components as a share of total. Error bars: 95% parametric bootstrap (probability scale).",
       x = "Share of total variance", y = NULL) +
  theme_gaia()
save_fig(p3, "fig3_gstudy_components", width = 8.5, height = 4.5)

# ---- Fig 4: D-study -----------------------------------------------------------------------
ds <- read_csv(out_path("06_gstudy", "dstudy.csv"), show_col_types = FALSE) |>
  mutate(universe = factor(universe,
    levels = c("one_condition", "avg_fixed_facets", "random_facets"),
    labels = c("Within one condition",
               "Averaged over the 4 conditions (fixed facets)",
               "Generalizing to other efforts / scaffolds (random facets)")))
p4 <- ggplot(ds, aes(x = r, y = dependability, colour = universe, fill = universe)) +
  geom_ribbon(aes(ymin = l95, ymax = u95), alpha = 0.15, colour = NA) +
  geom_line(linewidth = 1) + geom_point(size = 1.8) +
  geom_vline(xintercept = 20, linetype = "dashed", colour = "grey50") +
  annotate("text", x = 20, y = 0.32, label = "this study: r = 20", hjust = -0.05,
           size = 3.2, colour = "grey35") +
  scale_x_log10(breaks = c(1, 2, 3, 5, 10, 20, 50)) +
  scale_y_continuous(limits = c(0.3, 1), breaks = seq(0.3, 1, 0.1)) +
  scale_colour_manual(values = oi(c("bluish green", "blue", "vermillion")), name = NULL) +
  scale_fill_manual(values = oi(c("bluish green", "blue", "vermillion")), name = NULL) +
  guides(colour = guide_legend(ncol = 1), fill = guide_legend(ncol = 1)) +
  labs(title = "How many replications make a task's success rate dependable?",
       subtitle = "D-study dependability vs replications per condition r (log scale). Bands: 95% bootstrap.",
       x = "Replications per condition (r)", y = "Dependability") +
  theme_gaia()
save_fig(p4, "fig4_dstudy", width = 8, height = 5.2)

# ---- Fig 5: sensitivity of the condition effects ------------------------------------------
sens <- read_csv(out_path("07_sensitivity", "glmm_coefficients.csv"), show_col_types = FALSE) |>
  filter(term %in% c("efforthigh", "scaffoldcode", "efforthigh:scaffoldcode")) |>
  mutate(term = factor(term, levels = c("efforthigh", "scaffoldcode", "efforthigh:scaffoldcode"),
                       labels = c("High effort\n(vs low, tool scaffold)",
                                  "Code scaffold\n(vs tool, low effort)",
                                  "Effort x scaffold\ninteraction")),
         subset = factor(SUBSET_LABELS[subset], levels = rev(SUBSET_LABELS)))
sub_cols <- setNames(oi(c("black", "vermillion", "orange", "sky blue", "reddish purple")),
                     SUBSET_LABELS)
p5 <- ggplot(sens, aes(x = estimate, y = subset, colour = subset)) +
  geom_vline(xintercept = 0, colour = "grey60") +
  geom_errorbar(aes(xmin = l95, xmax = u95), orientation = "y", width = 0.25, linewidth = 0.7) +
  geom_point(size = 2.6) +
  facet_wrap(~ term, nrow = 1) +
  scale_colour_manual(values = sub_cols, guide = "none") +
  labs(title = "Condition effects under each data restriction",
       subtitle = "GLMM: correct ~ effort * scaffold + horizon + (1 | task). Log-odds, 95% Wald CI.",
       x = "Log-odds", y = NULL) +
  theme_gaia() + theme(panel.spacing.x = unit(1.2, "lines"))
save_fig(p5, "fig5_sensitivity_effects", width = 10, height = 4.2)

# ---- Fig 6: predicted success per condition (Bayesian base model) ---------------------------
pred_file <- out_path("01_bayes_glmm", "condition_predictions.csv")
if (file.exists(pred_file)) {
  pr <- read_csv(pred_file, show_col_types = FALSE) |>
    mutate(model = factor(model, levels = c("base", "base_timeouts_missing"),
                          labels = c("All runs (timeouts scored 0)", "Timeouts treated as missing")),
           effort = factor(effort, levels = c("low", "high")),
           scaffold = factor(scaffold, levels = c("tool", "code")))
  p6 <- ggplot(pr, aes(x = effort, y = p_median, colour = scaffold, group = scaffold)) +
    geom_line(position = position_dodge(width = 0.25), linewidth = 0.8) +
    geom_pointrange(aes(ymin = p_l95, ymax = p_u95), position = position_dodge(width = 0.25),
                    size = 0.5, linewidth = 0.8) +
    facet_wrap(~ model) +
    scale_colour_manual(values = oi(c("blue", "vermillion")), name = "Scaffold") +
    scale_y_continuous(labels = percent) +
    labs(title = "Predicted success for a typical task",
         subtitle = "Bayesian GLMM (01), population level (task effect = 0, mean horizon). Median and 95% credible interval.",
         x = "Reasoning effort", y = "P(correct)") +
    theme_gaia()
  save_fig(p6, "fig6_condition_predictions", width = 8, height = 4.6)
} else message("Skipping fig6: run R/01_bayes_glmm.R first.")

# ---- Fig 7: wall time of correct vs incorrect runs inside mixed cells ------------------------
mixed <- d |>
  filter(crashed == 0) |>
  group_by(cell) |> filter(any(correct == 1), any(correct == 0)) |> ungroup() |>
  mutate(outcome = factor(if_else(correct == 1, "Correct", "Incorrect"),
                          levels = c("Correct", "Incorrect")))
p7 <- ggplot(mixed, aes(x = wall_time_s, fill = outcome, colour = outcome)) +
  geom_density(alpha = 0.3, linewidth = 0.7, adjust = 1.1) +
  geom_vline(xintercept = 240, linetype = "dashed", colour = "grey40") +
  facet_wrap(~ condition, nrow = 2) +
  scale_x_log10(breaks = c(3, 10, 30, 100, 240, 1000), labels = label_number()) +
  scale_fill_manual(values = oi(c("bluish green", "vermillion")), name = NULL) +
  scale_colour_manual(values = oi(c("bluish green", "vermillion")), name = NULL) +
  labs(title = "Wall time of correct vs incorrect runs inside mixed cells",
       subtitle = paste("Completed runs in cells with both outcomes, pooled across cells. Dashed line: nominal 240 s timeout.",
                        "Holding the cell fixed (R/03_within_cell.R), wall time does not predict success.", sep = "\n"),
       x = "Wall time (s, log scale)", y = "Density") +
  theme_gaia()
save_fig(p7, "fig7_wall_time_mixed", width = 8.5, height = 5.5)

# ---- Fig 8: leakage and attachment diagnostics --------------------------------------------
diag <- bind_rows(
  d |> group_by(panel = "Runs, by leak flag",
                group = if_else(leak_flag == 1, "Leak-flagged", "Not flagged")) |>
    summarise(k = sum(correct), n = n(), .groups = "drop"),
  d |> group_by(panel = "Tasks, by attachment",
                group = if_else(has_attachment == 1, "Needs attachment\n(never supplied)",
                                "No attachment")) |>
    summarise(k = sum(correct), n = n(), .groups = "drop")
)
diag <- bind_cols(diag, wilson(diag$k, diag$n)) |>
  mutate(label = sprintf("%.0f%%  (n = %s runs)", 100 * p, comma(n)))
p8 <- ggplot(diag, aes(x = p, y = group)) +
  geom_errorbar(aes(xmin = lo, xmax = hi), orientation = "y", width = 0.2, colour = "grey30") +
  geom_point(aes(colour = panel), size = 3.2) +
  geom_text(aes(label = label, x = hi + 0.015), hjust = 0, size = 3.2) +
  facet_wrap(~ panel, scales = "free_y", ncol = 1) +
  geom_vline(xintercept = mean(d$correct), linetype = "dashed", colour = "grey50") +
  scale_colour_manual(values = oi(c("reddish purple", "orange")), guide = "none") +
  scale_x_continuous(labels = percent, limits = c(0.3, 1)) +
  labs(title = "Signs of benchmark leakage",
       subtitle = "Success rate with 95% Wilson interval. Dashed line: overall success (57.2%).",
       x = "Share of runs scored correct", y = NULL) +
  theme_gaia() + theme(panel.grid.major.y = element_blank())
save_fig(p8, "fig8_leakage", width = 7.5, height = 4.6)

message("Figures written to ", FIG_DIR, "/")
