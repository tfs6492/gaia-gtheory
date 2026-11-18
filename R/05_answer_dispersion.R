# =============================================================================
# 05_answer_dispersion.R  --  how scattered are a cell's answers?
# -----------------------------------------------------------------------------
# For each design cell (task x effort x scaffold), over the runs that produced
# an answer (crashes, timeouts and empty answers are counted separately, not
# treated as one more "answer"):
#   n_distinct    number of distinct normalized answers (via answer_id)
#   entropy_bits  Shannon entropy of the answer distribution
#   norm_entropy  entropy / log2(n answered)   0 = identical, 1 = all different
#   modal_freq    share of answered runs giving the most common answer
#   modal_correct whether that modal answer is the correct one
# Low entropy + wrong = consistently wrong the SAME way (systematic error).
# High entropy        = scattered answers (run-to-run instability).
# Packages: tidyverse
# =============================================================================

source("R/00_config.R")
d <- load_runs()

shannon_bits <- function(x) { p <- table(x) / length(x); -sum(p * log2(p)) }

disp <- d |>
  group_by(task_id, condition, effort, scaffold) |>
  summarise(
    n_runs     = n(),
    n_correct  = sum(correct),
    n_answered = sum(!is.na(answer_id)),
    n_distinct = n_distinct(answer_id, na.rm = TRUE),
    entropy_bits = if (n_answered > 0) shannon_bits(na.omit(answer_id)) else NA_real_,
    modal_freq = if (n_answered > 0) max(table(na.omit(answer_id))) / n_answered else NA_real_,
    modal_correct = if (n_answered > 0) {
      modal <- names(which.max(table(na.omit(answer_id))))
      any(correct[answer_id %in% modal] == 1)
    } else NA,
    .groups = "drop"
  ) |>
  mutate(
    norm_entropy = if_else(n_answered > 1, entropy_bits / log2(n_answered), NA_real_),
    tier = factor(case_when(n_correct == 0 ~ "never", n_correct == n_runs ~ "always",
                            TRUE ~ "mixed"), levels = c("never", "mixed", "always"))
  )

by_tier <- disp |>
  group_by(tier) |>
  summarise(cells = n(), mean_distinct = mean(n_distinct),
            mean_norm_entropy = mean(norm_entropy, na.rm = TRUE),
            mean_modal_freq = mean(modal_freq, na.rm = TRUE),
            share_modal_correct = mean(modal_correct, na.rm = TRUE), .groups = "drop")
print(by_tier)

never_style <- disp |>
  filter(tier == "never") |>
  mutate(style = if_else(n_distinct <= 1, "single repeated wrong answer",
                         "varied wrong answers")) |>
  count(style)
print(never_style)

write_csv(disp,        out_path("05_answer_dispersion", "cell_dispersion.csv"))
write_csv(by_tier,     out_path("05_answer_dispersion", "dispersion_by_tier.csv"))
write_csv(never_style, out_path("05_answer_dispersion", "never_cells_style.csv"))
message("05_answer_dispersion done.")
