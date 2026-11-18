# =============================================================================
# 04_answer_nearmiss.R  --  how wrong are the wrong answers?
# -----------------------------------------------------------------------------
# The GAIA scorer is exact-match, so a "miss" ranges from a units slip to a
# completely wrong answer. Each run's `answer_category` is assigned in
# R/00_prepare_data.R (that step needs the private answer strings; see the
# rules there). This script summarizes the categories and shows how accuracy
# moves under more lenient credit, overall and by condition.
# Packages: tidyverse
# =============================================================================

source("R/00_config.R")
d <- load_runs()

cat_levels <- c("exact_correct", "units_scale_error", "numeric_near_miss",
                "string_near_miss", "wrong", "no_answer")
d <- mutate(d, answer_category = factor(answer_category, levels = cat_levels))

overall <- d |> count(answer_category, .drop = FALSE) |> mutate(pct = 100 * n / sum(n))
print(overall)

by_condition <- d |>
  count(condition, answer_category, .drop = FALSE) |>
  group_by(condition) |> mutate(pct = 100 * n / sum(n)) |> ungroup()
print(by_condition |> select(-n) |>
        pivot_wider(names_from = answer_category, values_from = pct) |>
        mutate(across(where(is.numeric), \(x) round(x, 1))))

near <- c("exact_correct", "units_scale_error", "numeric_near_miss", "string_near_miss")
lenient <- bind_rows(mutate(d, condition = "all"),
                     mutate(d, condition = as.character(condition))) |>
  group_by(condition) |>
  summarise(exact = mean(correct),
            plus_units_scale = mean(answer_category %in% c("exact_correct", "units_scale_error")),
            plus_all_near = mean(answer_category %in% near),
            .groups = "drop")
print(lenient)

write_csv(overall,      out_path("04_answer_nearmiss", "categories_overall.csv"))
write_csv(by_condition, out_path("04_answer_nearmiss", "categories_by_condition.csv"))
write_csv(lenient,      out_path("04_answer_nearmiss", "lenient_accuracy.csv"))
message("04_answer_nearmiss done.")
