# Load necessary libraries
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales) # Load scales package for potential label formatting

# 1. Load Data
csv_path <- 'src/.output/mps_b2_ga4_amp_neptune_test/prime_measures/results.csv'
# Use read_csv to read the file
# Add error handling in case the file doesn't exist
if (!file.exists(csv_path)) {
  stop("Error: CSV file not found at path: ", csv_path)
}
# Suppress column type messages for cleaner output
df <- read_csv(csv_path, show_col_types = FALSE)

# 2. Split Data based on corpus_file prefix
df_exp <- df %>%
  filter(startsWith(corpus_file, "Exp"))

df_other <- df %>%
  filter(!startsWith(corpus_file, "Exp"))

# --- Function to generate plots (to reduce code duplication) ---
# This function takes a dataframe subset (either df_exp or df_other)
# and a title suffix (e.g., "for 'Exp' Corpora")
generate_plots <- function(data_subset, title_suffix = "") {

  plot_list <- list() # To store the generated plots

  if (nrow(data_subset) == 0) {
      message("Skipping plots ", title_suffix, ": No data.")
      return(plot_list) # Return empty list if no data
  }

  message("Generating plots ", title_suffix, "...")

  # --------------------------------------------------------------------
  # Plot 1: Mean Probability by Prime Condition
  # --------------------------------------------------------------------

  # Preprocess Data for Probability Plot
  df_processed_prob <- data_subset %>%
    mutate(target_structure = sub("^t", "", target_structure)) %>%
    pivot_longer(
      cols = c(logp_con, logp_incon),
      names_to = "prime_type",
      values_to = "logp"
    ) %>%
    mutate(prime = case_when(
      prime_type == "logp_con" ~ "same",
      prime_type == "logp_incon" ~ "different",
      TRUE ~ prime_type
    )) %>%
    mutate(probability = exp(logp)) %>%
    select(corpus_file, target_structure, item_index, prime, probability)

  # Summarize Data for Probability Plot
  df_summary_prob <- df_processed_prob %>%
    group_by(corpus_file, target_structure, prime) %>%
    summarise(
      mean_prob = mean(probability, na.rm = TRUE),
      sd_prob = sd(probability, na.rm = TRUE),
      n = n(),
      .groups = 'drop'
    ) %>%
    mutate(sem_prob = sd_prob / sqrt(n)) %>%
    mutate(sem_prob = ifelse(is.na(sem_prob) | n <= 1, 0, sem_prob))

  # Create Probability Plot
  prob_plot <- ggplot(df_summary_prob,
                      aes(x = target_structure, y = mean_prob, shape = prime)) +
    geom_point(position = position_dodge(width = 0.9), size = 2, color = "black") +
    geom_errorbar(
      aes(ymin = mean_prob - sem_prob, ymax = mean_prob + sem_prob),
      position = position_dodge(width = 0.9),
      width = 0.25,
      color = "black"
    ) +
    facet_wrap(~ corpus_file, scales = "free") +
    labs(
      title = paste("Mean Probability by Target Structure and Prime Condition", title_suffix),
      x = "target",
      y = "Mean Probability (Higher = More Probable)",
      shape = "prime"
    ) +
    scale_shape_manual(values = c("same" = 16, "different" = 17)) +
    # scale_y_continuous(labels = scales::scientific_format(digits = 2)) + # Uncomment if needed
    theme_minimal(base_size = 12) +
    theme(
      axis.text.x = element_text(angle = 0, hjust = 0.5),
      strip.text = element_text(size = 10),
      legend.position = "top"
    )

  plot_list$prob_plot <- prob_plot # Add plot to list

  # ----------------------------------------------------------
  # Plot 2: Mean PE by Target Structure
  # ----------------------------------------------------------

  # Preprocess Data for PE Plot
  df_pe <- data_subset %>%
    mutate(target_structure = sub("^t", "", target_structure)) %>%
    select(corpus_file, target_structure, item_index, pe)

  # Summarize Data for PE Plot
  df_summary_pe <- df_pe %>%
    group_by(corpus_file, target_structure) %>%
    summarise(
      mean_pe = mean(pe, na.rm = TRUE),
      sd_pe = sd(pe, na.rm = TRUE),
      n = n(),
      .groups = 'drop'
    ) %>%
    mutate(sem_pe = sd_pe / sqrt(n)) %>%
    mutate(sem_pe = ifelse(is.na(sem_pe) | n <= 1, 0, sem_pe))

  # Create PE Plot
  pe_plot <- ggplot(df_summary_pe, aes(x = target_structure, y = mean_pe)) +
    geom_hline(yintercept = 0, linetype = "dotted", color = "darkgray", linewidth = 0.7) +
    geom_point(position = position_dodge(width = 0.9), size = 2, color = "black") +
    geom_errorbar(
      aes(ymin = mean_pe - sem_pe, ymax = mean_pe + sem_pe),
      width = 0.25,
      position = position_dodge(width = 0.9),
      color = "black"
    ) +
    facet_wrap(~ corpus_file, scales = "free") +
    labs(
      title = paste("Mean PE by Target Structure", title_suffix),
      x = "Target Structure",
      y = "Mean PE"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      axis.text.x = element_text(angle = 0, hjust = 0.5),
      strip.text = element_text(size = 10)
    )

  plot_list$pe_plot <- pe_plot # Add plot to list

  return(plot_list) # Return list containing both plots
}

# --- Generate and Print Plots for "Exp" Corpora ---
plots_exp <- generate_plots(df_exp, "for 'Exp' Corpora")
if (length(plots_exp) > 0) {
  print(plots_exp$prob_plot)
  # ggsave("probability_plot_exp.png", plots_exp$prob_plot, width = 10, height = 6, dpi = 300) # Optional save
  print(plots_exp$pe_plot)
  # ggsave("pe_plot_exp.png", plots_exp$pe_plot, width = 10, height = 6, dpi = 300) # Optional save
}


# --- Generate and Print Plots for Other Corpora ---
plots_other <- generate_plots(df_other, "for Non-'Exp' Corpora")
if (length(plots_other) > 0) {
  print(plots_other$prob_plot)
  # ggsave("probability_plot_other.png", plots_other$prob_plot, width = 10, height = 6, dpi = 300) # Optional save
  print(plots_other$pe_plot)
  # ggsave("pe_plot_other.png", plots_other$pe_plot, width = 10, height = 6, dpi = 300) # Optional save
}