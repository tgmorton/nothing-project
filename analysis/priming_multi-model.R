# --- 0. Load Libraries ---
message("Loading libraries...")
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(forcats)) # For reordering factors
suppressPackageStartupMessages(library(scales))   # For log scales
message("Libraries loaded.")

# --- 1. Setup & Logging ---
# <<< IMPORTANT: Replace with your combined CSV path >>>
csv_path <- '/Users/thomasmorton/Nothing-Project/analysis/may20_10m_seedsweep_priming_summary_combined.csv'
# <<< Main directory for all plots >>>
output_plot_dir <- "analysis/plots_priming/may20_seedsweep_new_metrics_organized"

overall_start_time <- Sys.time()
message("--- Script started at ", format(overall_start_time, "%Y-%m-%d %H:%M:%S %Z"), " ---")
message("Using input file: ", csv_path)
message("Output plots will be saved in subdirectories under: ", output_plot_dir)

if (!file.exists(csv_path)) {
  stop("Error: Input CSV file not found at path: ", csv_path)
}

if (!dir.exists(output_plot_dir)) {
  message("Creating base output directory: ", output_plot_dir)
  dir.create(output_plot_dir, recursive = TRUE)
}

# Helper function to sanitize strings for filenames
sanitize_filename <- function(name_str, replacement = "_") {
  name_str <- gsub("[^a-zA-Z0-9_.-]", replacement, name_str)
  name_str <- gsub(paste0(replacement, replacement, "+"), replacement, name_str)
  return(name_str)
}

# Helper function for symmetric log transformation (for priming_effect_normalized which can be negative)
symmetric_log_trans <- function() {
  trans_new(
    "symmetric_log",
    transform = function(x) sign(x) * log1p(abs(x)),
    inverse = function(y) sign(y) * (exp(abs(y)) - 1)
  )
}

# --- 2. Read Data ---
message("Starting data read...")
read_start_time <- Sys.time()
df_raw <- read_csv(
  csv_path,
  col_types = cols(.default = col_double(),
                   model_run_name = col_character(),
                   corpus_file = col_character(),
                   metric_base = col_character(),
                   contrast_pair = col_character()),
  lazy = TRUE,
  show_col_types = FALSE
)
message("Data read complete (", nrow(df_raw), " rows). Time taken: ", format(Sys.time() - read_start_time))


# --- 3. Major Data Transformation (Corrected Logic) ---
message("Starting major data transformation...")
transform_start_time <- Sys.time()

# 3a. Pivot to a long format to handle multiple values per row (value_struct1, value_struct2)
df_long <- df_raw %>%
  filter(!is.na(contrast_pair)) %>%
  separate(contrast_pair, into = c("structure_name_1", "structure_name_2"), sep = "/", remove = FALSE, fill = "right") %>%
  pivot_longer(
    cols = c(value_struct1, value_struct2),
    names_to = "value_col_origin",
    values_to = "value",
    values_drop_na = TRUE
  ) %>%
  mutate(
    structure = case_when(
      value_col_origin == "value_struct1" ~ structure_name_1,
      value_col_origin == "value_struct2" ~ structure_name_2,
      TRUE ~ NA_character_
    )
  )

# 3b. Isolate the 'count' data. 'count' is a property of a structure for a given model, step, and corpus.
df_counts <- df_long %>%
  filter(metric_base == "count") %>%
  select(model_run_name, seed_number, checkpoint_step, corpus_file, structure, count = value) %>%
  distinct() # Ensure counts are unique per group

# 3c. Process all other metrics.
df_metrics <- df_long %>%
  filter(metric_base != "count") %>%
  # Intelligently parse the metric_base to separate summary type (avg, std) from the core metric name
  mutate(
    summary_stat = case_when(
      str_starts(metric_base, "avg_") ~ "avg",
      str_starts(metric_base, "std_") ~ "std",
      TRUE ~ "value" # For single-value metrics like 'priming_effect_normalized'
    ),
    core_metric = str_remove(metric_base, "^(avg|std)_")
  ) %>%
  select(-metric_base, -value_col_origin, -structure_name_1, -structure_name_2)

# 3d. Pivot the metrics data to create columns for 'avg', 'std', and 'value'.
df_metrics_wide <- df_metrics %>%
  pivot_wider(
    names_from = summary_stat,
    values_from = value
  )

# 3e. Join the 'count' data back to the main metrics data frame.
# This ensures every metric row has the correct sample size associated with its structure.
df_wide <- df_metrics_wide %>%
  left_join(df_counts, by = c("model_run_name", "seed_number", "checkpoint_step", "corpus_file", "structure"))

# 3f. Now, with a correctly formed 'count' column, calculate SEM. This should now work without error.
df_wide <- df_wide %>%
  mutate(
    sem = if_else(!is.na(std) & !is.na(count) & count > 0, std / sqrt(count), NA_real_)
  ) %>%
  # Reorder columns for clarity and future use
  select(
    model_run_name, seed_number, checkpoint_step, corpus_file, contrast_pair,
    core_metric, structure, avg, std, sem, count, value
  ) %>%
  arrange(model_run_name, checkpoint_step, corpus_file, core_metric, structure)

message("Data transformation complete. Final wide table has ", nrow(df_wide), " rows.")
message("Structure of the main transformed data:")
glimpse(df_wide)


# --- 4. Prepare Data for Final Checkpoint Comparison Plots ---
message("Preparing summarized data for final checkpoint comparison plots...")

final_checkpoints <- df_wide %>%
  group_by(model_run_name) %>%
  summarise(last_step = max(checkpoint_step), .groups = 'drop')

df_final_step <- df_wide %>%
  inner_join(final_checkpoints, by = c("model_run_name", "checkpoint_step" = "last_step"))

df_summary_for_plots <- df_final_step %>%
  mutate(metric_value = coalesce(avg, value)) %>%
  filter(!is.na(metric_value)) %>%
  group_by(model_run_name, checkpoint_step, corpus_file, contrast_pair, core_metric, structure) %>%
  summarise(
    mean_value = mean(metric_value, na.rm = TRUE),
    sd_value = sd(metric_value, na.rm = TRUE),
    n_seeds = n(),
    sem_value = if_else(n_seeds > 1, sd_value / sqrt(n_seeds), 0),
    .groups = 'drop'
  )

message("Summarized data for plots is ready.")
glimpse(df_summary_for_plots)

# --- 5. Generate Plots ---
message("--- Starting Plot Generation ---")
plot_start_time <- Sys.time()
unique_models <- unique(df_wide$model_run_name)

# Create a more organized folder structure
dir.create(file.path(output_plot_dir, "1_timeseries_individual_models", "pe_sinclair"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_plot_dir, "1_timeseries_individual_models", "norm_p_comparison"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_plot_dir, "2_timeseries_combined_models", "pe_sinclair"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_plot_dir, "3_final_checkpoint_comparison", "priming_effect_normalized"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_plot_dir, "3_final_checkpoint_comparison", "priming_effect_normalized_log"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_plot_dir, "3_final_checkpoint_comparison", "norm_p_comparison_bar"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_plot_dir, "3_final_checkpoint_comparison", "norm_p_comparison_bar_log"), recursive = TRUE, showWarnings = FALSE)


# --- 5a. Loop: `pe_sinclair` over Time (Individual Models) ---
message("--- Generating INDIVIDUAL `pe_sinclair` timeseries plots ---")
for (current_model in unique_models) {
  df_model_data <- df_wide %>% filter(model_run_name == current_model)
  for (current_corpus in unique(df_model_data$corpus_file)) {
    plot_data <- df_model_data %>%
      filter(corpus_file == current_corpus, core_metric == "pe_sinclair", !is.na(avg), !is.na(sem))
    if (nrow(plot_data) > 0 && n_distinct(plot_data$checkpoint_step) > 1) {
      plot_folder <- file.path(output_plot_dir, "1_timeseries_individual_models", "pe_sinclair", sanitize_filename(current_model))
      dir.create(plot_folder, recursive = TRUE, showWarnings = FALSE)
      plot_filename <- file.path(plot_folder, paste0("pe_sinclair_", sanitize_filename(current_corpus), ".png"))
      p <- ggplot(plot_data, aes(x = checkpoint_step, y = avg, color = structure, fill = structure, group = structure)) +
        geom_line(alpha = 0.9) + geom_ribbon(aes(ymin = avg - sem, ymax = avg + sem), alpha = 0.2, color = NA) + geom_point(size = 1.5) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
        labs(title = paste("Sinclair Priming Effect | Model:", current_model), subtitle = paste("Corpus:", current_corpus), x = "Checkpoint Step", y = "Average pe_sinclair (+/- SEM)") +
        theme_minimal(base_size = 10) + theme(legend.position = "top", axis.text.x = element_text(angle = 45, hjust = 1))
      ggsave(plot_filename, plot = p, width = 8, height = 6, dpi = 150)
    }
  }
}

# --- 5b. Loop: `pe_sinclair` over Time (Combined Models) ---
message("--- Generating COMBINED `pe_sinclair` timeseries plots ---")
for (current_corpus in unique(df_wide$corpus_file)) {
  for (current_structure in unique(filter(df_wide, corpus_file == current_corpus)$structure)) {
      plot_data <- df_wide %>%
        filter(corpus_file == current_corpus, core_metric == "pe_sinclair", structure == current_structure, !is.na(avg), !is.na(sem)) %>%
        group_by(model_run_name) %>% filter(n_distinct(checkpoint_step) > 1) %>% ungroup()
      if(nrow(plot_data) > 0) {
        plot_folder <- file.path(output_plot_dir, "2_timeseries_combined_models", "pe_sinclair")
        plot_filename <- file.path(plot_folder, paste0("combined_pe_sinclair_", sanitize_filename(current_corpus), "_", sanitize_filename(current_structure), ".png"))
        p <- ggplot(plot_data, aes(x = checkpoint_step, y = avg, color = model_run_name, fill = model_run_name, group = model_run_name)) +
          geom_line(alpha = 0.7) + geom_ribbon(aes(ymin = avg - sem, ymax = avg + sem), alpha = 0.15, color = NA) +
          geom_point(size = 1, alpha = 0.5) +
          geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
          labs(title = paste("Combined Sinclair PE for Structure:", current_structure), subtitle = paste("Corpus:", current_corpus), x = "Checkpoint Step", y = "Average pe_sinclair (+/- SEM)", color = "Model") +
          guides(fill="none") +
          theme_minimal(base_size = 11) + theme(legend.position = "top", axis.text.x = element_text(angle = 45, hjust = 1))
        ggsave(plot_filename, plot = p, width = 10, height = 7, dpi = 150)
      }
  }
}


# --- 5c. Loop: `norm_p_conT_given...` over Time (Individual Models) ---
message("--- Generating INDIVIDUAL `norm_p_conT_given...` timeseries plots ---")
for (current_model in unique_models) {
  df_model_data <- df_wide %>% filter(model_run_name == current_model)
  for (current_corpus in unique(df_model_data$corpus_file)) {
    plot_data <- df_model_data %>%
      filter(corpus_file == current_corpus, core_metric %in% c("norm_p_conT_given_conP", "norm_p_conT_given_inconP"), !is.na(avg)) %>%
      mutate(condition = fct_recode(core_metric, "Congruent" = "norm_p_conT_given_conP", "Incongruent" = "norm_p_conT_given_inconP"))
    if (nrow(plot_data) > 0 && n_distinct(plot_data$checkpoint_step) > 1) {
      plot_folder <- file.path(output_plot_dir, "1_timeseries_individual_models", "norm_p_comparison", sanitize_filename(current_model))
      dir.create(plot_folder, recursive = TRUE, showWarnings = FALSE)
      plot_filename <- file.path(plot_folder, paste0("norm_p_timeseries_", sanitize_filename(current_corpus), ".png"))
      p <- ggplot(plot_data, aes(x = checkpoint_step, y = avg, color = condition, group = condition)) +
        geom_line() +
        facet_wrap(~ structure, scales = "free_y") +
        scale_color_brewer(palette = "Set1") +
        labs(title = paste("Normalized Probability of Target over Time | Model:", current_model), subtitle = paste("Corpus:", current_corpus), x = "Checkpoint Step", y = "Average Normalized Probability", color = "Prime Condition") +
        theme_minimal(base_size = 10) + theme(legend.position = "top", axis.text.x = element_text(angle = 45, hjust = 1))
      ggsave(plot_filename, plot = p, width = 9, height = 7, dpi = 150)
    }
  }
}

# --- 5d. Loops for Final Checkpoint Comparisons ---
message("--- Generating Final Checkpoint Comparison Plots ---")
for (current_model in unique_models) {
  df_model_summary <- df_summary_for_plots %>% filter(model_run_name == current_model)
  for (current_corpus in unique(df_model_summary$corpus_file)) {
    # Plot 1: Normalized Priming Effect (Point Plot)
    plot_data_norm_pe <- df_model_summary %>% filter(corpus_file == current_corpus, core_metric == "priming_effect_normalized")
    if (nrow(plot_data_norm_pe) > 0) {
      plot_folder <- file.path(output_plot_dir, "3_final_checkpoint_comparison", "priming_effect_normalized", sanitize_filename(current_model))
      dir.create(plot_folder, recursive = TRUE, showWarnings = FALSE)
      plot_filename <- file.path(plot_folder, paste0("norm_pe_comp_", sanitize_filename(current_corpus), ".png"))
      p1 <- ggplot(plot_data_norm_pe, aes(x = structure, y = mean_value, color = structure)) +
        geom_point(size = 3) + geom_errorbar(aes(ymin = mean_value - sem_value, ymax = mean_value + sem_value), width = 0.2) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") + facet_wrap(~ contrast_pair, scales = "free_x") +
        labs(title = "Normalized Priming Effect (Final Ckpt)", subtitle = paste("Model:", current_model, "| Corpus:", current_corpus), x = "Structure", y = "Mean Norm. PE (+/- SEM)") +
        theme_minimal(base_size = 11) + theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))
      ggsave(plot_filename, plot = p1, width = 9, height = 6, dpi = 150)
    }

    # Plot 2: Log-Transformed Normalized Priming Effect (Point Plot)
    if (nrow(plot_data_norm_pe) > 0) {
      plot_folder <- file.path(output_plot_dir, "3_final_checkpoint_comparison", "priming_effect_normalized_log", sanitize_filename(current_model))
      dir.create(plot_folder, recursive = TRUE, showWarnings = FALSE)
      plot_filename_log <- file.path(plot_folder, paste0("norm_pe_log_comp_", sanitize_filename(current_corpus), ".png"))
      p2 <- ggplot(plot_data_norm_pe, aes(x = structure, y = mean_value, color = structure)) +
          geom_point(size = 3) + geom_errorbar(aes(ymin = mean_value - sem_value, ymax = mean_value + sem_value), width = 0.2) +
          geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
          scale_y_continuous(trans = symmetric_log_trans()) +
          facet_wrap(~ contrast_pair, scales = "free_x") +
          labs(title = "Log-Transformed Normalized Priming Effect (Final Ckpt)", subtitle = paste("Model:", current_model, "| Corpus:", current_corpus), x = "Structure", y = "Mean Norm. PE (+/- SEM, Symmetric Log Scale)") +
          theme_minimal(base_size = 11) + theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))
      ggsave(plot_filename_log, plot = p2, width = 9, height = 6, dpi = 150)
    }

    # Data for Norm P Bar plots
    plot_data_norm_p <- df_model_summary %>%
      filter(corpus_file == current_corpus, core_metric %in% c("norm_p_conT_given_conP", "norm_p_conT_given_inconP")) %>%
      mutate(condition = fct_recode(core_metric, "Congruent Prime" = "norm_p_conT_given_conP", "Incongruent Prime" = "norm_p_conT_given_inconP"))

    # Plot 3: Normalized Probability Comparison (Bar Plot)
    if (nrow(plot_data_norm_p) > 0) {
      plot_folder <- file.path(output_plot_dir, "3_final_checkpoint_comparison", "norm_p_comparison_bar", sanitize_filename(current_model))
      dir.create(plot_folder, recursive = TRUE, showWarnings = FALSE)
      plot_filename <- file.path(plot_folder, paste0("norm_p_bar_comp_", sanitize_filename(current_corpus), ".png"))
      p3 <- ggplot(plot_data_norm_p, aes(x = structure, y = mean_value, fill = condition)) +
        geom_bar(stat = "identity", position = position_dodge(0.9)) +
        geom_errorbar(aes(ymin = mean_value - sem_value, ymax = mean_value + sem_value), width = 0.25, position = position_dodge(0.9)) +
        facet_wrap(~ contrast_pair, scales = "free") +
        scale_fill_brewer(palette = "Set1") +
        labs(title = "Normalized Probability of Target (Final Ckpt)", subtitle = paste("Model:", current_model, "| Corpus:", current_corpus), x = "Target Structure", y = "Mean Norm. Prob. (+/- SEM)", fill = "Prime Condition") +
        theme_minimal(base_size = 11) + theme(legend.position = "top", axis.text.x = element_text(angle = 45, hjust = 1))
      ggsave(plot_filename, plot = p3, width = 10, height = 7, dpi = 150)
    }

    # Plot 4: Log-Transformed Normalized Probability Comparison (Bar Plot)
    if (nrow(plot_data_norm_p) > 0) {
       plot_folder <- file.path(output_plot_dir, "3_final_checkpoint_comparison", "norm_p_comparison_bar_log", sanitize_filename(current_model))
       dir.create(plot_folder, recursive = TRUE, showWarnings = FALSE)
       plot_filename_log <- file.path(plot_folder, paste0("norm_p_bar_log_comp_", sanitize_filename(current_corpus), ".png"))
       p4 <- ggplot(plot_data_norm_p, aes(x = structure, y = mean_value, fill = condition)) +
         geom_bar(stat = "identity", position = position_dodge(0.9)) +
         geom_errorbar(aes(ymin = mean_value - sem_value, ymax = mean_value + sem_value), width = 0.25, position = position_dodge(0.9)) +
         scale_y_log10(labels = label_scientific()) +
         facet_wrap(~ contrast_pair, scales = "free") +
         scale_fill_brewer(palette = "Set1") +
         labs(title = "Log-Scaled Normalized Probability of Target (Final Ckpt)", subtitle = paste("Model:", current_model, "| Corpus:", current_corpus), x = "Target Structure", y = "Mean Norm. Prob. (+/- SEM, Log Scale)", fill = "Prime Condition") +
         theme_minimal(base_size = 11) + theme(legend.position = "top", axis.text.x = element_text(angle = 45, hjust = 1))
       ggsave(plot_filename_log, plot = p4, width = 10, height = 7, dpi = 150)
    }
  }
}

# --- 6. End Script ---
message("--- All plot generation finished. ---")
message("Total execution time: ", format(Sys.time() - overall_start_time))
