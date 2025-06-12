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

# 3a. Create a simplified model label for plotting
df_with_labels <- df_raw %>%
  mutate(
    # Extracts 'seedX' or 'seedXX' from the end of the model name.
    model_label = str_extract(model_run_name, "seed\\d+$")
  )

# 3b. Pivot to a long format to handle multiple values per row (value_struct1, value_struct2)
df_long <- df_with_labels %>%
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

# 3c. Isolate the 'count' data. 'count' is a property of a structure for a given model, step, and corpus.
df_counts <- df_long %>%
  filter(metric_base == "count") %>%
  select(model_run_name, model_label, seed_number, checkpoint_step, corpus_file, structure, count = value) %>%
  distinct() # Ensure counts are unique per group

# 3d. Process all other metrics.
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

# 3e. Pivot the metrics data to create columns for 'avg', 'std', and 'value'.
df_metrics_wide <- df_metrics %>%
  pivot_wider(
    names_from = summary_stat,
    values_from = value
  )

# 3f. Join the 'count' data back to the main metrics data frame.
df_wide <- df_metrics_wide %>%
  left_join(df_counts, by = c("model_run_name", "model_label", "seed_number", "checkpoint_step", "corpus_file", "structure"))

# 3g. Now, with a correctly formed 'count' column, calculate SEM.
df_wide <- df_wide %>%
  mutate(
    sem = if_else(!is.na(std) & !is.na(count) & count > 0, std / sqrt(count), NA_real_)
  ) %>%
  # Reorder columns for clarity and future use
  select(
    model_run_name, model_label, seed_number, checkpoint_step, corpus_file, contrast_pair,
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
  group_by(model_run_name, model_label, checkpoint_step, corpus_file, contrast_pair, core_metric, structure) %>%
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
dir.create(file.path(output_plot_dir, "1_timeseries_individual_models", "norm_p_with_baseline_fixed_scale"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_plot_dir, "1_timeseries_individual_models", "norm_p_with_baseline_free_scale"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_plot_dir, "1_timeseries_individual_models", "norm_p_no_baseline_fixed_scale"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_plot_dir, "1_timeseries_individual_models", "norm_p_no_baseline_free_scale"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_plot_dir, "2_timeseries_combined_models", "pe_sinclair"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_plot_dir, "2_timeseries_combined_models", "norm_p_model_grid_with_baseline"), recursive = TRUE, showWarnings = FALSE) # NEW
dir.create(file.path(output_plot_dir, "2_timeseries_combined_models", "norm_p_model_grid_no_baseline"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_plot_dir, "3_final_checkpoint_comparison", "norm_p_bar_fixed_scale"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_plot_dir, "3_final_checkpoint_comparison", "norm_p_bar_log_fixed_scale"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_plot_dir, "3_final_checkpoint_comparison", "norm_p_bar_free_scale"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_plot_dir, "3_final_checkpoint_comparison", "norm_p_bar_log_free_scale"), recursive = TRUE, showWarnings = FALSE)

# Define the annotation for the epoch line
epoch_annotation <- annotate("text", x = 117, y = Inf, label = "End of First Epoch", hjust = 1.05, vjust = 1.5, color = "grey40", size = 2.5, angle = 90)
epoch_line <- geom_vline(xintercept = 117, linetype = "dashed", color = "grey40")


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
        epoch_line + epoch_annotation + # ADDED EPOCH LINE
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
        p <- ggplot(plot_data, aes(x = checkpoint_step, y = avg, color = model_label, fill = model_label, group = model_label)) +
          geom_line(alpha = 0.7) + geom_ribbon(aes(ymin = avg - sem, ymax = avg + sem), alpha = 0.15, color = NA) +
          geom_point(size = 1, alpha = 0.5) +
          epoch_line + epoch_annotation + # ADDED EPOCH LINE
          geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
          labs(title = paste("Combined Sinclair PE for Structure:", current_structure), subtitle = paste("Corpus:", current_corpus), x = "Checkpoint Step", y = "Average pe_sinclair (+/- SEM)", color = "Model") +
          guides(color = guide_legend(nrow = 2), fill = "none") + # Readability update
          theme_minimal(base_size = 11) +
          theme(
            legend.position = "bottom",
            axis.text.x = element_text(angle = 45, hjust = 1)
          )
        ggsave(plot_filename, plot = p, width = 10, height = 8, dpi = 150) # Increased height for legend
      }
  }
}

# --- 5c. Loop: `norm_p` Time-Series (Individual Models, Fixed & Free Scales) ---
message("--- Generating INDIVIDUAL `norm_p` timeseries plots (Fixed & Free Y-Scales) ---")
for (current_model in unique_models) {
  df_model_data <- df_wide %>% filter(model_run_name == current_model)
  for (current_corpus in unique(df_model_data$corpus_file)) {
    # WITH BASELINE
    metrics_wb <- c("norm_p_conT_given_conP", "norm_p_conT_given_inconP", "baseline_pref_for_conT")
    plot_data_wb <- df_model_data %>% filter(corpus_file == current_corpus, core_metric %in% metrics_wb, !is.na(avg)) %>%
      mutate(condition = fct_relevel(fct_recode(core_metric, "Congruent" = "norm_p_conT_given_conP", "Incongruent" = "norm_p_conT_given_inconP", "Baseline" = "baseline_pref_for_conT"), "Congruent", "Incongruent", "Baseline"))
    if (nrow(plot_data_wb) > 0 && n_distinct(plot_data_wb$checkpoint_step) > 1) {
      p_base <- ggplot(plot_data_wb, aes(x = checkpoint_step, y = avg, color = condition, group = condition)) + geom_line() +
        epoch_line + epoch_annotation + # ADDED EPOCH LINE
        scale_color_manual(values = c("Congruent" = "blue", "Incongruent" = "red", "Baseline" = "black")) +
        labs(title = paste("Norm. Prob. (w/ Baseline) | Model:", current_model), subtitle = paste("Corpus:", current_corpus), x = "Checkpoint Step", y = "Avg. Normalized Probability", color = "Condition") +
        theme_minimal(base_size = 10) + theme(legend.position = "top", axis.text.x = element_text(angle = 45, hjust = 1))

      plot_folder_fixed <- file.path(output_plot_dir, "1_timeseries_individual_models", "norm_p_with_baseline_fixed_scale", sanitize_filename(current_model))
      dir.create(plot_folder_fixed, recursive = TRUE, showWarnings = FALSE)
      p_fixed <- p_base + facet_wrap(~ structure, scales = "free_x")
      ggsave(file.path(plot_folder_fixed, paste0("norm_p_w_base_fixed_", sanitize_filename(current_corpus), ".png")), plot = p_fixed, width = 9, height = 7, dpi = 150)

      plot_folder_free <- file.path(output_plot_dir, "1_timeseries_individual_models", "norm_p_with_baseline_free_scale", sanitize_filename(current_model))
      dir.create(plot_folder_free, recursive = TRUE, showWarnings = FALSE)
      p_free <- p_base + facet_wrap(~ structure, scales = "free_y")
      ggsave(file.path(plot_folder_free, paste0("norm_p_w_base_free_", sanitize_filename(current_corpus), ".png")), plot = p_free, width = 9, height = 7, dpi = 150)
    }

    # WITHOUT BASELINE
    metrics_nb <- c("norm_p_conT_given_conP", "norm_p_conT_given_inconP")
    plot_data_nb <- df_model_data %>% filter(corpus_file == current_corpus, core_metric %in% metrics_nb, !is.na(avg)) %>%
      mutate(condition = fct_recode(core_metric, "Congruent" = "norm_p_conT_given_conP", "Incongruent" = "norm_p_conT_given_inconP"))
    if (nrow(plot_data_nb) > 0 && n_distinct(plot_data_nb$checkpoint_step) > 1) {
       p_base_nb <- ggplot(plot_data_nb, aes(x = checkpoint_step, y = avg, color = condition, group = condition)) + geom_line() +
        epoch_line + epoch_annotation + # ADDED EPOCH LINE
        scale_color_manual(values = c("Congruent" = "blue", "Incongruent" = "red")) +
        labs(title = paste("Norm. Prob. (No Baseline) | Model:", current_model), subtitle = paste("Corpus:", current_corpus), x = "Checkpoint Step", y = "Avg. Normalized Probability", color = "Condition") +
        theme_minimal(base_size = 10) + theme(legend.position = "top", axis.text.x = element_text(angle = 45, hjust = 1))

      plot_folder_fixed_nb <- file.path(output_plot_dir, "1_timeseries_individual_models", "norm_p_no_baseline_fixed_scale", sanitize_filename(current_model))
      dir.create(plot_folder_fixed_nb, recursive = TRUE, showWarnings = FALSE)
      p_fixed_nb <- p_base_nb + facet_wrap(~ structure, scales = "free_x")
      ggsave(file.path(plot_folder_fixed_nb, paste0("norm_p_no_base_fixed_", sanitize_filename(current_corpus), ".png")), plot = p_fixed_nb, width = 9, height = 7, dpi = 150)

      plot_folder_free_nb <- file.path(output_plot_dir, "1_timeseries_individual_models", "norm_p_no_baseline_free_scale", sanitize_filename(current_model))
      dir.create(plot_folder_free_nb, recursive = TRUE, showWarnings = FALSE)
      p_free_nb <- p_base_nb + facet_wrap(~ structure, scales = "free_y")
      ggsave(file.path(plot_folder_free_nb, paste0("norm_p_no_base_free_", sanitize_filename(current_corpus), ".png")), plot = p_free_nb, width = 9, height = 7, dpi = 150)
    }
  }
}

# --- 5d. Combined `norm_p` Time-Series (Model Grid, With and Without Baseline) ---
message("--- Generating COMBINED `norm_p` timeseries (Model Grid) plots ---")
for (current_corpus in unique(df_wide$corpus_file)) {
  # WITH BASELINE
  metrics_wb <- c("norm_p_conT_given_conP", "norm_p_conT_given_inconP", "baseline_pref_for_conT")
  plot_data_wb <- df_wide %>% filter(corpus_file == current_corpus, core_metric %in% metrics_wb) %>%
    mutate(condition = fct_relevel(fct_recode(core_metric, "Congruent" = "norm_p_conT_given_conP", "Incongruent" = "norm_p_conT_given_inconP", "Baseline" = "baseline_pref_for_conT"), "Congruent", "Incongruent", "Baseline")) %>%
    group_by(model_run_name) %>% filter(n_distinct(checkpoint_step) > 1) %>% ungroup()
  if (nrow(plot_data_wb) > 0) {
    plot_folder <- file.path(output_plot_dir, "2_timeseries_combined_models", "norm_p_model_grid_with_baseline")
    plot_filename <- file.path(plot_folder, paste0("grid_norm_p_w_base_", sanitize_filename(current_corpus), ".png"))
    p <- ggplot(plot_data_wb, aes(x = checkpoint_step, y = avg, color = condition, group = condition)) +
      geom_line(alpha = 0.9) + epoch_line + epoch_annotation +
      facet_grid(structure ~ model_label, scales = "free_y") +
      scale_color_manual(values = c("Congruent" = "blue", "Incongruent" = "red", "Baseline" = "black")) +
      labs(title = "Combined Model Comparison: Normalized Probability of Target (with Baseline)", subtitle = paste("Corpus:", current_corpus), x = "Checkpoint Step", y = "Avg. Normalized Probability", color = "Prime Condition") +
      theme_minimal(base_size = 10) + theme(legend.position = "top", axis.text.x = element_text(angle = 90, vjust = 0.5, size=8), strip.text.y = element_text(angle = 0))
    ggsave(plot_filename, plot = p, width = 16, height = 10, dpi = 150)
  }

  # WITHOUT BASELINE
  metrics_nb <- c("norm_p_conT_given_conP", "norm_p_conT_given_inconP")
  plot_data_nb <- df_wide %>% filter(corpus_file == current_corpus, core_metric %in% metrics_nb) %>%
    mutate(condition = fct_recode(core_metric, "Congruent" = "norm_p_conT_given_conP", "Incongruent" = "norm_p_conT_given_inconP")) %>%
    group_by(model_run_name) %>% filter(n_distinct(checkpoint_step) > 1) %>% ungroup()
  if (nrow(plot_data_nb) > 0) {
    plot_folder <- file.path(output_plot_dir, "2_timeseries_combined_models", "norm_p_model_grid_no_baseline")
    plot_filename <- file.path(plot_folder, paste0("grid_norm_p_no_base_", sanitize_filename(current_corpus), ".png"))
    p <- ggplot(plot_data_nb, aes(x = checkpoint_step, y = avg, color = condition, group = condition)) +
      geom_line(alpha = 0.9) + epoch_line + epoch_annotation +
      facet_grid(structure ~ model_label, scales = "free_y") +
      scale_color_manual(values = c("Congruent" = "blue", "Incongruent" = "red")) +
      labs(title = "Combined Model Comparison: Normalized Probability of Target (No Baseline)", subtitle = paste("Corpus:", current_corpus), x = "Checkpoint Step", y = "Avg. Normalized Probability", color = "Prime Condition") +
      theme_minimal(base_size = 10) + theme(legend.position = "top", axis.text.x = element_text(angle = 90, vjust = 0.5, size=8), strip.text.y = element_text(angle = 0))
    ggsave(plot_filename, plot = p, width = 16, height = 10, dpi = 150)
  }
}


# --- 5e. Loops for Final Checkpoint Comparisons ---
message("--- Generating Final Checkpoint Comparison Plots ---")
for (current_model in unique_models) {
  df_model_summary <- df_summary_for_plots %>% filter(model_run_name == current_model)
  for (current_corpus in unique(df_model_summary$corpus_file)) {
    # Data for Norm P Bar plots
    plot_data_norm_p <- df_model_summary %>%
      filter(corpus_file == current_corpus, core_metric %in% c("norm_p_conT_given_conP", "norm_p_conT_given_inconP")) %>%
      mutate(condition = fct_recode(core_metric, "Congruent Prime" = "norm_p_conT_given_conP", "Incongruent Prime" = "norm_p_conT_given_inconP"))

    # Plot 1 & 2: Norm P Bar Plot (Fixed Scale & Free Scale)
    if (nrow(plot_data_norm_p) > 0) {
      p_base_bar <- ggplot(plot_data_norm_p, aes(x = structure, y = mean_value, fill = condition)) +
        geom_bar(stat = "identity", position = position_dodge(0.9)) +
        geom_errorbar(aes(ymin = mean_value - sem_value, ymax = mean_value + sem_value), width = 0.25, position = position_dodge(0.9)) +
        scale_fill_brewer(palette = "Set1") +
        theme_minimal(base_size = 11) + theme(legend.position = "top", axis.text.x = element_text(angle = 45, hjust = 1))

      # Fixed Scale Version
      plot_folder_fixed <- file.path(output_plot_dir, "3_final_checkpoint_comparison", "norm_p_bar_fixed_scale", sanitize_filename(current_model))
      dir.create(plot_folder_fixed, recursive = TRUE, showWarnings = FALSE)
      plot_filename_fixed <- file.path(plot_folder_fixed, paste0("norm_p_bar_fixed_", sanitize_filename(current_corpus), ".png"))
      p_fixed <- p_base_bar + facet_wrap(~ contrast_pair, scales = "free_x") +
        labs(title = "Norm. Probability (Final Ckpt, Fixed Y-Scale)", subtitle = paste("Model:", current_model, "| Corpus:", current_corpus), x = "Target Structure", y = "Mean Norm. Prob. (+/- SEM)", fill = "Prime Condition")
      ggsave(plot_filename_fixed, plot = p_fixed, width = 10, height = 7, dpi = 150)

      # Free Scale Version
      plot_folder_free <- file.path(output_plot_dir, "3_final_checkpoint_comparison", "norm_p_bar_free_scale", sanitize_filename(current_model))
      dir.create(plot_folder_free, recursive = TRUE, showWarnings = FALSE)
      plot_filename_free <- file.path(plot_folder_free, paste0("norm_p_bar_free_", sanitize_filename(current_corpus), ".png"))
      p_free <- p_base_bar + facet_wrap(~ contrast_pair, scales = "free") +
        labs(title = "Norm. Probability (Final Ckpt, Free Y-Scale)", subtitle = paste("Model:", current_model, "| Corpus:", current_corpus), x = "Target Structure", y = "Mean Norm. Prob. (+/- SEM)", fill = "Prime Condition")
      ggsave(plot_filename_free, plot = p_free, width = 10, height = 7, dpi = 150)
    }

    # Plot 3 & 4: Log-Transformed Norm P Bar Plot (Fixed & Free Scale)
    if (nrow(plot_data_norm_p) > 0) {
       p_base_log_bar <- ggplot(plot_data_norm_p, aes(x = structure, y = mean_value, fill = condition)) +
         geom_bar(stat = "identity", position = position_dodge(0.9)) +
         geom_errorbar(aes(ymin = mean_value - sem_value, ymax = mean_value + sem_value), width = 0.25, position = position_dodge(0.9)) +
         scale_y_log10(labels = label_scientific()) + scale_fill_brewer(palette = "Set1") +
         theme_minimal(base_size = 11) + theme(legend.position = "top", axis.text.x = element_text(angle = 45, hjust = 1))

       # Fixed Scale Version
       plot_folder_log_fixed <- file.path(output_plot_dir, "3_final_checkpoint_comparison", "norm_p_bar_log_fixed_scale", sanitize_filename(current_model))
       dir.create(plot_folder_log_fixed, recursive = TRUE, showWarnings = FALSE)
       plot_filename_log_fixed <- file.path(plot_folder_log_fixed, paste0("norm_p_bar_log_fixed_", sanitize_filename(current_corpus), ".png"))
       p_log_fixed <- p_base_log_bar + facet_wrap(~ contrast_pair, scales = "free_x") +
         labs(title = "Log-Scaled Norm. Prob. (Final Ckpt, Fixed Y-Scale)", subtitle = paste("Model:", current_model, "| Corpus:", current_corpus), x = "Target Structure", y = "Mean Norm. Prob. (+/- SEM, Log Scale)", fill = "Prime Condition")
       ggsave(plot_filename_log_fixed, plot = p_log_fixed, width = 10, height = 7, dpi = 150)

       # Free Scale Version
       plot_folder_log_free <- file.path(output_plot_dir, "3_final_checkpoint_comparison", "norm_p_bar_log_free_scale", sanitize_filename(current_model))
       dir.create(plot_folder_log_free, recursive = TRUE, showWarnings = FALSE)
       plot_filename_log_free <- file.path(plot_folder_log_free, paste0("norm_p_bar_log_free_", sanitize_filename(current_corpus), ".png"))
       p_log_free <- p_base_log_bar + facet_wrap(~ contrast_pair, scales = "free") +
         labs(title = "Log-Scaled Norm. Prob. (Final Ckpt, Free Y-Scale)", subtitle = paste("Model:", current_model, "| Corpus:", current_corpus), x = "Target Structure", y = "Mean Norm. Prob. (+/- SEM, Log Scale)", fill = "Prime Condition")
       ggsave(plot_filename_log_free, plot = p_log_free, width = 10, height = 7, dpi = 150)
    }
  }
}

# --- 6. End Script ---
message("--- All plot generation finished. ---")
message("Total execution time: ", format(Sys.time() - overall_start_time))
