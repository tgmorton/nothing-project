# --- 0. Load Libraries ---
message("Loading libraries...")
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(stringr))
message("Libraries loaded.")

# --- 1. Setup & Logging ---
csv_path <- '/Users/thomasmorton/ModelFolder/analysis/May14_10m_s42_compiled_priming_summary_reshaped_upto_step_129.csv' # <<< IMPORTANT: Replace
output_plot_dir <- "analysis/plots_priming/may14_10m_s42_logp_comparison" # <<< Main directory for all plots

overall_start_time <- Sys.time()
current_date_time <- format(overall_start_time, "%Y-%m-%d %H:%M:%S %Z")
message("--- Script started at ", current_date_time, " ---")
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
  name_str <- gsub(paste0(replacement, replacement, "+"), replacement, name_str) # Replace multiple replacements with single
  return(name_str)
}


# --- 2. Read Data ---
message("Starting data read...")
read_start_time <- Sys.time()
df_raw <- read_csv(
  csv_path,
  col_types = cols(
    checkpoint_step = col_integer(),
    corpus_file = col_character(),
    metric_base = col_character(),
    contrast_pair = col_character(),
    value_struct1 = col_double(),
    value_struct2 = col_double()
  ),
  lazy = TRUE,
  show_col_types = FALSE
)
read_end_time <- Sys.time()
message("Data read complete (", nrow(df_raw), " rows). Time taken: ", format(read_end_time - read_start_time))

# --- 3. Pre-process & Transform Data (Long Format) ---
message("Starting initial data transformation (to long format)...")
transform_start_time_1 <- Sys.time()
required_cols <- c("checkpoint_step", "corpus_file", "metric_base", "contrast_pair", "value_struct1", "value_struct2")
if (!all(required_cols %in% names(df_raw))) {
  stop("Error: Missing one or more required columns: ", paste(setdiff(required_cols, names(df_raw)), collapse=", "))
}
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
  ) %>%
  select(
    checkpoint_step, corpus_file, metric_base, contrast_pair, structure, value
  ) %>%
  filter(!is.na(structure)) %>%
  arrange(checkpoint_step, corpus_file, metric_base, contrast_pair, structure)
transform_end_time_1 <- Sys.time()
message("Initial transformation complete (", nrow(df_long), " rows in long format). Time taken: ", format(transform_end_time_1 - transform_start_time_1))

# --- 4. Reshape Data (Wide Format by Summary Stat) ---
message("Reshaping data (to wide format by summary statistic)...")
transform_start_time_2 <- Sys.time()
df_wide_summary <- df_long %>%
  separate(metric_base, into = c("summary_stat", "core_metric"), sep = "_", extra = "merge", remove = TRUE) %>%
  filter(!is.na(summary_stat), !is.na(core_metric), summary_stat %in% c("avg", "sem", "std", "count")) %>%
  pivot_wider(
    names_from = summary_stat,
    values_from = value
  ) %>%
  arrange(checkpoint_step, corpus_file, core_metric, contrast_pair, structure)
transform_end_time_2 <- Sys.time()
message("Reshaping to wide format complete (", nrow(df_wide_summary), " rows). Time taken: ", format(transform_end_time_2 - transform_start_time_2))

# --- 5. Inspect Transformed Data (Wide Format) ---
message("Showing the first few rows of the final wide transformed data:")
print(head(df_wide_summary))
message("Structure of the final wide transformed data:")
glimpse(df_wide_summary)

# --- 6. Generate Plots (Using Wide Data) ---
message("Generating plots using wide data...")
plot_start_time <- Sys.time()

# --- 6a. Loop: Average PE over Time (+/- SEM) for each Corpus File ---
# This section is kept from your original script for PE plots.
# It will now save plots into subdirectories named after the corpus_file.
message("--- Starting loop: Generating PE plots per corpus file ---")
unique_corpus_files_pe <- unique(df_wide_summary$corpus_file)

for (current_corpus in unique_corpus_files_pe) {
  message("Processing PE for corpus file: ", current_corpus)
  safe_corpus_name <- sanitize_filename(current_corpus)
  corpus_specific_plot_dir <- file.path(output_plot_dir, safe_corpus_name)
  if (!dir.exists(corpus_specific_plot_dir)) {
    message("  Creating plot subdirectory for PE: ", corpus_specific_plot_dir)
    dir.create(corpus_specific_plot_dir, recursive = TRUE, showWarnings = FALSE)
  }

  plot_data_pe_current <- df_wide_summary %>%
    filter(corpus_file == current_corpus, core_metric == "PE") %>%
    filter(!is.na(avg) & is.numeric(avg) & !is.na(sem) & is.numeric(sem))

  if (nrow(plot_data_pe_current) > 0 && n_distinct(plot_data_pe_current$checkpoint_step) > 1) {
    plot_filename <- file.path(corpus_specific_plot_dir, paste0("avg_pe_sem_", safe_corpus_name, ".png"))
    plot_title <- paste("Average PE (+/- SEM) over Checkpoints")
    plot_subtitle <- paste("Corpus:", current_corpus)
    message("  Generating PE plot: ", plot_filename)

    p_pe_time_corpus <- ggplot(plot_data_pe_current, aes(x = checkpoint_step, y = avg, color = structure, fill = structure, group = structure)) +
      geom_line(alpha = 0.9) +
      geom_ribbon(aes(ymin = avg - pmax(0, sem), ymax = avg + pmax(0, sem)), alpha = 0.2, color = NA) +
      geom_point(size = 1.5) +
      labs(
        title = plot_title,
        subtitle = plot_subtitle,
        x = "Checkpoint Step",
        y = "Average PE Value",
        color = "Structure",
        fill = "Structure"
      ) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
      theme_minimal(base_size = 10) +
      theme(legend.position = "top", axis.text.x = element_text(angle = 45, hjust = 1))
    ggsave(plot_filename, plot = p_pe_time_corpus, width = 8, height = 6, dpi = 150)
    message("  Saved PE plot: ", plot_filename)
  } else {
    message("  Skipping PE plot for '", current_corpus, "': Not enough distinct checkpoint steps or missing avg/sem data.")
  }
}
message("--- Finished loop: PE plots per corpus file ---")


# --- 6b. NEW Loop: LogP_baseline, LogP_con, LogP_incon over Time per Corpus & Structure ---
message("--- Starting loop: Generating LogP Comparison plots per corpus and structure ---")

# Ensure required columns for LogP plots are present
logp_metrics <- c("LogP_baseline", "LogP_con", "LogP_incon")
required_logp_cols <- c("checkpoint_step", "corpus_file", "structure", "core_metric", "avg", "sem")

# Check if all LogP metrics are even present in the data
available_core_metrics <- unique(df_wide_summary$core_metric)
if (!all(logp_metrics %in% available_core_metrics)) {
  message("Warning: Not all required LogP metrics (LogP_baseline, LogP_con, LogP_incon) are present in the 'core_metric' column.")
  message("Available core_metrics: ", paste(available_core_metrics, collapse=", "))
  message("Skipping LogP Comparison plots if essential metrics are missing.")
} else {
  unique_corpus_files_logp <- unique(df_wide_summary$corpus_file)

  for (current_corpus in unique_corpus_files_logp) {
    message("Processing LogP for corpus file: ", current_corpus)
    safe_corpus_name <- sanitize_filename(current_corpus)
    corpus_specific_plot_dir <- file.path(output_plot_dir, safe_corpus_name) # Same subdir as PE plots for this corpus
    if (!dir.exists(corpus_specific_plot_dir)) {
      message("  Creating plot subdirectory for LogP: ", corpus_specific_plot_dir) # Should be created by PE loop if it ran for this corpus
      dir.create(corpus_specific_plot_dir, recursive = TRUE, showWarnings = FALSE)
    }

    corpus_data <- df_wide_summary %>%
      filter(corpus_file == current_corpus, core_metric %in% logp_metrics) %>%
      filter(!is.na(avg) & is.numeric(avg) & !is.na(sem) & is.numeric(sem)) %>%
      mutate(
        condition = factor(
          case_when(
            core_metric == "LogP_baseline" ~ "Baseline",
            core_metric == "LogP_con" ~ "Same Prime",
            core_metric == "LogP_incon" ~ "Different Prime",
            TRUE ~ NA_character_
          ),
          levels = c("Baseline", "Same Prime", "Different Prime") # Order for legend
        )
      ) %>%
      filter(!is.na(condition)) # Remove any rows that didn't match

    if (nrow(corpus_data) == 0) {
      message("  No LogP data (baseline, con, incon) found for corpus: ", current_corpus)
      next # Skip to the next corpus
    }

    unique_structures_in_corpus <- unique(corpus_data$structure)

    for (current_structure_name in unique_structures_in_corpus) {
      message("  Processing structure: ", current_structure_name)
      plot_data_structure_logp <- corpus_data %>%
        filter(structure == current_structure_name)

      # Check if there's enough data for plotting (at least 2 distinct checkpoints for a line, for each condition)
      data_summary_for_plot <- plot_data_structure_logp %>%
        group_by(condition) %>%
        summarise(distinct_checkpoints = n_distinct(checkpoint_step), .groups = 'drop')

      if (nrow(plot_data_structure_logp) > 0 && all(data_summary_for_plot$distinct_checkpoints > 1) && n_distinct(plot_data_structure_logp$condition) == 3) {
        safe_structure_name <- sanitize_filename(current_structure_name)
        plot_filename <- file.path(corpus_specific_plot_dir, paste0("logp_comparison_", safe_corpus_name, "_struct_", safe_structure_name, ".png"))
        plot_title <- paste("Log Probability Comparison for Structure:", current_structure_name)
        plot_subtitle <- paste("Corpus:", current_corpus)
        message("    Generating LogP plot: ", plot_filename)

        p_logp_struct_time <- ggplot(plot_data_structure_logp, aes(x = checkpoint_step, y = avg, color = condition, fill = condition, group = condition)) +
          geom_line(alpha = 0.9) +
          geom_ribbon(aes(ymin = avg - pmax(0, sem), ymax = avg + pmax(0, sem)), alpha = 0.2, color = NA) +
          geom_point(size = 1.5) +
          labs(
            title = plot_title,
            subtitle = plot_subtitle,
            x = "Checkpoint Step",
            y = "Average Log Probability",
            color = "Condition",
            fill = "Condition"
          ) +
          scale_color_manual(values = c("Baseline" = "black", "Same Prime" = "blue", "Different Prime" = "red")) +
          scale_fill_manual(values = c("Baseline" = "grey50", "Same Prime" = "blue", "Different Prime" = "red")) +
          theme_minimal(base_size = 10) +
          theme(legend.position = "top", axis.text.x = element_text(angle = 45, hjust = 1))

        ggsave(plot_filename, plot = p_logp_struct_time, width = 8, height = 6, dpi = 150)
        message("    Saved LogP plot: ", plot_filename)

      } else {
        message("    Skipping LogP plot for structure '", current_structure_name, "' in corpus '", current_corpus, "': Not enough distinct checkpoint steps for all conditions or missing conditions.")
        # print(data_summary_for_plot) # for debugging
      }
    } # End of loop through structures
  } # End of loop through corpus files for LogP
} # End of else block (if all logp_metrics are present)

message("--- Finished loop: LogP Comparison plots ---")


# --- (Original 6b, now 6c, example LogP comparison plot - can be kept or removed) ---
# message("--- Generating comparison plot for LogP_con at Checkpoint 400 ---")
# ... (rest of your original example plot code, ensuring paths are updated if kept) ...

plot_end_time <- Sys.time()
message("Plotting finished. Time taken: ", format(plot_end_time - plot_start_time))

# --- 7. End Script ---
overall_end_time <- Sys.time()
final_date_time <- format(overall_end_time, "%Y-%m-%d %H:%M:%S %Z")
message("\n--- Script finished at ", final_date_time, " ---")
message("Total execution time: ", format(overall_end_time - overall_start_time))