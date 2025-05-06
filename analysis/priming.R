# --- 0. Load Libraries ---
message("Loading libraries...")
# Install packages if you haven't already:
# install.packages(c("readr", "dplyr", "tidyr", "ggplot2", "stringr"))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(stringr)) # Included in tidyverse, but explicit loading is fine
message("Libraries loaded.")

# --- 1. Setup & Logging ---
# Define the path to your input CSV file
csv_path <- '/Users/thomasmorton/ModelFolder/src/.output/gpt2_p6000_sif_local_eval_run/compiled_priming_summary_reshaped_upto_step_12800.csv' # <<< IMPORTANT: Replace with the actual path to your CSV file
# Define path for potential output (e.g., transformed data)
output_csv_path <- 'transformed_summary_wide.csv' # Updated output filename
output_plot_dir <- "analysis/plots_priming"

overall_start_time <- Sys.time()
current_date_time <- format(overall_start_time, "%Y-%m-%d %H:%M:%S %Z")
message("--- Script started at ", current_date_time, " ---")
message("Using input file: ", csv_path)

# Check if input file exists
if (!file.exists(csv_path)) {
  # If using the sample data below, comment out or remove this stop() call
   stop("Error: Input CSV file not found at path: ", csv_path)
  # message("Warning: Input CSV file not found. Using internal sample data for demonstration.")
}

# Create output directory if it doesn't exist
if (!dir.exists(output_plot_dir)) {
  message("Creating output directory: ", output_plot_dir)
  dir.create(output_plot_dir)
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
  lazy = TRUE, # Set lazy = FALSE if you encounter issues
  show_col_types = FALSE
)

read_end_time <- Sys.time()
message("Data read complete (", nrow(df_raw), " rows). Time taken: ", format(read_end_time - read_start_time))

# --- 3. Pre-process & Transform Data (Long Format) ---
message("Starting initial data transformation (to long format)...")
transform_start_time_1 <- Sys.time()

# Check if required columns exist
required_cols <- c("checkpoint_step", "corpus_file", "metric_base", "contrast_pair", "value_struct1", "value_struct2")
if (!all(required_cols %in% names(df_raw))) {
  stop("Error: Missing one or more required columns in the input data: ", paste(setdiff(required_cols, names(df_raw)), collapse=", "))
}

# Separate contrast_pair and pivot longer
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
# rm(df_raw) # Optional: remove raw data frame

# --- 4. Reshape Data (Wide Format by Summary Stat) ---
message("Reshaping data (to wide format by summary statistic)...")
transform_start_time_2 <- Sys.time()

df_wide_summary <- df_long %>%
  # Separate metric_base into summary stat type and core metric name
  # Assumes format like "avg_MetricName", "std_MetricName"
  # 'extra = "merge"' ensures that if MetricName contains '_', it stays together
  separate(metric_base, into = c("summary_stat", "core_metric"), sep = "_", extra = "merge", remove = TRUE) %>%
  # Filter out any rows where separation might have failed (e.g., unexpected metric_base format)
  filter(!is.na(summary_stat), !is.na(core_metric), summary_stat %in% c("avg", "sem", "std", "count")) %>%
  # Pivot wider: make avg, sem, std, count into columns
  pivot_wider(
    names_from = summary_stat, # Column containing the new column names (avg, sem, std, count)
    values_from = value        # Column containing the values for the new columns
  ) %>%
  # Arrange for better readability
  arrange(checkpoint_step, corpus_file, core_metric, contrast_pair, structure)

transform_end_time_2 <- Sys.time()
message("Reshaping to wide format complete (", nrow(df_wide_summary), " rows). Time taken: ", format(transform_end_time_2 - transform_start_time_2))
# rm(df_long) # Optional: remove intermediate long data frame

# --- 5. Inspect Transformed Data (Wide Format) ---
message("Showing the first few rows of the final wide transformed data:")
print(head(df_wide_summary))
message("Structure of the final wide transformed data:")
glimpse(df_wide_summary) # Use glimpse for a concise structure overview

# Optional: Save the transformed data
message("Saving final wide transformed data to: ", output_csv_path)
write_csv(df_wide_summary, output_csv_path)

# --- 6. Generate Plots (Using Wide Data) ---
# This section generates plots based on the df_wide_summary data frame.
message("Generating plots using wide data...")
plot_start_time <- Sys.time()

# --- 6a. Loop: Average PE over Time (+/- SEM) for each Corpus File ---
message("--- Starting loop: Generating PE plots per corpus file ---")

# Get unique corpus file names
unique_corpus_files <- unique(df_wide_summary$corpus_file)

# Loop through each unique corpus file
for (current_corpus in unique_corpus_files) {
  message("Processing corpus file: ", current_corpus)

  # Filter data for the current corpus file and PE metric
  plot_data_pe_current <- df_wide_summary %>%
    filter(corpus_file == current_corpus, core_metric == "PE") %>%
    # Ensure avg and sem columns exist and are numeric for plotting
    filter(!is.na(avg) & is.numeric(avg) & !is.na(sem) & is.numeric(sem))

  # Check if there's enough data (at least 2 distinct checkpoints for a line)
  if (nrow(plot_data_pe_current) > 0 && n_distinct(plot_data_pe_current$checkpoint_step) > 1) {

    # Sanitize corpus file name for use in filename
    # Replace common problematic characters with underscores
    safe_corpus_name <- gsub("[^a-zA-Z0-9_.-]", "_", current_corpus)
    # Limit length if necessary (optional)
    # max_len <- 100
    # if (nchar(safe_corpus_name) > max_len) {
    #   safe_corpus_name <- substr(safe_corpus_name, 1, max_len)
    # }

    plot_filename <- file.path(output_plot_dir, paste0("avg_pe_sem_", safe_corpus_name, ".png"))
    plot_title <- paste("Average PE (+/- SEM) over Checkpoints")
    plot_subtitle <- paste("Corpus:", current_corpus)

    message("  Generating plot: ", plot_filename)

    p_pe_time_corpus <- ggplot(plot_data_pe_current, aes(x = checkpoint_step, y = avg, color = structure, fill = structure)) +
      # Use group = structure explicitly if interaction() was needed before, but color/fill usually handles it
      geom_line(alpha = 0.9, aes(group = structure)) +
      # Add SEM ribbon - ensure SEM is non-negative
      geom_ribbon(aes(ymin = avg - pmax(0, sem), ymax = avg + pmax(0, sem), group = structure), alpha = 0.2, color = NA) +
      geom_point(size = 1.5, aes(group = structure)) +
      # Removed facet_wrap as we are plotting per corpus file
      labs(
        title = plot_title,
        subtitle = plot_subtitle,
        x = "Checkpoint Step",
        y = "Average PE Value",
        color = "Structure",
        fill = "Structure"
      ) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") + # Add line at y=0
      theme_minimal(base_size = 10) +
      theme(legend.position = "top", axis.text.x = element_text(angle = 45, hjust = 1))

    # Save the plot
    ggsave(plot_filename, plot = p_pe_time_corpus, width = 8, height = 6, dpi = 150)
    message("  Saved plot: ", plot_filename)

  } else {
    message("  Skipping PE plot for '", current_corpus, "': Not enough distinct checkpoint steps or missing avg/sem data.")
  }
} # End of loop through corpus files
message("--- Finished loop: PE plots per corpus file ---")


# --- 6b. Example: Comparing average LogP_con structures at a single checkpoint ---
# (Keeping this example as it wasn't requested to be removed)
message("--- Generating comparison plot for LogP_con at Checkpoint 400 ---")
plot_data_logp_compare <- df_wide_summary %>%
  filter(checkpoint_step == 400, core_metric == "LogP_con") %>%
  filter(!is.na(avg) & is.numeric(avg)) # Ensure avg exists

if (nrow(plot_data_logp_compare) > 0) {
  message("Generating plot: Average LogP_con comparison at Checkpoint 400")
  p_logp_compare <- ggplot(plot_data_logp_compare, aes(x = structure, y = avg, fill = structure)) +
    geom_col(position = "dodge") +
    # Optional: Add error bars if std or sem is relevant and available
    # geom_errorbar(aes(ymin = avg - sem, ymax = avg + sem), width = 0.2, position = position_dodge(0.9)) +
    facet_wrap(~ corpus_file, scales = "free_x", ncol = 2) + # Facet by corpus file
    labs(
      title = "Average LogP_con Comparison at Checkpoint 400",
      subtitle = "Comparing structures within each corpus file",
      x = "Structure",
      y = "Average LogP_con Value",
      fill = "Structure"
    ) +
    theme_minimal(base_size = 10) +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 45, hjust = 1),
          strip.text = element_text(size = 8))

  # print(p_logp_compare) # Display the plot
  plot_filename_logp <- file.path(output_plot_dir, "example_avg_logp_con_comparison_cp400.png")
  ggsave(plot_filename_logp, plot = p_logp_compare, width = 10, height = 6, dpi = 150)
   message("Saved plot: ", plot_filename_logp)

} else {
  message("Skipping 'Average LogP_con comparison at Checkpoint 400' plot: No data found for the specified filters.")
}

plot_end_time <- Sys.time()
message("Plotting finished. Time taken: ", format(plot_end_time - plot_start_time))


# --- 7. End Script ---
# Clean up intermediate objects if desired
# rm(df_raw, df_long, df_wide_summary, plot_data_pe_wide, plot_data_logp_compare)
# gc() # Garbage collection

overall_end_time <- Sys.time()
final_date_time <- format(overall_end_time, "%Y-%m-%d %H:%M:%S %Z")
message("\n--- Script finished at ", final_date_time, " ---")
message("Total execution time: ", format(overall_end_time - overall_start_time))
