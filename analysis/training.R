# --- 0. Load Libraries ---
message("Loading libraries...")
# Install packages if you haven't already:
# install.packages(c("readr", "dplyr", "ggplot2", "tidyr"))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr)) # For pivot_longer if plotting both metrics together
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(scales)) # For better axis breaks/labels
message("Libraries loaded.")

# --- 1. Setup & Logging ---
# Define the path to your input training statistics CSV file
csv_path <- '/Users/thomasmorton/ModelFolder/src/.output/gpt2_p6000_sif_local_eval_run/compiled_standard_summary_upto_step_12800.csv' # <<< Path to your training stats file
# Define path for output plots
output_plot_dir <- "analysis/plots_training" # Use a different directory name to avoid overwriting priming plots

overall_start_time <- Sys.time()
current_date_time <- format(overall_start_time, "%Y-%m-%d %H:%M:%S %Z")
message("--- Script started at ", current_date_time, " ---")
message("Using input file: ", csv_path)

# Check if input file exists
if (!file.exists(csv_path)) {
   stop("Error: Input CSV file not found at path: ", csv_path)
}

# Create output directory if it doesn't exist
if (!dir.exists(output_plot_dir)) {
  message("Creating output directory: ", output_plot_dir)
  dir.create(output_plot_dir)
}

# --- 2. Read Data ---
message("Starting data read...")
read_start_time <- Sys.time()

# Read the training statistics CSV
df_train_stats <- read_csv(
  csv_path,
  col_types = cols(
    checkpoint_step = col_integer(),
    loss = col_double(),
    perplexity = col_double(),
    total_items = col_integer() # Read this column even if not plotted directly
  ),
  show_col_types = FALSE
)

read_end_time <- Sys.time()
message("Data read complete (", nrow(df_train_stats), " rows). Time taken: ", format(read_end_time - read_start_time))

# --- 3. Inspect and Prepare Data ---
message("Inspecting data...")
print(head(df_train_stats))
glimpse(df_train_stats)

# Ensure checkpoint_step is numeric for plotting axes
df_train_stats <- df_train_stats %>%
  mutate(checkpoint_step = as.numeric(checkpoint_step)) %>%
  # Optional: Filter out any potential NA values if necessary
  filter(!is.na(checkpoint_step) & !is.na(loss) & !is.na(perplexity)) %>%
  # Arrange by checkpoint step for line plots
  arrange(checkpoint_step)

message("Data prepared for plotting.")


# --- 4. Generate Plots ---
message("Generating plots...")
plot_start_time <- Sys.time()

# --- 4a. Plot Loss vs. Checkpoint Step ---
message("  Generating Loss plot...")
plot_loss <- ggplot(df_train_stats, aes(x = checkpoint_step, y = loss)) +
  geom_line(color = "steelblue") +
  geom_point(color = "steelblue", size = 1.5) +
  scale_x_continuous(breaks = pretty_breaks(n=10)) + # Adjust number of breaks if needed
  scale_y_continuous(breaks = pretty_breaks(n=8)) +
  labs(
    title = "Training Loss over Checkpoints",
    x = "Checkpoint Step",
    y = "Loss"
  ) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Save the loss plot
loss_plot_filename <- file.path(output_plot_dir, "training_loss_over_checkpoints.png")
ggsave(loss_plot_filename, plot = plot_loss, width = 8, height = 5, dpi = 150)
message("  Saved plot: ", loss_plot_filename)

# --- 4b. Plot Perplexity vs. Checkpoint Step ---
message("  Generating Perplexity plot...")
plot_perplexity <- ggplot(df_train_stats, aes(x = checkpoint_step, y = perplexity)) +
  geom_line(color = "darkorange") +
  geom_point(color = "darkorange", size = 1.5) +
  scale_x_continuous(breaks = pretty_breaks(n=10)) +
  scale_y_continuous(breaks = pretty_breaks(n=8)) +
  labs(
    title = "Training Perplexity over Checkpoints",
    x = "Checkpoint Step",
    y = "Perplexity"
  ) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Save the perplexity plot
perplexity_plot_filename <- file.path(output_plot_dir, "training_perplexity_over_checkpoints.png")
ggsave(perplexity_plot_filename, plot = plot_perplexity, width = 8, height = 5, dpi = 150)
message("  Saved plot: ", perplexity_plot_filename)

# --- 4c. Optional: Plot Loss and Perplexity Together (using facets) ---
# This requires pivoting the data longer
message("  Generating combined Loss & Perplexity plot (faceted)...")
df_train_stats_long <- df_train_stats %>%
  select(checkpoint_step, loss, perplexity) %>%
  pivot_longer(
    cols = c(loss, perplexity),
    names_to = "metric",
    values_to = "value"
  )

plot_combined_facet <- ggplot(df_train_stats_long, aes(x = checkpoint_step, y = value, color = metric)) +
  geom_line() +
  geom_point(size = 1.5) +
  facet_wrap(~ metric, scales = "free_y", ncol = 1) + # Separate rows, free y-axis
  scale_x_continuous(breaks = pretty_breaks(n=10)) +
  scale_y_continuous(breaks = pretty_breaks(n=8)) +
  scale_color_manual(values = c("loss" = "steelblue", "perplexity" = "darkorange")) +
  labs(
    title = "Training Metrics over Checkpoints",
    x = "Checkpoint Step",
    y = "Value"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none", # Color is shown by facet title
        strip.text = element_text(face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1))

# Save the combined plot
combined_plot_filename <- file.path(output_plot_dir, "training_metrics_combined_facet.png")
ggsave(combined_plot_filename, plot = plot_combined_facet, width = 8, height = 7, dpi = 150)
message("  Saved plot: ", combined_plot_filename)


plot_end_time <- Sys.time()
message("Plotting finished. Time taken: ", format(plot_end_time - plot_start_time))

# --- 5. End Script ---
# Clean up intermediate objects if desired
# rm(df_train_stats, df_train_stats_long, plot_loss, plot_perplexity, plot_combined_facet)
# gc() # Garbage collection

overall_end_time <- Sys.time()
final_date_time <- format(overall_end_time, "%Y-%m-%d %H:%M:%S %Z")
message("\n--- Script finished at ", final_date_time, " ---")
message("Total execution time: ", format(overall_end_time - overall_start_time))
