# --- 0. Load Libraries ---
message("Loading libraries...")
# Start timer for library loading (optional)
# lib_start_time <- Sys.time()

# Use suppressPackageStartupMessages to keep console clean
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr)) # For potential use later, e.g., unnesting
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(scales)) # For label formatting

# lib_end_time <- Sys.time()
# message("Libraries loaded in ", format(lib_end_time - lib_start_time))
message("Libraries loaded.")

# --- 1. Setup & Logging ---
csv_path <- 'results.csv'
overall_start_time <- Sys.time()
message("--- Script started at ", format(overall_start_time), " ---")
message("Using input file: ", csv_path)

if (!file.exists(csv_path)) {
  stop("Error: CSV file not found at path: ", csv_path)
}

# --- 2. Read Data ---
message("Starting data read...")
read_start_time <- Sys.time()
# Consider using vroom::vroom(csv_path, col_select = c(eval_num, corpus_file, target_structure, pe))
# or data.table::fread(csv_path, select = c("eval_num", "corpus_file", "target_structure", "pe"))
# for potentially better performance on very large files. Sticking with readr for now.
df_raw <- read_csv(
  csv_path,
  col_types = cols( # Specify types for efficiency & safety
    eval_num = col_integer(),
    corpus_file = col_character(),
    target_structure = col_character(),
    item_index = col_integer(), # Read then discard if not needed
    pe = col_double(),
    logp_con = col_double(),   # Read then discard
    logp_incon = col_double() # Read then discard
  ),
  lazy = TRUE, # May help memory for some operations, readr specific
  show_col_types = FALSE
)
read_end_time <- Sys.time()
message("Data read complete (", nrow(df_raw), " rows). Time taken: ", format(read_end_time - read_start_time))

# --- 3. Pre-process & Select ---
message("Starting pre-processing...")
df_processed <- df_raw %>%
  select(eval_num, corpus_file, target_structure, pe) %>%
  # Ensure no NA in grouping columns if present, handle pe NA if needed
  filter(!is.na(eval_num) & !is.na(corpus_file) & !is.na(target_structure)) %>%
  mutate(
    target_structure = sub("^t", "", target_structure),
    # Ensure eval_num is treated as numeric for axes
    eval_num = as.numeric(eval_num)
    )
message("Pre-processing complete.")

# Remove raw data frame to free memory
rm(df_raw)
message("Raw data removed from memory.")
# Suggest garbage collection
# gc() # Uncomment if memory issues persist

# --- 4. Summarize Data ---
message("Starting data summarization...")
summary_start_time <- Sys.time()
summary_df <- df_processed %>%
  group_by(eval_num, corpus_file, target_structure) %>%
  summarise(
    mean_pe = mean(pe, na.rm = TRUE),
    sd_pe = sd(pe, na.rm = TRUE),
    n = n(),
    .groups = 'drop' # Drop grouping structure
  ) %>%
  mutate(
      sem_pe = sd_pe / sqrt(n),
      # Handle SEM calculation for n=1 or NA sd
      sem_pe = ifelse(is.na(sem_pe) | n <= 1, 0, sem_pe)
  )
summary_end_time <- Sys.time()
message("Summarization complete (", nrow(summary_df), " summary rows). Time taken: ", format(summary_end_time - summary_start_time))

# Remove processed data frame if memory is very tight
rm(df_processed)
message("Processed data removed from memory.")
# gc() # Uncomment if memory issues persist


# --- 5. Split Summary Data ---
message("Splitting summary data...")
summary_exp <- summary_df %>%
  filter(startsWith(corpus_file, "Exp"))

summary_other <- summary_df %>%
  filter(!startsWith(corpus_file, "Exp"))
message("Data split into 'Exp' (", nrow(summary_exp), " rows) and 'Other' (", nrow(summary_other), " rows).")

# Remove full summary data frame
rm(summary_df)
message("Full summary data removed from memory.")
# gc()

# --- 6. Plotting Function - Line Plots ---
plot_lines <- function(summary_data_subset, title_suffix) {
  if (nrow(summary_data_subset) == 0) {
    message("Skipping Line Plot ", title_suffix, ": No data.")
    return(NULL)
  }
  message("Creating Line Plot ", title_suffix)

  # Determine number of facets to adjust height potentially
  num_facets <- n_distinct(summary_data_subset$corpus_file)

  p <- ggplot(summary_data_subset,
              aes(x = eval_num, y = mean_pe,
                  color = target_structure, # Color lines/points
                  fill = target_structure,  # Fill ribbons
                  linetype = target_structure)) + # Different lines
    geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
    # Line for the mean
    geom_line(linewidth = 0.7) +
    # Ribbon for SEM - make it semi-transparent
    geom_ribbon(aes(ymin = mean_pe - sem_pe, ymax = mean_pe + sem_pe),
                alpha = 0.25, color = NA) + # Use alpha, remove ribbon outline
    # Facet vertically, free y-axis
    facet_wrap(~ corpus_file, scales = "free_y", ncol = 1) +
    scale_color_brewer(palette = "Set1", name = "Structure") + # Use distinct colors
    scale_fill_brewer(palette = "Set1", name = "Structure") +  # Match fill to color
    scale_linetype_discrete(name = "Structure") +
    labs(
      title = paste("Longitudinal Priming Effect (Mean PE +/- SEM)", title_suffix),
      x = "Evaluation Number",
      y = "Mean Priming Effect (PE)"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      strip.text = element_text(face = "bold", size = 9),
      legend.position = "top",
      panel.spacing.y = unit(0.5, "lines"), # Adjust vertical space between facets
      axis.text.x = element_text(size=9),
      axis.text.y = element_text(size=9)
    )
  return(p)
}

# --- 7. Plotting Function - Heatmaps ---
plot_heatmap <- function(summary_data_subset, title_suffix) {
   if (nrow(summary_data_subset) == 0) {
    message("Skipping Heatmap ", title_suffix, ": No data.")
    return(NULL)
  }
   message("Creating Heatmap ", title_suffix)

   # Find max absolute PE value for symmetrical color scale limits
   max_abs_pe <- max(abs(summary_data_subset$mean_pe), na.rm = TRUE)
   if (!is.finite(max_abs_pe)) max_abs_pe <- 1 # Handle case with no data

   p <- ggplot(summary_data_subset, aes(x = eval_num, y = corpus_file, fill = mean_pe)) +
     geom_tile(color = "grey85", linewidth = 0.1) + # Add faint tile borders
     # Facet by structure - ensures y-axis is just corpus_file
     facet_wrap(~ target_structure, ncol = 1) +
     # Diverging color scale centered at 0
     scale_fill_gradient2(
         low = muted("blue"), mid = "white", high = muted("red"),
         midpoint = 0,
         limit = c(-max_abs_pe, max_abs_pe), # Symmetrical limits
         name = "Mean PE"
     ) +
     labs(
        title = paste("Priming Effect (Mean PE) Heatmap", title_suffix),
        x = "Evaluation Number",
        y = "Corpus File"
     ) +
     theme_minimal(base_size = 10) +
     theme(
         axis.text.x = element_text(angle = 45, hjust = 1, size=8), # Angle x-axis labels if many time points
         axis.text.y = element_text(size = 8), # Adjust y-axis text size
         panel.grid = element_blank(), # Remove grid lines for cleaner heatmap
         legend.position = "right",
         strip.text = element_text(face = "bold", size = 9),
         plot.title = element_text(hjust = 0.5) # Center title
        # coord_equal() # Uncomment cautiously - might distort if axes ranges differ wildly
     )
   return(p)
}


# --- 8. Generate and Display/Save Plots ---

# Set file paths for saving (optional)
output_dir <- "plots"
if (!dir.exists(output_dir)) dir.create(output_dir)

# --- Plots for 'Exp' Corpora ---
message("--- Generating plots for 'Exp' corpora ---")
exp_plot_start_time <- Sys.time()

line_plot_exp <- plot_lines(summary_exp, "for 'Exp' Corpora")
if (!is.null(line_plot_exp)) {
  print(line_plot_exp)
  # ggsave(file.path(output_dir, "line_plot_exp.png"), line_plot_exp, width = 8, height = max(4, 0.5 * n_distinct(summary_exp$corpus_file)), limitsize = FALSE, dpi = 300)
  message("Exp Line Plot generated.")
}

heatmap_exp <- plot_heatmap(summary_exp, "for 'Exp' Corpora")
if (!is.null(heatmap_exp)) {
  print(heatmap_exp)
  # ggsave(file.path(output_dir, "heatmap_exp.png"), heatmap_exp, width = 10, height = max(4, 0.3 * n_distinct(summary_exp$corpus_file)), limitsize = FALSE, dpi = 300)
  message("Exp Heatmap generated.")
}
exp_plot_end_time <- Sys.time()
message("'Exp' plots generated. Time taken: ", format(exp_plot_end_time - exp_plot_start_time))


# --- Plots for 'Other' Corpora ---
message("--- Generating plots for Non-'Exp' corpora ---")
other_plot_start_time <- Sys.time()

line_plot_other <- plot_lines(summary_other, "for Non-'Exp' Corpora")
if (!is.null(line_plot_other)) {
  print(line_plot_other)
  # ggsave(file.path(output_dir, "line_plot_other.png"), line_plot_other, width = 8, height = max(4, 0.5 * n_distinct(summary_other$corpus_file)), limitsize = FALSE, dpi = 300)
   message("Other Line Plot generated.")
}

heatmap_other <- plot_heatmap(summary_other, "for Non-'Exp' Corpora")
if (!is.null(heatmap_other)) {
  print(heatmap_other)
  # ggsave(file.path(output_dir, "heatmap_other.png"), heatmap_other, width = 10, height = max(4, 0.3 * n_distinct(summary_other$corpus_file)), limitsize = FALSE, dpi = 300)
   message("Other Heatmap generated.")
}
other_plot_end_time <- Sys.time()
message("'Other' plots generated. Time taken: ", format(other_plot_end_time - other_plot_start_time))


# --- 9. End Script ---
overall_end_time <- Sys.time()
message("--- Script finished at ", format(overall_end_time), " ---")
message("Total execution time: ", format(overall_end_time - overall_start_time))