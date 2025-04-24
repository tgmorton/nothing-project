# --- 0. Load Libraries ---
message("Loading libraries...")
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(ggplot2)) # Still needed for line plots
suppressPackageStartupMessages(library(scales))
suppressPackageStartupMessages(library(RColorBrewer))
suppressPackageStartupMessages(library(pheatmap)) # Load pheatmap

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
df_raw <- read_csv(
  csv_path,
  col_types = cols(
    eval_num = col_integer(), corpus_file = col_character(),
    target_structure = col_character(), item_index = col_integer(),
    pe = col_double(), logp_con = col_double(), logp_incon = col_double()
  ),
  lazy = TRUE, show_col_types = FALSE
)
read_end_time <- Sys.time()
message("Data read complete (", nrow(df_raw), " rows). Time taken: ", format(read_end_time - read_start_time))

# --- 3. Pre-process & Select ---
message("Starting pre-processing...")
df_processed <- df_raw %>%
  select(eval_num, corpus_file, target_structure, pe) %>%
  filter(!is.na(eval_num) & !is.na(corpus_file) & !is.na(target_structure)) %>%
  mutate(
    target_structure = sub("^t", "", target_structure),
    eval_num = as.numeric(eval_num) # Ensure eval_num is numeric for sorting columns
  )
message("Pre-processing complete.")
rm(df_raw); #gc()

# --- 4. Summarize Data ---
message("Starting data summarization...")
summary_start_time <- Sys.time()
summary_df <- df_processed %>%
  group_by(eval_num, corpus_file, target_structure) %>%
  summarise(mean_pe = mean(pe, na.rm = TRUE), .groups = 'drop') # Keep only needed columns
  # Note: SEM/SD not needed for heatmap itself

# Add check for missing mean_pe values if needed
summary_df <- summary_df %>% filter(!is.na(mean_pe))

summary_end_time <- Sys.time()
message("Summarization complete (", nrow(summary_df), " summary rows). Time taken: ", format(summary_end_time - summary_start_time))
rm(df_processed); #gc()

# --- 5. Split Summary Data ---
message("Splitting summary data...")
summary_exp <- summary_df %>%
  filter(startsWith(corpus_file, "Exp"))

summary_other <- summary_df %>%
  filter(!startsWith(corpus_file, "Exp"))
message("Data split into 'Exp' (", nrow(summary_exp), " rows) and 'Other' (", nrow(summary_other), " rows).")
rm(summary_df); #gc()

# --- 6. Plotting Function - Line Plots ---
# (Keep the previous ggplot2 line plot function here)
plot_lines <- function(summary_data_subset_line, title_suffix) {
    # Add SEM calculation if not done globally
    if (!"sem_pe" %in% colnames(summary_data_subset_line)) {
        warning("SEM not pre-calculated for line plot subset, calculating now (less efficient).")
        summary_data_subset_line <- summary_data_subset_line %>%
         # Need n for SEM - requires going back or recalculating n
         # This highlights why keeping summary_df longer might be needed
         # Or pass df_processed to calculate n within groups
         # For now, assume SEM calculation needs to be done differently or kept in summary_df
         # Add dummy SEM for now to make code run, REVISE THIS if running lines plots
         mutate(sem_pe = 0)

    }

  if (nrow(summary_data_subset_line) == 0) {
    message("Skipping Line Plot ", title_suffix, ": No data.")
    return(NULL)
  }
  message("Creating Line Plot ", title_suffix)
  num_facets <- n_distinct(summary_data_subset_line$corpus_file)
  p <- ggplot(summary_data_subset_line,
              aes(x = eval_num, y = mean_pe,
                  color = target_structure, fill = target_structure, linetype = target_structure)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
    geom_line(linewidth = 0.7) +
    geom_ribbon(aes(ymin = mean_pe - sem_pe, ymax = mean_pe + sem_pe), alpha = 0.25, color = NA) +
    facet_wrap(~ corpus_file, scales = "free_y", ncol = 1) +
    scale_color_brewer(palette = "Set1", name = "Structure") +
    scale_fill_brewer(palette = "Set1", name = "Structure") +
    scale_linetype_discrete(name = "Structure") +
    labs(
      title = paste("Longitudinal Priming Effect (Mean PE +/- SEM)", title_suffix),
      x = "Evaluation Number", y = "Mean Priming Effect (PE)" ) +
    theme_minimal(base_size = 11) +
    theme( strip.text = element_text(face = "bold", size = 9), legend.position = "top",
      panel.spacing.y = unit(0.5, "lines"), axis.text.x = element_text(size=9),
      axis.text.y = element_text(size=9) )
  return(p)
}


# --- 7. Plotting Function - Heatmaps (Using pheatmap - Revised Breaks) ---
plot_heatmap_pheatmap <- function(summary_data_subset, title_suffix) {
   if (nrow(summary_data_subset) == 0) {
    message("Skipping pheatmap ", title_suffix, ": No data.")
    return(NULL)
  }
   message("Creating pheatmap ", title_suffix)

   # --- Prepare data for pheatmap ---
   heatmap_data_wide <- summary_data_subset %>%
     mutate(target_structure = factor(target_structure, levels = sort(unique(target_structure)))) %>%
     arrange(corpus_file, target_structure, eval_num) %>%
     mutate(row_label = interaction(corpus_file, target_structure, sep = ":", lex.order = TRUE)) %>%
     select(row_label, eval_num, mean_pe) %>%
     pivot_wider(names_from = eval_num, values_from = mean_pe)

   heatmap_matrix <- as.matrix(select(heatmap_data_wide, -row_label))
   rownames(heatmap_matrix) <- heatmap_data_wide$row_label

   if (nrow(heatmap_matrix) == 0 || ncol(heatmap_matrix) == 0) {
       message("Matrix for pheatmap is empty ", title_suffix, ". Skipping.")
       return(NULL)
   }

   # --- Define Colors and **Linear** Breaks ---
   max_abs_pe <- max(abs(heatmap_matrix), na.rm = TRUE)
   # Handle case where all data is NA or matrix is empty after NA removal
   if (!is.finite(max_abs_pe)) {
       message("Cannot determine finite max_abs_pe for heatmap ", title_suffix, ". Skipping.")
       return(NULL)
   }
   # Handle case where max abs PE is effectively zero
   if (max_abs_pe < 1e-9) max_abs_pe <- 1e-9 # Use a tiny range if data is all zero


   # Define number of color steps (odd number for distinct center)
   n_colors <- 101
   # Create color palette (e.g., Blue-White-Red)
   my_colors <- colorRampPalette(rev(brewer.pal(n = 7, name = "RdBu")))(n_colors)

   # Create a linear sequence of breaks: n_colors + 1 points
   # Spanning slightly beyond the max_abs_pe ensures all values are included
   my_breaks <- seq(-max_abs_pe * 1.01, max_abs_pe * 1.01, length.out = n_colors + 1)

   # --- Sanity check breaks ---
   if(any(duplicated(my_breaks))) {
       stop("Fatal Error: Duplicate breaks generated even with linear sequence.")
   }
   if(length(my_breaks) != n_colors + 1) {
       stop(paste("Fatal Error: Incorrect number of breaks:", length(my_breaks), "generated for", length(my_colors), "colors."))
   }

   # --- Create pheatmap ---
   fontsize_row <- max(4, min(8, round(200 / nrow(heatmap_matrix))))
   fontsize_col <- max(4, min(8, round(200 / ncol(heatmap_matrix))))

   # Use tryCatch to handle potential plotting errors gracefully
   p <- tryCatch({
       pheatmap(
           heatmap_matrix,
           color = my_colors,
           breaks = my_breaks,
           cluster_rows = FALSE,
           cluster_cols = FALSE,
           border_color = "grey85", #"grey85", # Set to NA for no borders
           na_col = "grey50",
           main = paste("Priming Effect (Mean PE) Heatmap", title_suffix),
           fontsize = 8,
           fontsize_row = fontsize_row,
           fontsize_col = fontsize_col,
           angle_col = "45",
           silent = TRUE # Set to FALSE for immediate plotting, TRUE if saving manually
          # cellwidth = 10, # Avoid forcing cell size unless absolutely necessary
          # cellheight = 10
       )
   }, error = function(e) {
       message("Error during pheatmap generation for ", title_suffix, ": ", e$message)
       return(NULL) # Return NULL on error
   })

   return(p) # Return the heatmap object (grob) or NULL
}

# --- Rest of the script ---
# (Make sure the summary step provides the necessary data if running line plots)
# (Update the saving/printing logic for pheatmap objects)

# --- 8. Generate and Display/Save Plots ---
output_dir <- "plots"
if (!dir.exists(output_dir)) dir.create(output_dir)

# --- Plots for 'Exp' Corpora ---
message("--- Generating plots for 'Exp' corpora ---")
exp_plot_start_time <- Sys.time()
# line_plot_exp <- plot_lines(summary_exp, "for 'Exp' Corpora") # Needs SEM
# if (!is.null(line_plot_exp)) { print(line_plot_exp) }

heatmap_exp_grob <- plot_heatmap_pheatmap(summary_exp, "for 'Exp' Corpora")
if (!is.null(heatmap_exp_grob)) {
    # Save using grid graphics functions
    png(file.path(output_dir, "pheatmap_exp.png"), width = 2000, height = max(800, 5 * nrow(summary_exp)), res = 150) # Adjust multiplier/res
    grid::grid.newpage()
    grid::grid.draw(heatmap_exp_grob$gtable) # Draw the grob
    dev.off()
    message("Exp pheatmap generated and saved.")
} else {
    message("Exp pheatmap generation failed or skipped.")
}
exp_plot_end_time <- Sys.time()
message("'Exp' plots generated. Time taken: ", format(exp_plot_end_time - exp_plot_start_time))


# --- Plots for 'Other' Corpora ---
message("--- Generating plots for Non-'Exp' corpora ---")
other_plot_start_time <- Sys.time()
# line_plot_other <- plot_lines(summary_other, "for Non-'Exp' Corpora") # Needs SEM
# if (!is.null(line_plot_other)) { print(line_plot_other) }

heatmap_other_grob <- plot_heatmap_pheatmap(summary_other, "for Non-'Exp' Corpora")
if (!is.null(heatmap_other_grob)) {
    png(file.path(output_dir, "pheatmap_other.png"), width = 2000, height = max(800, 5 * nrow(summary_other)), res = 150) # Adjust multiplier/res
    grid::grid.newpage()
    grid::grid.draw(heatmap_other_grob$gtable)
    dev.off()
    message("Other pheatmap generated and saved.")
} else {
    message("Other pheatmap generation failed or skipped.")
}
other_plot_end_time <- Sys.time()
message("'Other' plots generated. Time taken: ", format(other_plot_end_time - other_plot_start_time))


# --- 9. End Script ---
overall_end_time <- Sys.time()
message("--- Script finished at ", format(overall_end_time), " ---")
message("Total execution time: ", format(overall_end_time - overall_start_time))