# --- 0. Load Libraries ---
message("Loading libraries...")
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(scales))
suppressPackageStartupMessages(library(RColorBrewer))
suppressPackageStartupMessages(library(pheatmap))
suppressPackageStartupMessages(library(grid))
# Using startsWith and grepl from base R

message("Libraries loaded.")

# --- 1. Setup & Logging ---
csv_path <- 'results_filtered.csv' # Make sure this points to your cleaned CSV if necessary
overall_start_time <- Sys.time()
current_date_time <- format(overall_start_time, "%Y-%m-%d %H:%M:%S %Z")
message("--- Script started at ", current_date_time, " ---")
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
  filter(!is.na(eval_num) & !is.na(corpus_file) & !is.na(target_structure) & !is.na(pe)) %>%
  mutate(
    target_structure = sub("^t", "", target_structure),
    eval_num = as.numeric(eval_num)
  )
message("Pre-processing complete.")
rm(df_raw); #gc()

# --- 4. Summarize Data (Includes SEM) ---
message("Starting data summarization (for all data)...")
summary_start_time <- Sys.time()
summary_df <- df_processed %>%
  group_by(eval_num, corpus_file, target_structure) %>%
  summarise(
    mean_pe = mean(pe, na.rm = TRUE),
    sd_pe = sd(pe, na.rm = TRUE),
    n_pe = n(),
    .groups = 'drop'
  ) %>%
  mutate(
    sem_pe = sd_pe / sqrt(n_pe),
    sem_pe = ifelse(n_pe <= 1 | is.na(sd_pe), 0, sem_pe)
  ) %>%
  select(-sd_pe, -n_pe) %>%
  filter(!is.na(mean_pe))

summary_end_time <- Sys.time()
message("Summarization complete (", nrow(summary_df), " summary rows). Time taken: ", format(summary_end_time - summary_start_time))
rm(df_processed); #gc()

# --- 5. Define Patterns for Filtering ---
# *** UPDATED to exclude Exp6 from 'Other' ***
exp_pattern_other_exclusion <- "^Exp[1236]" # Pattern to define the "Other" groups (excluding Exp1, 2, 3, AND 6)
lexboost_pattern <- "^LEXBOOSTFULL"
# *** UPDATED to include Exp6 in individual plots ***
target_exp_prefixes <- c("Exp1", "Exp2", "Exp3", "Exp6") # Individual experiments to plot


# --- 6. Plotting Function - Line Plots (Unchanged - Ribbon logic depends on data) ---
plot_lines <- function(summary_data_subset_line, title_suffix) {
    # (Function code remains the same)
    if (nrow(summary_data_subset_line) == 0) {
        message("Skipping Line Plot ", title_suffix, ": No data.")
        return(NULL)
    }
    summary_data_subset_line <- summary_data_subset_line %>% arrange(eval_num)
    message("Creating Line Plot ", title_suffix)
    num_facets <- n_distinct(summary_data_subset_line$corpus_file)
    show_ribbon <- any(summary_data_subset_line$sem_pe > 0, na.rm = TRUE)

    p <- ggplot(summary_data_subset_line,
                aes(x = eval_num, y = mean_pe,
                    color = target_structure, fill = target_structure, linetype = target_structure)) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
        geom_line(linewidth = 0.7)

    if (show_ribbon) {
        p <- p + geom_ribbon(aes(ymin = mean_pe - sem_pe, ymax = mean_pe + sem_pe), alpha = 0.25, color = NA)
    } else {
        message("Note: No meaningful SEM values (>0) found for line plot ", title_suffix, ". Ribbons omitted.")
    }

    p <- p +
        facet_wrap(~ corpus_file, scales = "free_y", ncol = 1) +
        scale_color_brewer(palette = "Set1", name = "Structure") +
        scale_fill_brewer(palette = "Set1", name = "Structure") +
        scale_linetype_discrete(name = "Structure") +
        scale_x_continuous(breaks = pretty_breaks()) +
        labs(
            title = paste("Longitudinal Priming Effect (Mean PE +/- SEM)", title_suffix),
            x = "Evaluation Number", y = "Mean Priming Effect (PE)" ) +
        theme_minimal(base_size = 11) +
        theme( strip.text = element_text(face = "bold", size = 9), legend.position = "top",
                panel.spacing.y = unit(0.5, "lines"), axis.text.x = element_text(size=9, angle = 0, hjust=0.5),
                axis.text.y = element_text(size=9) )
    return(p)
}

# --- 7. Plotting Function - Heatmaps (Unchanged - Row order flipped in previous version) ---
plot_heatmap_pheatmap <- function(summary_data_subset, title_suffix) {
   # (Function code remains the same - includes reversed factor levels for target_structure)
   if (nrow(summary_data_subset) == 0) {
    message("Skipping pheatmap ", title_suffix, ": No data.")
    return(NULL)
  }
   message("Creating pheatmap ", title_suffix)
   heatmap_data_wide <- summary_data_subset %>%
     mutate(target_structure = factor(target_structure, levels = rev(sort(unique(target_structure))))) %>%
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

   max_abs_pe <- max(abs(heatmap_matrix), na.rm = TRUE)
   if (!is.finite(max_abs_pe)) {
       message("Cannot determine finite max_abs_pe for heatmap ", title_suffix, ". Skipping.")
       return(NULL)
   }
   if (max_abs_pe < 1e-9) max_abs_pe <- 1e-9

   n_colors <- 101
   my_colors <- colorRampPalette(rev(brewer.pal(n = 7, name = "RdBu")))(n_colors)
   my_breaks <- seq(-max_abs_pe * 1.01, max_abs_pe * 1.01, length.out = n_colors + 1)

   if(any(duplicated(my_breaks))) {
       stop("Fatal Error: Duplicate breaks generated even with linear sequence.")
   }
   if(length(my_breaks) != n_colors + 1) {
       stop(paste("Fatal Error: Incorrect number of breaks:", length(my_breaks), "generated for", length(my_colors), "colors."))
   }

   fontsize_row <- max(4, min(8, round(200 / nrow(heatmap_matrix))))
   fontsize_col <- max(4, min(8, round(200 / ncol(heatmap_matrix))))

   p <- tryCatch({
       pheatmap(
           heatmap_matrix, color = my_colors, breaks = my_breaks,
           cluster_rows = FALSE,
           cluster_cols = FALSE,
           border_color = "grey85",
           na_col = "grey50", main = paste("Priming Effect (Mean PE) Heatmap", title_suffix),
           fontsize = 8, fontsize_row = fontsize_row, fontsize_col = fontsize_col,
           angle_col = "45", silent = TRUE
       )
   }, error = function(e) {
       message("Error during pheatmap generation for ", title_suffix, ": ", e$message)
       return(NULL)
   })
   return(p)
}


# --- 8. Generate and Display/Save Plots for Each Required Subset ---
output_dir <- "plots"
if (!dir.exists(output_dir)) dir.create(output_dir)

# --- Plotting Block 1: Individual Experiments (Exp1, Exp2, Exp3, Exp6) --- ## UPDATED ##
message("\n--- Starting plot generation loop for individual experiments (Exp1, Exp2, Exp3, Exp6) ---")
# Target prefixes defined in Step 5 now include Exp6
for (prefix in target_exp_prefixes) {
  plot_start_time_loop <- Sys.time()
  message("\n--- Processing prefix: ", prefix, " ---")
  current_summary_subset <- summary_df %>% filter(startsWith(corpus_file, prefix))

  if (nrow(current_summary_subset) == 0) {
    message("  Skipping plots for '", prefix, "' corpora: No data found.")
    next
  } else { message("  Found ", nrow(current_summary_subset), " summary rows for '", prefix, "'.") }

  plot_title_suffix <- paste0("for '", prefix, "' Corpora")
  plot_filename_base <- prefix

  # Generate Line Plot (Ribbons depend on SEM > 0)
  line_plot <- plot_lines(current_summary_subset, plot_title_suffix)
  if (!is.null(line_plot)) {
      line_filename <- file.path(output_dir, paste0("lineplot_", plot_filename_base, ".png"))
      num_facets <- n_distinct(current_summary_subset$corpus_file)
      ggsave(line_filename, plot = line_plot, width = 8, height = max(4, 2 * num_facets),
             units = "in", dpi = 150, limitsize = FALSE)
      message("  Line plot saved to ", line_filename)
  } else { message("  Line plot generation failed or skipped.") }

  # Generate Heatmap
  heatmap_grob <- plot_heatmap_pheatmap(current_summary_subset, plot_title_suffix)
  if (!is.null(heatmap_grob)) {
      heatmap_input_rows <- current_summary_subset %>% distinct(corpus_file, target_structure) %>% nrow()
      if (is.null(heatmap_input_rows) || !is.numeric(heatmap_input_rows) || heatmap_input_rows == 0) heatmap_input_rows <- 20
      heatmap_filename <- file.path(output_dir, paste0("pheatmap_", plot_filename_base, ".png"))
      png(heatmap_filename, width = 2000, height = max(800, 15 * heatmap_input_rows), res = 150)
      grid::grid.newpage(); grid::grid.draw(heatmap_grob$gtable); dev.off()
      message("  Heatmap saved to ", heatmap_filename)
  } else { message("  Heatmap generation failed or skipped.") }

  plot_end_time_loop <- Sys.time()
  message("  Plotting for '", prefix, "' complete. Time taken: ", format(plot_end_time_loop - plot_start_time_loop))
}
message("\n--- Finished individual experiment plot loop ---")


# --- Plotting Block 2: LEXBOOSTFULL Corpora ---
message("\n--- Generating plots for 'LEXBOOSTFULL' corpora ---")
plot_start_time <- Sys.time()
summary_lexboost <- summary_df %>% filter(startsWith(corpus_file, "LEXBOOSTFULL"))

if (nrow(summary_lexboost) > 0) {
    message("  Found ", nrow(summary_lexboost), " summary rows for 'LEXBOOSTFULL'.")
    plot_title_suffix <- "for 'LEXBOOSTFULL' Corpora"
    plot_filename_base <- "LEXBOOSTFULL"

    # Generate Line Plot (Ribbons depend on SEM > 0)
    line_plot <- plot_lines(summary_lexboost, plot_title_suffix)
    if (!is.null(line_plot)) {
        line_filename <- file.path(output_dir, paste0("lineplot_", plot_filename_base, ".png"))
        num_facets <- n_distinct(summary_lexboost$corpus_file)
        ggsave(line_filename, plot = line_plot, width = 8, height = max(4, 2 * num_facets),
               units = "in", dpi = 150, limitsize = FALSE)
        message("  Line plot saved to ", line_filename)
    } else { message("  Line plot generation failed or skipped.") }

    # Generate Heatmap
    heatmap_grob <- plot_heatmap_pheatmap(summary_lexboost, plot_title_suffix)
    if (!is.null(heatmap_grob)) {
        heatmap_input_rows <- summary_lexboost %>% distinct(corpus_file, target_structure) %>% nrow()
        if (is.null(heatmap_input_rows) || !is.numeric(heatmap_input_rows) || heatmap_input_rows == 0) heatmap_input_rows <- 20
        heatmap_filename <- file.path(output_dir, paste0("pheatmap_", plot_filename_base, ".png"))
        png(heatmap_filename, width = 2000, height = max(800, 15 * heatmap_input_rows), res = 150)
        grid::grid.newpage(); grid::grid.draw(heatmap_grob$gtable); dev.off()
        message("  Heatmap saved to ", heatmap_filename)
    } else { message("  Heatmap generation failed or skipped.") }

    plot_end_time <- Sys.time()
    message("  Plotting complete. Time taken: ", format(plot_end_time - plot_start_time))
    rm(summary_lexboost)
} else { message("--- Skipping plots for 'LEXBOOSTFULL' corpora: No data ---") }


# --- Plotting Block 3: All Other Corpora (Not Exp1/2/3/6) --- ## UPDATED ##
message("\n--- Generating plots for All Non-'Exp1/2/3/6' corpora ---")
plot_start_time <- Sys.time()
# Filter for everything NOT starting with Exp1, Exp2, Exp3, or Exp6
# Uses exp_pattern_other_exclusion defined in Step 5, which now includes 6
summary_other_all <- summary_df %>%
    filter(!grepl(exp_pattern_other_exclusion, corpus_file))

if (nrow(summary_other_all) > 0) {
    message("  Found ", nrow(summary_other_all), " summary rows for 'Other (All - excl. Exp1/2/3/6)'.")
    plot_title_suffix <- "for All Non-'Exp1/2/3/6' Corpora" # Updated title
    plot_filename_base <- "Other_All"

    # Generate Line Plot (Ribbons depend on SEM > 0)
    line_plot <- plot_lines(summary_other_all, plot_title_suffix)
    if (!is.null(line_plot)) {
        line_filename <- file.path(output_dir, paste0("lineplot_", plot_filename_base, ".png"))
        num_facets <- n_distinct(summary_other_all$corpus_file)
        ggsave(line_filename, plot = line_plot, width = 8, height = max(4, 2 * num_facets),
               units = "in", dpi = 150, limitsize = FALSE)
        message("  Line plot saved to ", line_filename)
    } else { message("  Line plot generation failed or skipped.") }

    # Generate Heatmap
    heatmap_grob <- plot_heatmap_pheatmap(summary_other_all, plot_title_suffix)
    if (!is.null(heatmap_grob)) {
        heatmap_input_rows <- summary_other_all %>% distinct(corpus_file, target_structure) %>% nrow()
        if (is.null(heatmap_input_rows) || !is.numeric(heatmap_input_rows) || heatmap_input_rows == 0) heatmap_input_rows <- 20
        heatmap_filename <- file.path(output_dir, paste0("pheatmap_", plot_filename_base, ".png"))
        png(heatmap_filename, width = 2000, height = max(800, 15 * heatmap_input_rows), res = 150)
        grid::grid.newpage(); grid::grid.draw(heatmap_grob$gtable); dev.off()
        message("  Heatmap saved to ", heatmap_filename)
    } else { message("  Heatmap generation failed or skipped.") }

    plot_end_time <- Sys.time()
    message("  Plotting complete. Time taken: ", format(plot_end_time - plot_start_time))
    # Keep summary_other_all for next block
} else { message("--- Skipping plots for All Non-'Exp1/2/3/6' corpora: No data ---") }


# --- Plotting Block 4: Filtered Other Corpora (excluding LEXBOOSTFULL) --- ## UPDATED ##
message("\n--- Generating plots for Non-'Exp1/2/3/6' (excluding LEXBOOSTFULL) corpora ---")
plot_start_time <- Sys.time()
if (exists("summary_other_all") && nrow(summary_other_all) > 0) {
    summary_other_filtered <- summary_other_all %>% filter(!startsWith(corpus_file, "LEXBOOSTFULL"))

    if (nrow(summary_other_filtered) > 0) {
        message("  Found ", nrow(summary_other_filtered), " summary rows for 'Other (Filtered)'.")
        plot_title_suffix <- "for Non-'Exp1/2/3/6' Corpora (excl. LEXBOOSTFULL)" # Updated title
        plot_filename_base <- "Other_Filtered"

        # Generate Line Plot (Ribbons depend on SEM > 0)
        line_plot <- plot_lines(summary_other_filtered, plot_title_suffix)
        if (!is.null(line_plot)) {
            line_filename <- file.path(output_dir, paste0("lineplot_", plot_filename_base, ".png"))
            num_facets <- n_distinct(summary_other_filtered$corpus_file)
            ggsave(line_filename, plot = line_plot, width = 8, height = max(4, 2 * num_facets),
                   units = "in", dpi = 150, limitsize = FALSE)
            message("  Line plot saved to ", line_filename)
        } else { message("  Line plot generation failed or skipped.") }

        # Generate Heatmap
        heatmap_grob <- plot_heatmap_pheatmap(summary_other_filtered, plot_title_suffix)
        if (!is.null(heatmap_grob)) {
            heatmap_input_rows <- summary_other_filtered %>% distinct(corpus_file, target_structure) %>% nrow()
            if (is.null(heatmap_input_rows) || !is.numeric(heatmap_input_rows) || heatmap_input_rows == 0) heatmap_input_rows <- 20
            heatmap_filename <- file.path(output_dir, paste0("pheatmap_", plot_filename_base, ".png"))
            png(heatmap_filename, width = 2000, height = max(800, 15 * heatmap_input_rows), res = 150)
            grid::grid.newpage(); grid::grid.draw(heatmap_grob$gtable); dev.off()
            message("  Heatmap saved to ", heatmap_filename)
        } else { message("  Heatmap generation failed or skipped.") }

        plot_end_time <- Sys.time()
        message("  Plotting complete. Time taken: ", format(plot_end_time - plot_start_time))
        rm(summary_other_filtered)
    } else { message("--- Skipping plots for Non-'Exp1/2/3/6' (excluding LEXBOOSTFULL) corpora: No data after filtering ---") }
    # Clean up the unfiltered 'other' subset if it exists
    if(exists("summary_other_all")) rm(summary_other_all)
} else { message("--- Skipping plots for Non-'Exp1/2/3/6' (excluding LEXBOOSTFULL) corpora: No base 'Other' data ---") }


# --- 9. End Script ---
if(exists("summary_df")) rm(summary_df); #gc()

overall_end_time <- Sys.time()
final_date_time <- format(overall_end_time, "%Y-%m-%d %H:%M:%S %Z")
message("\n--- Script finished at ", final_date_time, " ---")
message("Total execution time: ", format(overall_end_time - overall_start_time))