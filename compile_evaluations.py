import argparse
import json
import logging
from typing import Optional

import pandas as pd
from pathlib import Path
import re
from collections import defaultdict # Import defaultdict

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Helper Function ---

def extract_step_from_path(file_path: Path) -> Optional[int]:
    """
    Extracts the checkpoint step number from the filename.
    Assumes filename format like '..._step_12345.json'.
    """
    match = re.search(r"_step_(\d+)\.json$", file_path.name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            logger.warning(f"Could not convert step number in filename {file_path.name} to integer.")
            return None
    else:
        logger.warning(f"Could not find step number pattern ('_step_NUMBER.json') in filename {file_path.name}.")
        return None

# --- Main Compilation Function ---

def compile_evaluation_summaries(
    input_dir: Path,
    json_filename_pattern: str = "evaluation_summary_step_*.json"
):
    """
    Traverses subdirectories, finds evaluation summary JSONs,
    extracts standard summaries, extracts and RESHAPES priming summaries
    by DYNAMICALLY inferring contrast pairs based on 't{label}' naming,
    compiles them into DataFrames, and saves them within the input_dir.
    """
    priming_data_list = [] # Will store reshaped priming data
    standard_data_list = []
    files_processed = 0
    files_found = 0
    max_step = -1

    logger.info(f"Starting compilation process in directory: {input_dir}")
    logger.info(f"Searching recursively for files matching: {json_filename_pattern}")
    logger.info("Priming data reshaping will dynamically infer pairs based on 't{label}' suffixes.")

    for json_path in input_dir.rglob(json_filename_pattern):
        files_found += 1
        logger.debug(f"Found potential file: {json_path}")

        checkpoint_step = extract_step_from_path(json_path)
        if checkpoint_step is None:
            logger.warning(f"Skipping file due to missing/invalid step number: {json_path}")
            continue

        max_step = max(max_step, checkpoint_step)
        logger.debug(f"Processing step {checkpoint_step} from file {json_path.name}. Current max step: {max_step}")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading or parsing JSON file {json_path}: {e}. Skipping.")
            continue # Skip to next file

        # --- Process 'priming_summary' section (Reshaped Dynamically) ---
        priming_summary = data.get("priming_summary")
        if isinstance(priming_summary, dict):
            for csv_filename, metrics_dict in priming_summary.items():
                if not isinstance(metrics_dict, dict):
                    logger.warning(
                        f"Step {checkpoint_step}, File {csv_filename}: Expected 'metrics' to be a dict, but got {type(metrics_dict)}. Skipping entry."
                    )
                    continue

                # --- Dynamic Reshaping Logic ---
                # 1. Parse all keys into base_metric and structure_label
                # Structure: {base_metric: {struct_label: value}}
                parsed_metrics = defaultdict(dict)
                base_metrics_present = set() # Keep track of base metrics found in this file

                for key, value in metrics_dict.items():
                    match = re.match(r"(.*)_([^_]+)$", key)
                    if match:
                        base_metric, struct_label = match.groups()
                        parsed_metrics[base_metric][struct_label] = value
                        base_metrics_present.add(base_metric)
                    else:
                        logger.debug(f"Step {checkpoint_step}, File {csv_filename}: Could not parse key '{key}' into base_metric and structure_label.")

                # 2. For each base metric, find associated 't' labels and form pairs if exactly two are found
                for base_metric in base_metrics_present:
                    # Get all labels associated with this base_metric in this file
                    associated_labels = parsed_metrics.get(base_metric, {}).keys()

                    # Filter for labels starting with 't' (case-insensitive check just in case)
                    t_labels = {lbl for lbl in associated_labels if lbl.lower().startswith('t')}

                    # 3. Check if exactly two 't' labels were found for this metric
                    if len(t_labels) == 2:
                        # Found a pair! Sort labels alphabetically for consistent contrast_pair naming.
                        label_list = sorted(list(t_labels))
                        s1_label = label_list[0]
                        s2_label = label_list[1]
                        contrast_label = f"{s1_label}/{s2_label}"

                        # Retrieve corresponding values
                        value1 = parsed_metrics[base_metric][s1_label]
                        value2 = parsed_metrics[base_metric][s2_label]

                        # Create the reshaped row
                        row_data = {
                            "checkpoint_step": checkpoint_step,
                            "corpus_file": csv_filename,
                            "metric_base": base_metric,
                            "contrast_pair": contrast_label,
                            "value_struct1": value1,
                            "value_struct2": value2,
                        }
                        priming_data_list.append(row_data)
                        # logger.debug(f"Created row for {base_metric}, pair {contrast_label} in file {csv_filename}")

                    elif len(t_labels) > 2:
                         # Log if more than two 't' labels are found for a metric (unexpected)
                         logger.warning(
                             f"Step {checkpoint_step}, File {csv_filename}, Metric '{base_metric}': "
                             f"Found {len(t_labels)} structure labels starting with 't' ({t_labels}), expected 2. "
                             f"Cannot automatically determine pair. Skipping pair generation for this metric."
                         )
                    # If len(t_labels) is 0 or 1, we simply don't form a pair, no warning needed unless debugging.
                    # elif len(t_labels) == 1:
                    #      logger.debug(f"Step {checkpoint_step}, File {csv_filename}, Metric '{base_metric}': Found only one 't' label: {t_labels}. No pair formed.")

                # --- End Dynamic Reshaping Logic ---

        elif priming_summary is not None:
            logger.warning(
                f"Step {checkpoint_step}: Expected 'priming_summary' to be a dict, "
                f"but got {type(priming_summary)}. Skipping section."
            )

        # --- Process 'standard_summary' section (Unchanged) ---
        standard_summary = data.get("standard_summary")
        if isinstance(standard_summary, dict):
            row_data = {"checkpoint_step": checkpoint_step}
            row_data.update(standard_summary)
            standard_data_list.append(row_data)
        elif standard_summary is not None:
            logger.warning(
                f"Step {checkpoint_step}: Expected 'standard_summary' to be a dict, "
                f"but got {type(standard_summary)}. Skipping section."
            )

        files_processed += 1

    logger.info(f"Found {files_found} files matching pattern.")
    if files_processed == 0:
        logger.warning("No valid evaluation summary files were processed. No CSV files will be created.")
        return

    logger.info(f"Successfully processed data from {files_processed} files. Highest checkpoint step found: {max_step}")

    # --- Determine Output Filenames and Save DataFrames ---
    output_suffix = f"upto_step_{max_step}.csv"
    output_priming_csv = input_dir / f"compiled_priming_summary_reshaped_{output_suffix}"
    output_standard_csv = input_dir / f"compiled_standard_summary_{output_suffix}"

    # Priming Summary DataFrame (Reshaped)
    if priming_data_list:
        try:
            priming_df = pd.DataFrame(priming_data_list)
            priming_df = priming_df.sort_values(
                by=["checkpoint_step", "corpus_file", "contrast_pair", "metric_base"]
            )
            logger.info(f"Created RESHAPED Priming Summary DataFrame with {len(priming_df)} rows and {len(priming_df.columns)} columns.")
            priming_df.to_csv(output_priming_csv, index=False, encoding='utf-8')
            logger.info(f"Successfully saved RESHAPED Priming Summary data to: {output_priming_csv}")
        except Exception as e:
            logger.error(f"Failed to create or save RESHAPED Priming Summary DataFrame: {e}")
    else:
        logger.warning("No data found for RESHAPED Priming Summary despite processing files. CSV file will not be created.")

    # Standard Summary DataFrame (Unchanged)
    if standard_data_list:
        try:
            standard_df = pd.DataFrame(standard_data_list)
            standard_df = standard_df.sort_values(by=["checkpoint_step"])
            logger.info(f"Created Standard Summary DataFrame with {len(standard_df)} rows and {len(standard_df.columns)} columns.")
            standard_df.to_csv(output_standard_csv, index=False, encoding='utf-8')
            logger.info(f"Successfully saved Standard Summary data to: {output_standard_csv}")
        except Exception as e:
            logger.error(f"Failed to create or save Standard Summary DataFrame: {e}")
    else:
        logger.warning("No data found for Standard Summary despite processing files. CSV file will not be created.")

    logger.info("Compilation process finished.")

# --- Command Line Argument Parsing ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile evaluation summary JSON files from multiple checkpoint runs into CSVs. "
                    "Priming data is reshaped into a 'long' format by dynamically inferring contrast pairs (labels starting with 't'). "
                    "Output CSVs are saved in the input directory."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="The root directory containing the evaluation output folders or JSON files. "
             "The script will search recursively. Output CSVs will also be saved here."
    )
    parser.add_argument(
        "--filename_pattern",
        type=str,
        default="evaluation_summary_step_*.json",
        help="Glob pattern used to find the JSON summary files within the input directory."
    )

    args = parser.parse_args()
    input_path = Path(args.input_dir)

    if not input_path.is_dir():
        logger.error(f"Input directory not found or is not a directory: {input_path}")
        exit(1)

    compile_evaluation_summaries(
        input_dir=input_path,
        json_filename_pattern=args.filename_pattern
    )