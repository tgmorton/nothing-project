import argparse
import json
import logging
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

# --- Define Known Structure Pairs ---
# The script will look for these pairs in the data.
# The first element in the tuple will correspond to 'value_struct1', the second to 'value_struct2'.
STRUCTURE_PAIRS = [
    ('tdo', 'tpo'),
    ('pdo', 'ppo'),
    ('ta', 'tp'),
    # Add other pairs if they exist in your data, e.g., ('da', 'dp')
]
logger.info(f"Defined structure pairs for reshaping: {STRUCTURE_PAIRS}")

# --- Helper Function ---

def extract_step_from_path(file_path: Path) -> int | None:
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
    extracts standard summaries, extracts and RESHAPES priming summaries,
    compiles them into DataFrames, and saves them within the input_dir.
    """
    priming_data_list = [] # Will store reshaped priming data
    standard_data_list = []
    files_processed = 0
    files_found = 0
    max_step = -1

    logger.info(f"Starting compilation process in directory: {input_dir}")
    logger.info(f"Searching recursively for files matching: {json_filename_pattern}")

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

        # --- Process 'priming_summary' section (RESHAPED) ---
        priming_summary = data.get("priming_summary")
        if isinstance(priming_summary, dict):
            for csv_filename, metrics_dict in priming_summary.items():
                if not isinstance(metrics_dict, dict):
                    logger.warning(
                        f"Step {checkpoint_step}, File {csv_filename}: Expected 'metrics' to be a dict, but got {type(metrics_dict)}. Skipping entry."
                    )
                    continue

                # --- Reshaping Logic ---
                # 1. Parse all keys into base_metric and structure_label
                parsed_metrics = defaultdict(dict) # Structure: {base_metric: {struct_label: value}}
                structure_labels_present = set()
                base_metrics_present = set()

                for key, value in metrics_dict.items():
                    # Regex to capture base metric (group 1) and structure label (group 2)
                    # Assumes label is the last part after '_' and contains no '_' itself
                    match = re.match(r"(.*)_([^_]+)$", key)
                    if match:
                        base_metric, struct_label = match.groups()
                        # Handle potential non-numeric conversion issues early if needed
                        # try:
                        #     num_value = float(value)
                        # except (ValueError, TypeError):
                        #     num_value = None # Or keep as string, or log warning
                        parsed_metrics[base_metric][struct_label] = value # Store original value for now
                        structure_labels_present.add(struct_label)
                        base_metrics_present.add(base_metric)
                        # logger.debug(f"Parsed key '{key}' -> base: '{base_metric}', label: '{struct_label}'")
                    else:
                        logger.debug(f"Step {checkpoint_step}, File {csv_filename}: Could not parse key '{key}' into base_metric and structure_label.")

                # 2. Iterate through defined structure pairs
                for s1_label, s2_label in STRUCTURE_PAIRS:
                    # Check if BOTH labels of the pair are present in this file's data
                    if s1_label in structure_labels_present and s2_label in structure_labels_present:
                        contrast_label = f"{s1_label}/{s2_label}"
                        # 3. Iterate through the base metrics found in this file
                        for base_metric in base_metrics_present:
                            # Retrieve the values for struct1 and struct2 for this base_metric
                            # Uses .get() defaults to None if a specific struct_label wasn't found for this base_metric
                            value1 = parsed_metrics.get(base_metric, {}).get(s1_label)
                            value2 = parsed_metrics.get(base_metric, {}).get(s2_label)

                            # Create a row only if we found this base_metric for EITHER structure label
                            # (This check might be redundant if both s1 and s2 must be present)
                            if value1 is not None or value2 is not None:
                                row_data = {
                                    "checkpoint_step": checkpoint_step,
                                    "corpus_file": csv_filename,
                                    "metric_base": base_metric,
                                    "contrast_pair": contrast_label,
                                    # "label_struct1": s1_label, # Optional: uncomment to keep original labels
                                    # "label_struct2": s2_label, # Optional: uncomment to keep original labels
                                    "value_struct1": value1, # Value corresponding to s1_label
                                    "value_struct2": value2, # Value corresponding to s2_label
                                }
                                priming_data_list.append(row_data)
                        # logger.debug(f"Processed pair {contrast_label} for file {csv_filename}")
                    # else:
                        # logger.debug(f"Skipping pair {s1_label}/{s2_label} for file {csv_filename} as one or both labels missing.")
                # --- End Reshaping Logic ---

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
        # Reduce logging verbosity - comment out if not debugging
        # logger.info(f"Successfully processed step {checkpoint_step} from {json_path.name}")


    logger.info(f"Found {files_found} files matching pattern.")
    if files_processed == 0:
        logger.warning("No valid evaluation summary files were processed. No CSV files will be created.")
        return

    logger.info(f"Successfully processed data from {files_processed} files. Highest checkpoint step found: {max_step}")

    # --- Determine Output Filenames and Save DataFrames ---
    output_suffix = f"upto_step_{max_step}.csv"
    # Add 'reshaped' to priming filename for clarity
    output_priming_csv = input_dir / f"compiled_priming_summary_reshaped_{output_suffix}"
    output_standard_csv = input_dir / f"compiled_standard_summary_{output_suffix}" # Standard name unchanged

    # Priming Summary DataFrame (Reshaped)
    if priming_data_list:
        try:
            priming_df = pd.DataFrame(priming_data_list)
            # Sort reshaped data appropriately
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
                    "Priming data is reshaped into a 'long' format based on structure pairs. "
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