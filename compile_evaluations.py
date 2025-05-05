import argparse
import json
import logging
import pandas as pd
from pathlib import Path
import re
from collections import defaultdict

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Helper Function ---

def extract_step_from_path(file_path: Path) -> int | None:
    """Extracts the checkpoint step number from the filename."""
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

# --- Automatic Pair Detection Function ---

def detect_structure_pair(metrics_dict: dict) -> tuple | None:
    """
    Attempts to detect exactly one pair of structure labels based on keys
    starting with 'avg_PE_t' within the given metrics dictionary.

    Args:
        metrics_dict: The dictionary of metrics for a single corpus file.

    Returns:
        A tuple containing the two detected structure labels (sorted alphabetically),
        or None if exactly two distinct labels were not found.
    """
    t_structure_labels = set()
    prefix = "avg_PE_t" # Using the specific prefix requested

    if not isinstance(metrics_dict, dict):
        return None # Cannot process if not a dictionary

    for key in metrics_dict.keys():
        if key.startswith(prefix):
            label = key[len(prefix):] # Extract the part after the prefix
            if label: # Ensure the label is not empty
                t_structure_labels.add(label)

    if len(t_structure_labels) == 2:
        detected_pair = tuple(sorted(list(t_structure_labels)))
        logger.info(f"Auto-detected structure pair based on '{prefix}*' keys: {detected_pair}")
        return detected_pair
    else:
        logger.warning(
            f"Could not auto-detect exactly one structure pair using prefix '{prefix}'. "
            f"Found {len(t_structure_labels)} distinct labels: {t_structure_labels}. "
            f"Priming data reshaping will be skipped."
        )
        return None


# --- Main Compilation Function ---

def compile_evaluation_summaries(
    input_dir: Path,
    json_filename_pattern: str = "evaluation_summary_step_*.json"
):
    """
    Traverses subdirectories, finds evaluation summary JSONs,
    extracts standard summaries, attempts to auto-detect pairs and RESHAPE
    priming summaries, compiles them into DataFrames, and saves them.
    """
    priming_data_list = []
    standard_data_list = []
    files_processed = 0
    files_found = 0
    max_step = -1
    detected_pair_tuple = None # Stores the single detected pair, e.g., ('do', 'po')
    pair_detection_attempted = False # Flag to ensure detection only happens once

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
            continue

        priming_summary = data.get("priming_summary")

        # --- Attempt Pair Detection (only on the first valid priming summary found) ---
        if not pair_detection_attempted and isinstance(priming_summary, dict) and priming_summary:
            # Try detecting from the metrics of the *first* corpus file listed
            first_corpus_file = next(iter(priming_summary), None)
            if first_corpus_file:
                first_metrics_dict = priming_summary[first_corpus_file]
                detected_pair_tuple = detect_structure_pair(first_metrics_dict)
            else:
                 logger.warning(f"First priming summary found in {json_path.name} was empty. Cannot detect pairs.")
            pair_detection_attempted = True # Mark detection as attempted, even if failed

        # --- Process 'priming_summary' section (RESHAPED, only if pair was detected) ---
        if detected_pair_tuple and isinstance(priming_summary, dict): # Only reshape if a pair was successfully detected
            s1_label, s2_label = detected_pair_tuple # Unpack the detected pair
            contrast_label = f"{s1_label}/{s2_label}" # Precompute contrast string

            for csv_filename, metrics_dict in priming_summary.items():
                if not isinstance(metrics_dict, dict):
                    logger.warning(f"Step {checkpoint_step}, File {csv_filename}: Invalid metrics format. Skipping.")
                    continue

                # --- Reshaping Logic (using the single detected pair) ---
                parsed_metrics = defaultdict(dict)
                base_metrics_present = set()
                for key, value in metrics_dict.items():
                    match = re.match(r"(.*)_([^_]+)$", key)
                    if match:
                        base_metric, struct_label = match.groups()
                        if struct_label in detected_pair_tuple: # Only parse metrics relevant to the detected pair
                             parsed_metrics[base_metric][struct_label] = value
                             base_metrics_present.add(base_metric)
                    # else: # Optional: log keys that don't match the expected format
                        # logger.debug(f"Key '{key}' in file {csv_filename} doesn't match expected parsing format.")

                for base_metric in base_metrics_present:
                    value1 = parsed_metrics.get(base_metric, {}).get(s1_label)
                    value2 = parsed_metrics.get(base_metric, {}).get(s2_label)

                    if value1 is not None or value2 is not None:
                        row_data = {
                            "checkpoint_step": checkpoint_step,
                            "corpus_file": csv_filename,
                            "metric_base": base_metric,
                            "contrast_pair": contrast_label, # Use the consistent label derived from the pair
                            "value_struct1": value1,
                            "value_struct2": value2,
                        }
                        priming_data_list.append(row_data)
                # --- End Reshaping Logic ---
        elif isinstance(priming_summary, dict) and pair_detection_attempted and not detected_pair_tuple:
            # Log only once per file if detection was attempted but failed
            logger.debug(f"Step {checkpoint_step}, File {json_path.name}: Skipping priming data reshaping as pair detection failed or wasn't applicable.")


        # --- Process 'standard_summary' section (Unchanged) ---
        standard_summary = data.get("standard_summary")
        if isinstance(standard_summary, dict):
            row_data = {"checkpoint_step": checkpoint_step}
            row_data.update(standard_summary)
            standard_data_list.append(row_data)
        elif standard_summary is not None:
             logger.warning(f"Step {checkpoint_step}: Invalid standard_summary format. Skipping.")


        files_processed += 1


    logger.info(f"Found {files_found} files matching pattern.")
    if files_processed == 0:
        logger.warning("No valid evaluation summary files were processed. No CSV files will be created.")
        return

    logger.info(f"Successfully processed data from {files_processed} files. Highest checkpoint step found: {max_step}")

    # --- Determine Output Filenames and Save DataFrames ---
    output_suffix = f"upto_step_{max_step}.csv"
    output_standard_csv = input_dir / f"compiled_standard_summary_{output_suffix}"

    # Only save priming CSV if pair detection was successful and data was generated
    if detected_pair_tuple and priming_data_list:
        output_priming_csv = input_dir / f"compiled_priming_summary_reshaped_auto_{output_suffix}"
        try:
            priming_df = pd.DataFrame(priming_data_list)
            priming_df = priming_df.sort_values(
                by=["checkpoint_step", "corpus_file", "contrast_pair", "metric_base"]
            )
            logger.info(f"Created AUTO-RESHAPED Priming Summary DataFrame with {len(priming_df)} rows and {len(priming_df.columns)} columns.")
            priming_df.to_csv(output_priming_csv, index=False, encoding='utf-8')
            logger.info(f"Successfully saved AUTO-RESHAPED Priming Summary data to: {output_priming_csv}")
        except Exception as e:
            logger.error(f"Failed to create or save AUTO-RESHAPED Priming Summary DataFrame: {e}")
    elif not detected_pair_tuple and pair_detection_attempted:
         logger.warning("Priming data was found, but reshaping was skipped because structure pair detection failed.")
    elif not priming_data_list and detected_pair_tuple:
         logger.warning("Structure pair was detected, but no corresponding priming data rows were generated.")
    else:
         # This case includes scenarios where no priming_summary was ever found
         logger.info("No reshaped priming data was generated (either no priming data found or pair detection not applicable).")


    # Standard Summary DataFrame
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
        logger.warning("No data found for Standard Summary. CSV file will not be created.")

    logger.info("Compilation process finished.")


# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile evaluation summary JSON files into CSVs. "
                    "Attempts to auto-detect structure pairs from 'avg_PE_t*' keys in the first file "
                    "to reshape priming data. Output CSVs saved in the input directory."
    )
    # Arguments remain the same as the previous simplified version
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Root directory with evaluation outputs. Searched recursively. Output saved here."
    )
    parser.add_argument(
        "--filename_pattern", type=str, default="evaluation_summary_step_*.json",
        help="Glob pattern for summary JSON files."
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