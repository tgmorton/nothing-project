import argparse
import json
import logging
from typing import Optional, List, Any, Dict

import pandas as pd
from pathlib import Path
import re
from collections import defaultdict

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- Helper Functions (Unchanged) ---

def extract_step_from_path(file_path: Path) -> Optional[int]:
    """
    Extracts the checkpoint step number from the filename.
    Handles formats like '..._step_12345.json' or '...checkpoint-12345.json'.
    """
    filename = file_path.name
    logger.debug(f"Attempting to extract step from filename: {filename}")

    # Match '..._step_12345.json'
    match_step = re.search(r"_step_(\d+)\.json$", filename)
    if match_step:
        try:
            step = int(match_step.group(1))
            logger.debug(f"Matched _step_ pattern. Extracted step: {step} from {filename}")
            return step
        except ValueError:
            logger.warning(f"Could not convert step number in filename {filename} to integer (from _step_ pattern).")
            return None

    # Match '...checkpoint-12345.json'
    match_checkpoint = re.search(r"checkpoint-(\d+)\.json$", filename)
    if match_checkpoint:
        try:
            step = int(match_checkpoint.group(1))
            logger.debug(f"Matched checkpoint- pattern. Extracted step: {step} from {filename}")
            return step
        except ValueError:
            logger.warning(
                f"Could not convert step number in filename {filename} to integer (from checkpoint- pattern).")
            return None

    logger.warning(
        f"Could not find step number pattern ('_step_NUMBER.json' or 'checkpoint-NUMBER.json') in filename {filename}.")
    return None


# --- Main Compilation Function (Adapted for a single model) ---

def compile_single_model_summary(
        model_dir: Path
):
    """
    Compiles evaluation summaries from a single model directory.
    """
    all_priming_data_list = []
    all_standard_data_list = []
    files_processed_successfully = 0
    max_step_overall = -1
    processed_json_paths = set()

    json_filename_patterns: List[str] = [
        "evaluation_summary_step_*.json",
        "evaluation_summary_checkpoint-*.json"
    ]

    model_name = model_dir.name
    logger.info(f"Starting compilation process for model: {model_name}")
    logger.info(f"Searching recursively within {model_dir} for files matching patterns: {json_filename_patterns}")

    # --- CHANGE: Removed the outer loop that iterated through subdirectories ---
    # The script now directly processes the provided model_dir.

    for pattern in json_filename_patterns:
        logger.debug(f"Searching for pattern: {pattern} in {model_dir}")
        for json_path in model_dir.rglob(pattern):
            logger.debug(f"Found potential file: {json_path} (matching pattern: {pattern})")

            if json_path in processed_json_paths:
                logger.debug(f"Skipping already processed file: {json_path}")
                continue
            processed_json_paths.add(json_path)

            checkpoint_step = extract_step_from_path(json_path)
            logger.info(f"File: {json_path.name}, Extracted Step: {checkpoint_step}")

            if checkpoint_step is None:
                logger.warning(f"Skipping file {json_path} due to missing/invalid step number.")
                continue

            max_step_overall = max(max_step_overall, checkpoint_step)
            logger.debug(f"Current overall max step: {max_step_overall}")

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.debug(f"Successfully loaded JSON from {json_path}. Top-level keys: {list(data.keys())}")
            except Exception as e:
                logger.error(f"Error reading or parsing JSON file {json_path}: {e}. Skipping.")
                continue

            # --- Process Priming Data ---
            file_had_priming_data = False
            priming_summary_data: Optional[Dict[str, Any]] = data.get("priming_evaluation_summary",
                                                                      data.get("priming_summary"))

            if isinstance(priming_summary_data, dict):
                initial_priming_list_len = len(all_priming_data_list)
                for csv_filename, metrics_dict in priming_summary_data.items():
                    if not isinstance(metrics_dict, dict):
                        logger.warning(f"Skipping non-dict metrics for {csv_filename} in {json_path.name}")
                        continue
                    if "error" in metrics_dict:
                        logger.warning(f"Skipping error entry for {csv_filename} in {json_path.name}")
                        continue

                    parsed_metrics = defaultdict(dict)
                    base_metrics_present = set()
                    for key, value in metrics_dict.items():
                        match = re.match(r"(.*)_([^_]+)$", key)
                        if match:
                            base_metric, struct_label = match.groups()
                            parsed_metrics[base_metric][struct_label] = value
                            base_metrics_present.add(base_metric)
                        else:
                            logger.debug(f"Could not parse key '{key}' in {json_path.name}")

                    for base_metric in base_metrics_present:
                        associated_labels = list(parsed_metrics.get(base_metric, {}).keys())
                        if len(associated_labels) >= 2:
                            for i in range(len(associated_labels)):
                                for j in range(i + 1, len(associated_labels)):
                                    s1_label, s2_label = sorted([associated_labels[i], associated_labels[j]])
                                    contrast_label = f"{s1_label}/{s2_label}"
                                    value1 = parsed_metrics[base_metric][s1_label]
                                    value2 = parsed_metrics[base_metric][s2_label]
                                    # --- CHANGE: Removed 'learning_rate' from the data row ---
                                    row_data = {
                                        "model_name": model_name,
                                        "checkpoint_step": checkpoint_step,
                                        "corpus_file": csv_filename,
                                        "metric_base": base_metric,
                                        "contrast_pair": contrast_label,
                                        "value_struct1": value1,
                                        "value_struct2": value2,
                                    }
                                    all_priming_data_list.append(row_data)

                if len(all_priming_data_list) > initial_priming_list_len:
                    file_had_priming_data = True

            # --- Process Standard Data ---
            file_had_standard_data = False
            standard_summary_data: Optional[Dict[str, Any]] = data.get("standard_perplexity_summary",
                                                                       data.get("standard_summary"))

            if isinstance(standard_summary_data, dict):
                # --- CHANGE: Removed 'learning_rate' from the data row ---
                row_data = {
                    "model_name": model_name,
                    "checkpoint_step": checkpoint_step
                }
                row_data.update(standard_summary_data)
                all_standard_data_list.append(row_data)
                file_had_standard_data = True

            if file_had_priming_data or file_had_standard_data:
                files_processed_successfully += 1


    if not processed_json_paths:
        logger.warning("No evaluation summary files found. No CSV files will be created.")
        return

    logger.info(
        f"Successfully extracted data from {files_processed_successfully} files. Max step: {max_step_overall}")

    output_suffix = f"upto_step_{max_step_overall}_combined.csv"

    # --- Save Priming Data CSV ---
    if all_priming_data_list:
        try:
            priming_df = pd.DataFrame(all_priming_data_list)
            # --- CHANGE: Removed 'learning_rate' from sort columns ---
            sort_cols = ["model_name", "checkpoint_step", "corpus_file"]
            existing_cols = [col for col in sort_cols if col in priming_df.columns]
            priming_df = priming_df.sort_values(by=existing_cols)

            # --- CHANGE: Output file is saved inside the model_dir ---
            output_priming_csv = model_dir / f"compiled_priming_summary_reshaped_{output_suffix}"
            priming_df.to_csv(output_priming_csv, index=False, encoding='utf-8')
            logger.info(f"Saved RESHAPED Priming Summary to: {output_priming_csv}")
        except Exception as e:
            logger.error(f"Failed to save RESHAPED Priming Summary DataFrame: {e}")
    else:
        logger.warning("No priming data found. Priming summary CSV will not be created.")

    # --- Save Standard Data CSV ---
    if all_standard_data_list:
        try:
            standard_df = pd.DataFrame(all_standard_data_list)
            # --- CHANGE: Removed 'learning_rate' from sort columns ---
            sort_cols = ["model_name", "checkpoint_step"]
            existing_cols = [col for col in sort_cols if col in standard_df.columns]
            standard_df = standard_df.sort_values(by=existing_cols)

            # --- CHANGE: Output file is saved inside the model_dir ---
            output_standard_csv = model_dir / f"compiled_standard_summary_{output_suffix}"
            standard_df.to_csv(output_standard_csv, index=False, encoding='utf-8')
            logger.info(f"Saved Standard Summary to: {output_standard_csv}")
        except Exception as e:
            logger.error(f"Failed to save Standard Summary DataFrame: {e}")
    else:
        logger.warning("No standard data found. Standard summary CSV will not be created.")

    logger.info("Compilation process finished.")


if __name__ == "__main__":
    # --- CHANGE: Updated description and argument help text for clarity ---
    parser = argparse.ArgumentParser(
        description="Compile evaluation summary JSON files from a single model directory. "
                    "Output CSVs are saved within that same directory."
    )
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="The directory of a single model run containing evaluation summary JSONs."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG level logging for detailed output."
    )

    args = parser.parse_args()
    # --- CHANGE: Using a more descriptive variable name ---
    model_path = Path(args.model_dir)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.info("DEBUG mode enabled.")

    if not model_path.is_dir():
        logger.error(f"Input directory not found or is not a directory: {model_path}")
        exit(1)

    # --- CHANGE: Calling the new, adapted function ---
    compile_single_model_summary(model_dir=model_path)