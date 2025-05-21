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


# --- Helper Functions ---

def extract_step_from_path(file_path: Path) -> Optional[int]:
    """
    Extracts the checkpoint step number from the filename.
    Handles formats like '..._step_12345.json' or '...checkpoint-12345.json'.
    """
    filename = file_path.name
    logger.debug(f"Attempting to extract step from filename: {filename}")

    match_step = re.search(r"_step_(\d+)\.json$", filename)
    if match_step:
        try:
            step = int(match_step.group(1))
            logger.debug(f"Matched _step_ pattern. Extracted step: {step} from {filename}")
            return step
        except ValueError:
            logger.warning(f"Could not convert step number in filename {filename} to integer (from _step_ pattern).")
            return None

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


def extract_seed_from_folder_name(folder_path: Path) -> Optional[int]:
    """Extracts the seed number from the folder name (e.g., ..._seed123)."""
    folder_name = folder_path.name
    match = re.search(r"_seed(\d+)$", folder_name)
    if match:
        try:
            seed = int(match.group(1))
            logger.debug(f"Extracted seed {seed} from folder name {folder_name}")
            return seed
        except ValueError:
            logger.warning(f"Could not convert extracted seed in folder name {folder_name} to integer.")
            return None
    logger.debug(f"No seed pattern found in folder name {folder_name}. Searched in '{folder_name}'")
    return None


# --- Main Compilation Function ---

def compile_evaluation_summaries(
        parent_models_dir: Path
):
    all_priming_data_list = []
    all_standard_data_list = []
    files_processed_successfully_overall = 0
    max_step_overall = -1
    processed_json_paths_globally = set()

    json_filename_patterns: List[str] = [
        "evaluation_summary_step_*.json",
        "evaluation_summary_checkpoint-*.json"
    ]

    logger.info(f"Starting compilation process in parent directory: {parent_models_dir}")
    logger.info(f"Searching for model run subdirectories within {parent_models_dir}")

    model_run_dirs_found = 0
    for model_run_path in parent_models_dir.iterdir():
        if not model_run_path.is_dir():
            logger.debug(f"Skipping non-directory item: {model_run_path.name}")
            continue

        model_run_dirs_found += 1
        model_run_name = model_run_path.name
        seed_number = extract_seed_from_folder_name(model_run_path)

        logger.info(
            f"Processing model run: {model_run_name} (Seed: {seed_number if seed_number is not None else 'N/A'})")
        logger.info(
            f"Searching recursively within {model_run_path} for files matching patterns: {json_filename_patterns}")

        for pattern in json_filename_patterns:
            logger.debug(f"Searching for pattern: {pattern} in {model_run_path}")
            for json_path in model_run_path.rglob(pattern):
                logger.debug(f"Found potential file: {json_path} (matching pattern: {pattern})")

                if json_path in processed_json_paths_globally:
                    logger.debug(f"Skipping already processed file: {json_path}")
                    continue

                processed_json_paths_globally.add(json_path)  # Add to global set

                checkpoint_step = extract_step_from_path(json_path)
                logger.info(f"File: {json_path.name} (from {model_run_name}), Extracted Step: {checkpoint_step}")

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

                file_had_priming_data = False
                priming_summary_data: Optional[Dict[str, Any]] = data.get("priming_evaluation_summary",
                                                                          data.get("priming_summary"))

                if isinstance(priming_summary_data, dict):
                    initial_priming_list_len = len(all_priming_data_list)
                    for csv_filename, metrics_dict in priming_summary_data.items():
                        if not isinstance(metrics_dict, dict):
                            logger.warning(
                                f"Model: {model_run_name}, Step {checkpoint_step}, File {json_path.name}, CSV {csv_filename}: Expected 'metrics' to be a dict, but got {type(metrics_dict)}. Skipping entry."
                            )
                            continue
                        if "error" in metrics_dict:
                            logger.warning(
                                f"Model: {model_run_name}, Step {checkpoint_step}, File {json_path.name}, CSV {csv_filename}: Found error entry: {metrics_dict['error']}. Skipping.")
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
                                logger.debug(
                                    f"Model: {model_run_name}, Step {checkpoint_step}, File {json_path.name}, CSV {csv_filename}: Could not parse key '{key}' into base_metric and structure_label.")

                        for base_metric in base_metrics_present:
                            associated_labels = parsed_metrics.get(base_metric, {}).keys()
                            t_labels = {lbl for lbl in associated_labels if lbl.lower().startswith('t')}
                            if len(t_labels) == 2:
                                label_list = sorted(list(t_labels))
                                s1_label, s2_label = label_list[0], label_list[1]
                                contrast_label = f"{s1_label}/{s2_label}"
                                value1 = parsed_metrics[base_metric][s1_label]
                                value2 = parsed_metrics[base_metric][s2_label]
                                row_data = {
                                    "model_run_name": model_run_name,
                                    "seed_number": seed_number,
                                    "checkpoint_step": checkpoint_step,
                                    "corpus_file": csv_filename,
                                    "metric_base": base_metric,
                                    "contrast_pair": contrast_label,
                                    "value_struct1": value1,
                                    "value_struct2": value2,
                                }
                                all_priming_data_list.append(row_data)
                            elif len(t_labels) > 2:
                                logger.warning(
                                    f"Model: {model_run_name}, Step {checkpoint_step}, File {json_path.name}, CSV {csv_filename}, Metric '{base_metric}': "
                                    f"Found {len(t_labels)} 't' labels ({t_labels}), expected 2. Skipping pair generation."
                                )
                    if len(all_priming_data_list) > initial_priming_list_len:
                        file_had_priming_data = True
                        logger.debug(
                            f"Added {len(all_priming_data_list) - initial_priming_list_len} priming rows from {json_path.name} (Model: {model_run_name})")
                elif priming_summary_data is not None:
                    logger.warning(
                        f"Model: {model_run_name}, Step {checkpoint_step}, File {json_path.name}: Expected priming summary to be a dict, "
                        f"but got {type(priming_summary_data)}. Skipping priming section."
                    )
                else:
                    logger.debug(
                        f"No 'priming_summary' or 'priming_evaluation_summary' key found in {json_path.name} for step {checkpoint_step} (Model: {model_run_name}).")

                file_had_standard_data = False
                standard_summary_data: Optional[Dict[str, Any]] = data.get("standard_perplexity_summary",
                                                                           data.get("standard_summary"))
                data_source_key_name = "standard_perplexity_summary" if data.get(
                    "standard_perplexity_summary") else "standard_summary"

                if isinstance(standard_summary_data, dict):
                    row_data = {
                        "model_run_name": model_run_name,
                        "seed_number": seed_number,
                        "checkpoint_step": checkpoint_step
                    }
                    row_data.update(standard_summary_data)
                    all_standard_data_list.append(row_data)
                    file_had_standard_data = True
                    logger.debug(
                        f"Added standard summary row from {json_path.name} (using key '{data_source_key_name}') for step {checkpoint_step} (Model: {model_run_name}).")
                elif standard_summary_data is not None:
                    logger.warning(
                        f"Model: {model_run_name}, Step {checkpoint_step}, File {json_path.name}: Expected standard summary to be a dict, "
                        f"but got {type(standard_summary_data)}. Skipping standard section."
                    )
                else:
                    logger.debug(
                        f"No 'standard_summary' or 'standard_perplexity_summary' key found in {json_path.name} for step {checkpoint_step} (Model: {model_run_name}).")

                if file_had_priming_data or file_had_standard_data:
                    files_processed_successfully_overall += 1
                else:
                    logger.warning(
                        f"File {json_path.name} (Step {checkpoint_step}, Model: {model_run_name}) did not yield any priming or standard data, though JSON was parsed.")

    if model_run_dirs_found == 0:
        logger.warning(f"No subdirectories found in {parent_models_dir}. No data to process.")
        return

    logger.info(f"Total unique JSON file paths encountered across all model runs: {len(processed_json_paths_globally)}")
    if not processed_json_paths_globally:
        logger.warning(
            "No evaluation summary files were found matching the patterns in any model run. No CSV files will be created.")
        return
    if files_processed_successfully_overall == 0:
        logger.warning(
            "Found JSON files, but no valid evaluation summary data was extracted from any of them. No CSV files will be created.")
        return

    logger.info(
        f"Successfully extracted data from {files_processed_successfully_overall} unique JSON files across all model runs. Highest checkpoint step found: {max_step_overall}")

    output_suffix = f"upto_step_{max_step_overall}_combined.csv"
    output_priming_csv = parent_models_dir / f"compiled_priming_summary_reshaped_{output_suffix}"
    output_standard_csv = parent_models_dir / f"compiled_standard_summary_{output_suffix}"

    if all_priming_data_list:
        try:
            priming_df = pd.DataFrame(all_priming_data_list)
            # Define sort order, ensuring 'seed_number' might be None and handling it
            priming_df['seed_number'] = priming_df['seed_number'].astype('Int64')  # Allows pd.NA

            sort_cols_priming = ["model_run_name", "seed_number", "checkpoint_step", "corpus_file", "metric_base",
                                 "contrast_pair"]
            existing_sort_cols_priming = [col for col in sort_cols_priming if col in priming_df.columns]
            if existing_sort_cols_priming:
                priming_df = priming_df.sort_values(by=existing_sort_cols_priming)

            logger.info(
                f"Created COMBINED RESHAPED Priming Summary DataFrame with {len(priming_df)} rows and {len(priming_df.columns)} columns.")
            priming_df.to_csv(output_priming_csv, index=False, encoding='utf-8')
            logger.info(f"Successfully saved COMBINED RESHAPED Priming Summary data to: {output_priming_csv}")
        except Exception as e:
            logger.error(f"Failed to create or save COMBINED RESHAPED Priming Summary DataFrame: {e}")
    else:
        logger.warning("No data found for COMBINED RESHAPED Priming Summary. CSV file will not be created.")

    if all_standard_data_list:
        try:
            standard_df = pd.DataFrame(all_standard_data_list)
            standard_df['seed_number'] = standard_df['seed_number'].astype('Int64')  # Allows pd.NA

            sort_cols_standard = ["model_run_name", "seed_number", "checkpoint_step"]
            existing_sort_cols_standard = [col for col in sort_cols_standard if col in standard_df.columns]

            if "checkpoint_step" not in standard_df.columns and existing_sort_cols_standard:  # fallback if somehow checkpoint_step isn't there for sorting
                logger.warning(
                    "Column 'checkpoint_step' not found in standard_df for primary sort, but other sort keys exist.")
            elif not existing_sort_cols_standard:
                logger.warning(
                    "No sort columns (model_run_name, seed_number, checkpoint_step) found in standard_df, cannot sort.")

            if existing_sort_cols_standard:
                standard_df = standard_df.sort_values(by=existing_sort_cols_standard)

            logger.info(
                f"Created COMBINED Standard Summary DataFrame with {len(standard_df)} rows and {len(standard_df.columns)} columns.")
            standard_df.to_csv(output_standard_csv, index=False, encoding='utf-8')
            logger.info(f"Successfully saved COMBINED Standard Summary data to: {output_standard_csv}")
        except Exception as e:
            logger.error(f"Failed to create or save COMBINED Standard Summary DataFrame: {e}")
    else:
        logger.warning("No data found for COMBINED Standard Summary. CSV file will not be created.")

    logger.info("Compilation process finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile evaluation summary JSON files from multiple model run subdirectories into combined CSVs. "
                    "Searches for 'evaluation_summary_step_*.json' and 'evaluation_summary_checkpoint-*.json' within each model run subdirectory. "
                    "Extracts 'seed_number' from model run folder names (if pattern '..._seed<NUMBER>' exists). "
                    "Priming data is reshaped. Output CSVs are saved in the specified input directory."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="The parent directory containing model run subdirectories. Searched recursively within subdirectories. Combined output CSVs saved here."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG level logging for detailed output."
    )

    args = parser.parse_args()
    input_path = Path(args.input_dir)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.info("DEBUG mode enabled.")  # Changed to info so it's visible even if root is INFO but local is DEBUG

    if not input_path.is_dir():
        logger.error(f"Input directory not found or is not a directory: {input_path}")
        exit(1)

    compile_evaluation_summaries(parent_models_dir=input_path)