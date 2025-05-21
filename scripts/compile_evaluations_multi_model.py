import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
from collections import defaultdict

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
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

    # For eval_results_checkpoint-X files
    match_eval_results = re.search(r"eval_results_checkpoint-(\d+)$", filename)
    if match_eval_results:
        try:
            step = int(match_eval_results.group(1))
            logger.debug(f"Matched eval_results_checkpoint- pattern. Extracted step: {step} from {filename}")
            return step
        except ValueError:
            logger.warning(
                f"Could not convert step number in filename {filename} to integer (from eval_results_checkpoint- pattern).")
            return None

    logger.warning(
        f"Could not find step number pattern in filename {filename}.")
    return None


def extract_seed_from_path(model_dir: Path) -> Optional[int]:
    """
    Extracts the seed number from the model directory name.
    Handles formats like 'gpt2_p6000_sif_local_eval_run_May20_sweep_seedX'.
    """
    dir_name = model_dir.name
    logger.debug(f"Attempting to extract seed from directory name: {dir_name}")

    # Try to match seed pattern in directory name
    match_seed = re.search(r"seed(\d+)$", dir_name)
    if match_seed:
        try:
            seed = int(match_seed.group(1))
            logger.debug(f"Matched seed pattern. Extracted seed: {seed} from {dir_name}")
            return seed
        except ValueError:
            logger.warning(f"Could not convert seed number in directory name {dir_name} to integer.")
            return None

    logger.warning(f"Could not find seed number pattern in directory name {dir_name}.")
    return None


def process_model_directory(
        model_dir: Path,
        seed: Optional[int] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process a single model directory to extract evaluation data.
    Returns two lists of dictionaries: priming_data_list and standard_data_list.
    """
    priming_data_list = []
    standard_data_list = []
    files_processed_successfully = 0
    max_step = -1
    processed_paths = set()

    # If seed is not provided, try to extract it from the directory name
    if seed is None:
        seed = extract_seed_from_path(model_dir)
        if seed is None:
            logger.warning(f"Could not extract seed from directory name: {model_dir.name}. Using None.")

    # Look for evaluation files in the eval directory
    eval_dir = model_dir / "eval"
    if not eval_dir.exists() or not eval_dir.is_dir():
        logger.warning(f"Eval directory not found in {model_dir}. Skipping.")
        return priming_data_list, standard_data_list

    json_filename_patterns: List[str] = [
        "evaluation_summary_step_*.json",
        "evaluation_summary_checkpoint-*.json",
        "eval_results_checkpoint-*"
    ]

    logger.info(f"Starting compilation process in directory: {eval_dir}")
    logger.info(f"Searching for files matching patterns: {json_filename_patterns}")

    for pattern in json_filename_patterns:
        logger.info(f"Searching for pattern: {pattern}")
        for json_path in eval_dir.glob(pattern):
            if json_path in processed_paths:
                logger.debug(f"Skipping already processed file: {json_path}")
                continue

            processed_paths.add(json_path)
            checkpoint_step = extract_step_from_path(json_path)
            logger.info(f"File: {json_path.name}, Extracted Step: {checkpoint_step}")

            if checkpoint_step is None:
                logger.warning(f"Skipping file {json_path} due to missing/invalid step number.")
                continue

            max_step = max(max_step, checkpoint_step)

            try:
                # For JSON files
                if json_path.suffix == '.json':
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    logger.debug(f"Successfully loaded JSON from {json_path}.")
                # For non-JSON files (like eval_results_checkpoint-X)
                else:
                    # Assume it's a text file with key-value pairs
                    data = {}
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                if '=' in line:
                                    key, value = line.strip().split('=', 1)
                                    try:
                                        # Try to convert to float if possible
                                        data[key] = float(value)
                                    except ValueError:
                                        data[key] = value
                    except Exception as e:
                        logger.error(f"Error reading or parsing file {json_path}: {e}. Skipping.")
                        continue
            except Exception as e:
                logger.error(f"Error reading or parsing file {json_path}: {e}. Skipping.")
                continue

            file_had_priming_data = False
            # Try new key name first, then old key name
            priming_summary_data: Optional[Dict[str, Any]] = data.get("priming_evaluation_summary")
            if priming_summary_data is None:
                priming_summary_data = data.get("priming_summary")

            if isinstance(priming_summary_data, dict):
                initial_priming_list_len = len(priming_data_list)
                for csv_filename, metrics_dict in priming_summary_data.items():
                    if not isinstance(metrics_dict, dict):
                        logger.warning(
                            f"Step {checkpoint_step}, File {json_path.name}, CSV {csv_filename}: Expected 'metrics' to be a dict, but got {type(metrics_dict)}. Skipping entry."
                        )
                        continue

                    if "error" in metrics_dict:
                        logger.warning(
                            f"Step {checkpoint_step}, File {json_path.name}, CSV {csv_filename}: Found error entry: {metrics_dict['error']}. Skipping.")
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
                                f"Step {checkpoint_step}, File {json_path.name}, CSV {csv_filename}: Could not parse key '{key}' into base_metric and structure_label.")
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
                                "seed": seed,
                                "checkpoint_step": checkpoint_step,
                                "corpus_file": csv_filename,
                                "metric_base": base_metric,
                                "contrast_pair": contrast_label,
                                "value_struct1": value1,
                                "value_struct2": value2,
                            }
                            priming_data_list.append(row_data)
                        elif len(t_labels) > 2:
                            logger.warning(
                                f"Step {checkpoint_step}, File {json_path.name}, CSV {csv_filename}, Metric '{base_metric}': "
                                f"Found {len(t_labels)} 't' labels ({t_labels}), expected 2. Skipping pair generation."
                            )
                if len(priming_data_list) > initial_priming_list_len:
                    file_had_priming_data = True
                    logger.debug(
                        f"Added {len(priming_data_list) - initial_priming_list_len} priming rows from {json_path.name}")

            file_had_standard_data = False
            # Try new key name first, then old key name
            standard_summary_data: Optional[Dict[str, Any]] = data.get("standard_perplexity_summary")
            data_source_key_name = "standard_perplexity_summary"
            if standard_summary_data is None:
                standard_summary_data = data.get("standard_summary")
                data_source_key_name = "standard_summary"

            # If it's not a JSON file with standard summary keys, use the entire data dict
            if standard_summary_data is None and json_path.suffix != '.json':
                standard_summary_data = data
                data_source_key_name = "file_content"

            if isinstance(standard_summary_data, dict):
                row_data = {"seed": seed, "checkpoint_step": checkpoint_step}
                row_data.update(standard_summary_data)
                standard_data_list.append(row_data)
                file_had_standard_data = True
                logger.debug(
                    f"Added standard summary row from {json_path.name} (using key '{data_source_key_name}') for step {checkpoint_step}.")
            elif standard_summary_data is not None:  # It was found but not a dict
                logger.warning(
                    f"Step {checkpoint_step}, File {json_path.name}: Expected standard summary to be a dict, "
                    f"but got {type(standard_summary_data)}. Skipping standard section."
                )

            if file_had_priming_data or file_had_standard_data:
                files_processed_successfully += 1
            else:
                logger.warning(
                    f"File {json_path.name} (Step {checkpoint_step}) did not yield any priming or standard data.")

    logger.info(f"Successfully extracted data from {files_processed_successfully} files in {model_dir}.")
    return priming_data_list, standard_data_list


def compile_evaluations_multi_model(
        model_dirs: List[Path],
        output_dir: Path
):
    """
    Compile evaluation data from multiple model directories into a single CSV file.
    """
    all_priming_data = []
    all_standard_data = []
    max_step = -1

    for model_dir in model_dirs:
        logger.info(f"Processing model directory: {model_dir}")
        priming_data, standard_data = process_model_directory(model_dir)

        # Find max step across all models
        if priming_data:
            max_step_priming = max(item["checkpoint_step"] for item in priming_data if "checkpoint_step" in item)
            max_step = max(max_step, max_step_priming)

        if standard_data:
            max_step_standard = max(item["checkpoint_step"] for item in standard_data if "checkpoint_step" in item)
            max_step = max(max_step, max_step_standard)

        all_priming_data.extend(priming_data)
        all_standard_data.extend(standard_data)

    if not all_priming_data and not all_standard_data:
        logger.warning("No evaluation data found in any of the model directories.")
        return

    output_suffix = f"upto_step_{max_step}.csv"
    output_priming_csv = output_dir / f"multi_model_compiled_priming_summary_{output_suffix}"
    output_standard_csv = output_dir / f"multi_model_compiled_standard_summary_{output_suffix}"

    if all_priming_data:
        try:
            priming_df = pd.DataFrame(all_priming_data)
            priming_df = priming_df.sort_values(
                by=["seed", "checkpoint_step", "corpus_file", "contrast_pair", "metric_base"]
            )
            logger.info(
                f"Created Multi-Model Priming Summary DataFrame with {len(priming_df)} rows and {len(priming_df.columns)} columns.")
            priming_df.to_csv(output_priming_csv, index=False, encoding='utf-8')
            logger.info(f"Successfully saved Multi-Model Priming Summary data to: {output_priming_csv}")
        except Exception as e:
            logger.error(f"Failed to create or save Multi-Model Priming Summary DataFrame: {e}")
    else:
        logger.warning("No data found for Multi-Model Priming Summary. CSV file will not be created.")

    if all_standard_data:
        try:
            standard_df = pd.DataFrame(all_standard_data)
            # Ensure all expected columns are present, fill with NaN if not, before sorting
            if "seed" in standard_df.columns and "checkpoint_step" in standard_df.columns:
                standard_df = standard_df.sort_values(by=["seed", "checkpoint_step"])
            elif "checkpoint_step" in standard_df.columns:
                standard_df = standard_df.sort_values(by=["checkpoint_step"])
            else:
                logger.warning("Column 'checkpoint_step' not found in standard_df, cannot sort by it.")

            logger.info(
                f"Created Multi-Model Standard Summary DataFrame with {len(standard_df)} rows and {len(standard_df.columns)} columns.")
            standard_df.to_csv(output_standard_csv, index=False, encoding='utf-8')
            logger.info(f"Successfully saved Multi-Model Standard Summary data to: {output_standard_csv}")
        except Exception as e:
            logger.error(f"Failed to create or save Multi-Model Standard Summary DataFrame: {e}")
    else:
        logger.warning("No data found for Multi-Model Standard Summary. CSV file will not be created.")

    logger.info("Multi-model compilation process finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile evaluation data from multiple model directories into a single CSV file. "
                    "Automatically finds all model directories in the specified superdirectory. "
                    "Extracts seed numbers from directory names and includes them as a column in the output CSV. "
                    "Assumes each directory in the superdirectory is a model with an 'eval' subdirectory."
    )
    parser.add_argument(
        "--superdirectory", type=str, required=True,
        help="Superdirectory containing model directories. Each subdirectory is assumed to be a model with an 'eval' folder."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory where the output CSV files will be saved."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG level logging for detailed output."
    )

    args = parser.parse_args()
    superdirectory_path = Path(args.superdirectory)
    output_path = Path(args.output_dir)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("DEBUG mode enabled.")

    # Validate superdirectory
    if not superdirectory_path.is_dir():
        logger.error(f"Superdirectory not found or is not a directory: {superdirectory_path}")
        exit(1)

    # Find all subdirectories in the superdirectory
    model_paths = [d for d in superdirectory_path.iterdir() if d.is_dir()]
    logger.info(f"Found {len(model_paths)} potential model directories in {superdirectory_path}")

    if not model_paths:
        logger.error(f"No subdirectories found in superdirectory: {superdirectory_path}")
        exit(1)

    # Validate output directory
    if not output_path.exists():
        logger.info(f"Output directory does not exist. Creating: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
    elif not output_path.is_dir():
        logger.error(f"Output path exists but is not a directory: {output_path}")
        exit(1)

    compile_evaluations_multi_model(model_dirs=model_paths, output_dir=output_path)
