import argparse
import json
import logging
import pandas as pd
from pathlib import Path
import re

# --- Configuration ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Helper Function ---

def extract_step_from_path(file_path: Path) -> int | None:
    """
    Extracts the checkpoint step number from the filename.
    Assumes filename format like '..._step_12345.json'.
    """
    # Updated regex to be slightly more flexible, finding '_step_NUMBER' before '.json'
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
    extracts priming and standard summaries, compiles them into DataFrames,
    and saves them within the input_dir using filenames based on the highest step found.

    Args:
        input_dir: The root directory containing evaluation output folders/files.
                   This is also where the output CSVs will be saved.
        json_filename_pattern: Glob pattern to find the summary JSON files.
    """
    priming_data_list = []
    standard_data_list = []
    files_processed = 0
    files_found = 0
    max_step = -1 # Initialize to track the highest step number found

    logger.info(f"Starting compilation process in directory: {input_dir}")
    logger.info(f"Searching recursively for files matching: {json_filename_pattern}")

    # Use rglob to search recursively
    for json_path in input_dir.rglob(json_filename_pattern):
        files_found += 1
        logger.debug(f"Found potential file: {json_path}") # Use debug for less verbose logging

        # Extract checkpoint step from filename
        checkpoint_step = extract_step_from_path(json_path)
        if checkpoint_step is None:
            logger.warning(f"Skipping file due to missing/invalid step number: {json_path}")
            continue

        # Update max_step found so far
        max_step = max(max_step, checkpoint_step)
        logger.debug(f"Processing step {checkpoint_step} from file {json_path.name}. Current max step: {max_step}")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON in file: {json_path}. Skipping.")
            continue
        except IOError as e:
            logger.error(f"Error reading file {json_path}: {e}. Skipping.")
            continue
        except Exception as e:
            logger.error(f"An unexpected error occurred reading {json_path}: {e}. Skipping.")
            continue

        # --- Process 'priming_summary' section ---
        priming_summary = data.get("priming_summary")
        if isinstance(priming_summary, dict):
            for csv_filename, metrics in priming_summary.items():
                if isinstance(metrics, dict):
                    row_data = {
                        "checkpoint_step": checkpoint_step,
                        "corpus_file": csv_filename,
                    }
                    row_data.update(metrics)
                    priming_data_list.append(row_data)
                else:
                    logger.warning(
                        f"Step {checkpoint_step}: Expected dict for metrics under '{csv_filename}' "
                        f"in 'priming_summary', but got {type(metrics)}. Skipping entry."
                    )
        elif priming_summary is not None:
            logger.warning(
                f"Step {checkpoint_step}: Expected 'priming_summary' to be a dict, "
                f"but got {type(priming_summary)}. Skipping section."
            )

        # --- Process 'standard_summary' section ---
        standard_summary = data.get("standard_summary")
        if isinstance(standard_summary, dict):
            row_data = {
                "checkpoint_step": checkpoint_step,
            }
            row_data.update(standard_summary)
            standard_data_list.append(row_data)
        elif standard_summary is not None:
            logger.warning(
                f"Step {checkpoint_step}: Expected 'standard_summary' to be a dict, "
                f"but got {type(standard_summary)}. Skipping section."
            )

        files_processed += 1
        logger.info(f"Successfully processed step {checkpoint_step} from {json_path.name}")


    logger.info(f"Found {files_found} files matching pattern.")
    if files_processed == 0:
        logger.warning("No valid evaluation summary files were processed. No CSV files will be created.")
        return # Exit the function early if nothing was processed

    logger.info(f"Successfully processed data from {files_processed} files. Highest checkpoint step found: {max_step}")

    # --- Determine Output Filenames and Save DataFrames ---
    output_suffix = f"upto_step_{max_step}.csv"
    output_priming_csv = input_dir / f"compiled_priming_summary_{output_suffix}"
    output_standard_csv = input_dir / f"compiled_standard_summary_{output_suffix}"

    # Priming Summary DataFrame
    if priming_data_list:
        try:
            priming_df = pd.DataFrame(priming_data_list)
            priming_df = priming_df.sort_values(by=["checkpoint_step", "corpus_file"])
            logger.info(f"Created Priming Summary DataFrame with {len(priming_df)} rows and {len(priming_df.columns)} columns.")
            priming_df.to_csv(output_priming_csv, index=False, encoding='utf-8')
            logger.info(f"Successfully saved Priming Summary data to: {output_priming_csv}")
        except Exception as e:
            logger.error(f"Failed to create or save Priming Summary DataFrame: {e}")
    else:
        logger.warning("No data found for Priming Summary despite processing files. CSV file will not be created.")

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
        logger.warning("No data found for Standard Summary despite processing files. CSV file will not be created.")

    logger.info("Compilation process finished.")


# --- Command Line Argument Parsing ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile evaluation summary JSON files from multiple checkpoint runs into CSVs. "
                    "Output CSVs are saved in the input directory with filenames indicating the highest step found."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="The root directory containing the evaluation output folders or JSON files. "
             "The script will search recursively. Output CSVs will also be saved here."
    )
    # Removed output path arguments
    # parser.add_argument(
    #     "--output_priming_csv", ...
    # )
    # parser.add_argument(
    #     "--output_standard_csv", ...
    # )
    parser.add_argument(
        "--filename_pattern",
        type=str,
        default="evaluation_summary_step_*.json",
        help="Glob pattern used to find the JSON summary files within the input directory."
    )

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    # Output paths are now determined inside the function

    if not input_path.is_dir():
        logger.error(f"Input directory not found or is not a directory: {input_path}")
        exit(1)

    # Call the function with only the necessary arguments
    compile_evaluation_summaries(
        input_dir=input_path,
        json_filename_pattern=args.filename_pattern
    )