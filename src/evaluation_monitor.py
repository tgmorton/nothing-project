# evaluation_monitor.py

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Set, Tuple, Optional, Union
import uuid

# --- Globals ---
logger = None

# --- Helper Functions ---

def setup_monitor_logging(log_level=logging.INFO, log_file=None):
    """Configures basic logging for the monitor script."""
    global logger
    fmt = "%(asctime)s - %(levelname)s - %(name)s (Monitor) - %(message)s"
    dfmt = "%Y-%m-%d %H:%M:%S"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file, mode='a'))
        except OSError as e:
            print(f"Warning (Monitor): Could not create log file handler for {log_file}: {e}")

    logging.basicConfig(level=log_level, format=fmt, datefmt=dfmt, handlers=handlers, force=True)
    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete for evaluation_monitor.py.")

def natural_sort_key(s: str) -> List[Union[str, int]]:
    """Sorts strings with numbers naturally (e.g., checkpoint-10 before checkpoint-100)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def get_checkpoint_step_label_and_numeric(checkpoint_name: str) -> Tuple[str, int]:
    """
    Derives a label and a numeric step from the checkpoint folder name.
    Example: "checkpoint-1000" -> ("checkpoint-1000", 1000)
             "final-model" -> ("final-model", -1)
             "checkpoint-final" -> ("checkpoint-final", -1)
    """
    label = checkpoint_name
    numeric_step = -1 # Default for non-numeric or final
    if checkpoint_name.startswith("checkpoint-"):
        try:
            numeric_step = int(checkpoint_name.split('-')[-1])
        except ValueError:
            # Could be "checkpoint-best" or "checkpoint-final"
            pass
    elif checkpoint_name == "final-model":
        # Often, for time-series logging, final_model might represent the largest step.
        # For now, -1 is a safe default. The evaluate.py script itself handles this.
        pass
    return label, numeric_step


def find_checkpoint_folders(model_parent_dir: Path) -> List[Path]:
    """
    Finds folders matching 'checkpoint-*' or 'final-model' pattern.
    Sorts them naturally: checkpoint-10, checkpoint-100, final-model.
    """
    checkpoints = []
    if not model_parent_dir.is_dir():
        logger.warning(f"Model parent directory not found: {model_parent_dir}")
        return checkpoints

    for item in model_parent_dir.iterdir():
        if item.is_dir():
            if item.name.startswith("checkpoint-") or item.name == "final-model":
                checkpoints.append(item)

    # Sort checkpoints: numeric ones first, then 'final-model' or other named checkpoints
    checkpoints.sort(key=lambda p: (
        0 if p.name.startswith("checkpoint-") and p.name.split('-')[-1].isdigit() else 1, # numeric checkpoints first
        natural_sort_key(p.name)
    ))
    logger.debug(f"Found checkpoint folders: {[p.name for p in checkpoints]}")
    return checkpoints

def load_processed_checkpoints(log_path: Path) -> Set[str]:
    """Loads the set of already processed checkpoint names from a file."""
    processed = set()
    if log_path.exists():
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    processed.add(line.strip())
            logger.info(f"Loaded {len(processed)} processed checkpoint names from {log_path}")
        except Exception as e:
            logger.error(f"Error loading processed checkpoints log {log_path}: {e}")
    return processed

def mark_checkpoint_processed(log_path: Path, checkpoint_name: str):
    """Marks a checkpoint as processed by appending its name to a log file."""
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"{checkpoint_name}\n")
        logger.debug(f"Marked checkpoint '{checkpoint_name}' as processed in {log_path}")
    except Exception as e:
        logger.error(f"Error marking checkpoint {checkpoint_name} in {log_path}: {e}")

def run_evaluation_for_checkpoint(
    monitor_args: argparse.Namespace,
    checkpoint_path: Path,
    shared_run_id: Optional[str] = None,
    training_run_name: Optional[str] = None
):
    """
    Constructs and runs the evaluate.py command for a single checkpoint.
    """
    checkpoint_name = checkpoint_path.name
    checkpoint_label, _ = get_checkpoint_step_label_and_numeric(checkpoint_name) # numeric_step will be recalculated by evaluate.py

    # Define a unique output directory for this specific evaluation run
    eval_output_dir = monitor_args.output_base_dir / f"eval_results_{checkpoint_name}"
    try:
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured output directory for {checkpoint_name}: {eval_output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory {eval_output_dir} for {checkpoint_name}: {e}. Skipping.")
        return False


    # --- Construct the command for evaluate.py ---
    # Ensure evaluate_script_path is correctly pointing to your evaluate.py
    # If evaluate.py is in the same directory as evaluation_monitor.py:
    evaluate_script_path = Path(__file__).parent / "evaluate.py"
    # Or, if it's elsewhere, provide the correct path:
    # evaluate_script_path = Path("/path/to/your/evaluate.py")

    if not evaluate_script_path.exists():
        logger.critical(f"evaluate.py script not found at {evaluate_script_path}. Please check the path.")
        return False

    cmd = [
        sys.executable,  # Use the same Python interpreter
        str(evaluate_script_path),
        "--checkpoint_path", str(checkpoint_path),
        "--output_dir", str(eval_output_dir),
        "--checkpoint_label", checkpoint_label, # evaluate.py uses this for logging and file names
        # Pass through other arguments from monitor_args to evaluate_args
        "--per_device_eval_batch_size", str(monitor_args.per_device_eval_batch_size),
        "--eval_max_samples", str(monitor_args.eval_max_samples),
        "--model_class_name", monitor_args.model_class_name,
        "--base_model_name", monitor_args.base_model_name,
        "--num_workers", str(monitor_args.num_workers),
        "--seed", str(monitor_args.seed),
    ]

    if monitor_args.priming_per_device_eval_batch_size is not None:
        cmd.extend(["--priming_per_device_eval_batch_size", str(monitor_args.priming_per_device_eval_batch_size)])
    cmd.extend(["--priming_eval_max_samples_per_file", str(monitor_args.priming_eval_max_samples_per_file)])
    cmd.extend(["--priming_delimiter", monitor_args.priming_delimiter])


    if monitor_args.run_standard_eval:
        cmd.append("--run_standard_eval")
        if monitor_args.validation_dataset_path:
            cmd.extend(["--validation_dataset_path", str(monitor_args.validation_dataset_path)])
        else:
            logger.warning("Monitor: --run_standard_eval is set, but --validation_dataset_path is not. "
                           "evaluate.py might error if it requires it.")

    if monitor_args.run_priming_eval:
        cmd.append("--run_priming_eval")
        if monitor_args.priming_eval_dir_path:
            cmd.extend(["--priming_eval_dir_path", str(monitor_args.priming_eval_dir_path)])
        else:
            logger.warning("Monitor: --run_priming_eval is set, but --priming_eval_dir_path is not. "
                           "evaluate.py might error if it requires it.")

    if monitor_args.use_amp:
        cmd.append("--use_amp")

    # Neptune related arguments
    if monitor_args.neptune_project:
        cmd.extend(["--neptune_project", monitor_args.neptune_project])
        if monitor_args.neptune_api_token:
            cmd.extend(["--neptune_api_token", monitor_args.neptune_api_token])
        if monitor_args.neptune_tags:
            cmd.extend(["--neptune_tags"] + monitor_args.neptune_tags)

        # Construct a Neptune run name for this specific evaluation
        # The evaluate.py script will create a *new* run for this checkpoint.
        # We can help group them by providing a consistent prefix via SHARED_RUN_ID or similar.
        neptune_run_name_for_eval_script = f"Eval_{shared_run_id or 'standalone'}_{checkpoint_label}"
        cmd.extend(["--neptune_run_name", neptune_run_name_for_eval_script])
        # If you have an existing neptune_run_id for the *training* run and want to link to it,
        # you could pass it as an environment variable for evaluate.py to pick up,
        # or if evaluate.py supports an arg like --linked_training_run_id.
        # The current evaluate.py looks for NEPTUNE_TRAINING_RUN_NAME.

    logger.info(f"--- Starting evaluation for checkpoint: {checkpoint_name} ---")
    logger.info(f"Output will be in: {eval_output_dir}")
    logger.info(f"Running command: {' '.join(cmd)}")

    # Prepare environment for the subprocess
    sub_env = os.environ.copy()
    if shared_run_id:
        sub_env["SHARED_RUN_ID"] = shared_run_id # evaluate.py can use this for naming/tagging
    if training_run_name: # If known, pass the original training run name
        sub_env["NEPTUNE_TRAINING_RUN_NAME"] = training_run_name
        sub_env["SINGULARITYENV_NEPTUNE_TRAINING_RUN_NAME"] = training_run_name # For Singularity

    process = None
    try:
        # Use Popen to stream output if desired, or run for simplicity if not.
        # For detailed logging from evaluate.py, Popen is better.
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', env=sub_env)

        # Stream stdout and stderr
        # Give a prefix to distinguish evaluate.py logs from monitor logs
        eval_log_prefix = f"[{checkpoint_name} eval.py] "
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                logger.info(f"{eval_log_prefix}{line.strip()}")
        if process.stderr:
            for line in iter(process.stderr.readline, ''):
                logger.error(f"{eval_log_prefix}{line.strip()}")

        process.wait() # Wait for the subprocess to complete

        if process.returncode == 0:
            logger.info(f"--- Successfully completed evaluation for: {checkpoint_name} ---")
            return True
        else:
            logger.error(f"--- Evaluation FAILED for: {checkpoint_name} (return code: {process.returncode}) ---")
            # The evaluate.py script should log its own errors to its specific log file.
            return False

    except FileNotFoundError:
        logger.critical(f"CRITICAL: Python interpreter or evaluate.py script not found. "
                        f"Ensure '{sys.executable}' and '{evaluate_script_path}' are valid.")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation script for {checkpoint_name} failed with CalledProcessError: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while running evaluation for {checkpoint_name}: {e}", exc_info=True)
        return False
    finally:
        if process and process.poll() is None: # if process is still running (e.g. due to external interrupt)
            logger.warning(f"Terminating evaluation process for {checkpoint_name} due to script exit.")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def main_monitor():
    global logger
    parser = argparse.ArgumentParser(description="Monitors a directory for model checkpoints and runs evaluation.")

    # --- Monitor Specific Arguments ---
    parser.add_argument("--model_parent_dir", type=Path, required=True,
                        help="Parent directory containing checkpoint folders (e.g., 'checkpoint-N', 'final-model').")
    parser.add_argument("--output_base_dir", type=Path, required=True,
                        help="Base directory to save all evaluation outputs. Subdirectories will be created for each checkpoint.")
    parser.add_argument("--watch", action="store_true",
                        help="Enable watch mode to continuously monitor for new checkpoints.")
    parser.add_argument("--poll_interval", type=int, default=300,
                        help="Polling interval in seconds for watch mode (default: 300s = 5min).")
    parser.add_argument("--skip_processed_checkpoints", action="store_true", default=True,
                        help="Skip checkpoints already listed in the processed log (default: True).")
    parser.add_argument("--no_skip_processed_checkpoints", action="store_false", dest="skip_processed_checkpoints",
                        help="Force re-evaluation of all found checkpoints.")
    parser.add_argument("--processed_checkpoints_log", type=Path, default=None,
                        help="Path to a log file for tracking processed checkpoints. Defaults to 'output_base_dir/_processed_checkpoints.log'.")
    parser.add_argument("--monitor_log_file", type=Path, default=None,
                        help="Optional path to a log file for the monitor script itself. Defaults to 'output_base_dir/evaluation_monitor.log'.")
    parser.add_argument("--shared_run_id_prefix", type=str, default="train",
                        help="Prefix for the SHARED_RUN_ID. A UUID will be appended if not in watch mode and this is default. Otherwise, derived from model_parent_dir.")
    parser.add_argument("--neptune_training_run_name", type=str, default=None,
                        help="Optional: The name/ID of the Neptune run for the *training* session, to link evaluations back.")


    # --- Arguments to be passed through to evaluate.py ---
    # These should mirror the arguments in evaluate.py, excluding checkpoint_path, output_dir, checkpoint_label
    # (which are handled by the monitor)
    eval_group = parser.add_argument_group("evaluate.py Passthrough Arguments")
    eval_group.add_argument("--run_standard_eval", action="store_true", default=False)
    eval_group.add_argument("--run_priming_eval", action="store_true", default=False)
    eval_group.add_argument("--validation_dataset_path", type=Path, default=None)
    eval_group.add_argument("--priming_eval_dir_path", type=Path, default=None)
    eval_group.add_argument("--per_device_eval_batch_size", type=int, default=16)
    eval_group.add_argument("--priming_per_device_eval_batch_size", type=int, default=None,
                            help="Defaults to per_device_eval_batch_size in evaluate.py if not set")
    eval_group.add_argument("--eval_max_samples", type=int, default=50000)
    eval_group.add_argument("--priming_eval_max_samples_per_file", type=int, default=1000)
    eval_group.add_argument("--priming_delimiter", type=str, default=".")
    eval_group.add_argument("--model_class_name", type=str, default="GPT2LMHeadModel")
    eval_group.add_argument("--base_model_name", type=str, default="gpt2")
    eval_group.add_argument("--use_amp", action="store_true", default=False)
    eval_group.add_argument("--num_workers", type=int, default=4)
    eval_group.add_argument("--seed", type=int, default=42)
    # Neptune related for evaluate.py
    eval_group.add_argument("--neptune_project", type=str, default=None)
    eval_group.add_argument("--neptune_api_token", type=str, default=None, help="Can also be set via NEPTUNE_API_TOKEN env var.")
    eval_group.add_argument("--neptune_tags", type=str, nargs='+', default=None, help="Additional tags for evaluate.py's Neptune run.")


    monitor_args = parser.parse_args()

    # --- Setup monitor logging ---
    if monitor_args.monitor_log_file is None:
        monitor_args.monitor_log_file = monitor_args.output_base_dir / "evaluation_monitor.log"
    setup_monitor_logging(log_file=monitor_args.monitor_log_file) # global logger gets configured

    logger.info(f"***** Starting Evaluation Monitor *****")
    logger.info(f"Monitor Arguments: {vars(monitor_args)}")

    # --- Validate crucial paths for passthrough arguments ---
    if monitor_args.run_standard_eval and not monitor_args.validation_dataset_path:
        logger.error("--run_standard_eval is set, but --validation_dataset_path is missing. This might cause errors in evaluate.py.")
        # Potentially exit or just warn, depending on desired strictness
        # sys.exit(1)
    if monitor_args.run_priming_eval and not monitor_args.priming_eval_dir_path:
        logger.error("--run_priming_eval is set, but --priming_eval_dir_path is missing. This might cause errors in evaluate.py.")
        # sys.exit(1)


    # --- Processed checkpoints log ---
    if monitor_args.processed_checkpoints_log is None:
        monitor_args.processed_checkpoints_log = monitor_args.output_base_dir / "_processed_checkpoints.log"
    try:
        monitor_args.output_base_dir.mkdir(parents=True, exist_ok=True)
        monitor_args.processed_checkpoints_log.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.critical(f"Failed to create base output directory or processed log directory: {e}")
        sys.exit(1)

    # --- Shared Run ID for grouping evaluations (e.g., in Neptune) ---
    # This ID can be used by evaluate.py to name its runs consistently
    shared_run_id = None
    if monitor_args.neptune_project:
        if monitor_args.watch or monitor_args.shared_run_id_prefix != "train":
             # Use a more stable ID based on the model directory for watch mode or if prefix is custom
            clean_parent_dir_name = re.sub(r'[^a-zA-Z0-9_-]', '_', monitor_args.model_parent_dir.resolve().name)
            shared_run_id = f"{monitor_args.shared_run_id_prefix}_{clean_parent_dir_name}"
        else:
            # For a single run with default prefix, a UUID might be fine if that's preferred
            shared_run_id = f"{monitor_args.shared_run_id_prefix}_{str(uuid.uuid4())[:8]}"
        logger.info(f"Using SHARED_RUN_ID: {shared_run_id} for grouping evaluation runs (e.g., in Neptune).")


    # --- Main Loop ---
    try:
        while True:
            logger.info(f"Scanning for checkpoints in: {monitor_args.model_parent_dir}")
            all_checkpoints = find_checkpoint_folders(monitor_args.model_parent_dir)

            if not all_checkpoints:
                logger.info("No checkpoint folders found.")
            else:
                processed_checkpoints = set()
                if monitor_args.skip_processed_checkpoints:
                    processed_checkpoints = load_processed_checkpoints(monitor_args.processed_checkpoints_log)

                checkpoints_to_evaluate = [
                    ckpt for ckpt in all_checkpoints if ckpt.name not in processed_checkpoints
                ]

                if not checkpoints_to_evaluate:
                    logger.info("No new (unprocessed) checkpoints found to evaluate.")
                else:
                    logger.info(f"Found {len(checkpoints_to_evaluate)} new checkpoints to evaluate: "
                                f"{[ckpt.name for ckpt in checkpoints_to_evaluate]}")

                    for checkpoint_path in checkpoints_to_evaluate:
                        if run_evaluation_for_checkpoint(
                            monitor_args,
                            checkpoint_path,
                            shared_run_id=shared_run_id,
                            training_run_name=monitor_args.neptune_training_run_name
                        ):
                            if monitor_args.skip_processed_checkpoints:
                                mark_checkpoint_processed(monitor_args.processed_checkpoints_log, checkpoint_path.name)
                        else:
                            logger.warning(f"Evaluation for {checkpoint_path.name} reported failure. "
                                           "It will NOT be marked as processed and may be retried next cycle if in watch mode.")
                        # Optional: Add a small delay between evaluations if needed
                        # time.sleep(5)

            if not monitor_args.watch:
                logger.info("Watch mode disabled. Exiting after one scan.")
                break

            logger.info(f"Watch mode enabled. Sleeping for {monitor_args.poll_interval} seconds...")
            time.sleep(monitor_args.poll_interval)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down monitor.")
    except Exception as e:
        logger.critical(f"An unhandled exception occurred in the monitor: {e}", exc_info=True)
    finally:
        logger.info("***** Evaluation Monitor Finished *****")

if __name__ == "__main__":
    # This basicConfig is a fallback. setup_monitor_logging in main_monitor will override.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s (Monitor) - %(message)s")
    main_monitor()