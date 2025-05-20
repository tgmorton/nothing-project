# evaluate.py

import argparse
import logging
import math
import json
import os
import sys
import time
import traceback
import gc
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Standard ML/data imports
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,  # Assuming this is the model class used
    PreTrainedTokenizer,
    BatchEncoding
)
from datasets import load_from_disk
from torch.cuda.amp import autocast  # For evaluate_standard AMP context

# Priming evaluation imports
try:
    from priming_evaluation.data_loader import create_priming_dataloader
    from priming_evaluation.evaluator import run_native_priming_eval

    PRIMING_LIBS_AVAILABLE = True
except ImportError as e:
    # This print will go to stderr, visible in Slurm error logs
    print(f"Warning: Failed to import priming_evaluation modules: {e}. Priming evaluation will be skipped.",
          file=sys.stderr)
    create_priming_dataloader = None
    run_native_priming_eval = None
    PRIMING_LIBS_AVAILABLE = False

# Optional Neptune import
try:
    import neptune

    NEPTUNE_AVAILABLE = True
except ImportError:
    neptune = None
    NEPTUNE_AVAILABLE = False
    print("Neptune.ai library not found, Neptune logging will be disabled for evaluation.", file=sys.stderr)

# --- Globals ---
logger = None
tqdm = None  # Will be imported or polyfilled in main


# --- Helper Functions ---

def get_device():
    """Gets the appropriate device for PyTorch computations."""
    global logger
    dev_str = "cpu"
    dev_name = "CPU"
    if torch.backends.mps.is_available():
        dev_str = "mps"
        dev_name = "MPS"
    elif torch.cuda.is_available():
        dev_str = "cuda:0"  # Default to first GPU for evaluation
        try:
            dev_name = torch.cuda.get_device_name(0)
        except Exception:
            dev_name = "CUDA GPU"

    device = torch.device(dev_str)

    if logger:
        logger.info(f"Using device: {device} ({dev_name})")
    else:
        print(f"Using device: {device} ({dev_name})")
    return device


def setup_logging(log_level=logging.INFO, log_file=None):
    """Configures basic logging, optionally to a file."""
    global logger
    fmt = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s"
    dfmt = "%Y-%m-%d %H:%M:%S"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file, mode='a'))  # Append mode
        except OSError as e:
            print(f"Warning: Could not create log file handler for {log_file}: {e}")

    logging.basicConfig(level=log_level, format=fmt, datefmt=dfmt, handlers=handlers, force=True)
    logger = logging.getLogger(__name__)  # Get the logger for this module
    logger.info("Logging setup complete for evaluation script.")


def set_seed(seed_value):
    """Sets random seeds."""
    import random
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    if logger:
        logger.info(f"Set random seed: {seed_value}")
    else:
        print(f"Set random seed: {seed_value}")


def load_model_for_evaluation(model_class, checkpoint_path: str, base_model_name="gpt2"):
    """Loads model, tokenizer, and config from checkpoint for evaluation."""
    global logger
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_dir():
        log_msg = f"Checkpoint directory not found: {ckpt_path}"
        if logger:
            logger.error(log_msg)
        else:
            print(f"ERROR: {log_msg}")
        raise FileNotFoundError(log_msg)

    if logger: logger.info(f"Loading tokenizer and config from checkpoint: {ckpt_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=True)
        config = AutoConfig.from_pretrained(ckpt_path)
        if logger: logger.info(f"Loaded tokenizer and config successfully from checkpoint directory: {ckpt_path}")
    except OSError:
        if logger: logger.warning(f"Tokenizer/config not found directly in checkpoint {ckpt_path}. "
                                  f"Falling back to loading from base model name: {base_model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
            config = AutoConfig.from_pretrained(base_model_name)
        except Exception as e:
            if logger: logger.error(f"Fallback loading of tokenizer/config from base '{base_model_name}' failed: {e}",
                                    exc_info=True)
            raise

    if logger: logger.info(f"Loading model weights for '{config.model_type}' from checkpoint: {ckpt_path}")
    try:
        model = model_class.from_pretrained(ckpt_path, config=config)
        if logger: logger.info("Successfully loaded model weights from checkpoint.")
    except Exception as e:
        if logger: logger.error(f"Failed to load model weights from checkpoint {ckpt_path}: {e}", exc_info=True)
        raise

    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            if logger: logger.info(f"Set tokenizer pad_token to its eos_token ('{tokenizer.eos_token}')")
        else:
            if logger: logger.warning(
                "Tokenizer has no pad_token and no eos_token. This might cause issues with padding during collation.")
            # For evaluation, avoid modifying the tokenizer vocabulary if possible.
            # If padding is essential and no pad_token exists, it's an issue with the original tokenizer.
    return model, tokenizer, config


def evaluate_standard(args, model, eval_dataloader, device):
    """Runs standard evaluation (perplexity)."""
    global logger, tqdm
    if eval_dataloader is None:
        logger.warning("Standard evaluation dataloader is None. Skipping standard perplexity eval.")
        return {}

    original_mode = model.training  # Save original mode
    model.eval()
    total_loss = 0.0
    total_items = 0

    logger.info("Starting standard perplexity evaluation...")
    progress_bar = tqdm(eval_dataloader, desc="Eval (Std Perplexity)", leave=False, disable=not sys.stdout.isatty())

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            try:
                batch_on_device = {k: v.to(device, non_blocking=True) for k, v in batch.items() if
                                   isinstance(v, torch.Tensor)}
            except RuntimeError as e:
                logger.error(f"Error moving standard eval batch {batch_idx} to device {device}: {e}")
                continue  # Skip this batch

            try:
                amp_enabled = args.use_amp and device.type == 'cuda'
                with autocast(enabled=amp_enabled):
                    outputs = model(**batch_on_device)
                    loss = outputs.loss
            except Exception as e:
                logger.error(f"Error during standard eval forward pass on batch {batch_idx}: {e}", exc_info=True)
                continue  # Skip this batch

            if loss is not None and torch.isfinite(loss):
                num_items_in_batch = batch_on_device['input_ids'].size(0)
                total_loss += loss.detach().item() * num_items_in_batch  # Accumulate sum of losses
                total_items += num_items_in_batch
                if batch_idx % 50 == 0:  # Log periodically
                    progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
            else:
                logger.warning(
                    f"Non-finite loss detected in standard eval (batch {batch_idx}): {loss.item() if loss is not None else 'None'}. Skipping contribution.")

    metrics = {}
    if total_items > 0:
        final_avg_loss = total_loss / total_items
        try:
            # Cap perplexity calculation to avoid math.exp overflow with very high losses
            perplexity = math.exp(min(final_avg_loss, 700)) if final_avg_loss > 0 else float('nan')
        except (OverflowError, ValueError):  # Should be caught by min with 700
            perplexity = float('inf') if final_avg_loss > 0 else float('nan')

        logger.info(
            f"Standard Evaluation Results: Average Loss = {final_avg_loss:.4f}, Perplexity = {perplexity:.4f}, Total Items Evaluated = {total_items}")
        metrics = {"loss": final_avg_loss, "perplexity": perplexity, "total_items": total_items}
    else:
        logger.warning("Standard evaluation completed, but total_items processed is zero. No metrics calculated.")
        metrics = {"loss": float('nan'), "perplexity": float('nan'), "total_items": 0}

    if original_mode: model.train()  # Restore original model mode
    progress_bar.close()
    return metrics


def run_priming_evaluation_on_directory(
        eval_args, model, tokenizer, device, neptune_run_obj, checkpoint_numeric_step
):
    """
    Finds CSVs, creates dataloaders, runs priming eval, aggregates summary results,
    appends raw per-item results to a persistent CSV, and logs to Neptune.
    """
    global logger, tqdm
    if not PRIMING_LIBS_AVAILABLE:
        logger.error("Priming evaluation libraries were not imported successfully. Skipping priming evaluation.")
        return {"error": "Priming library import failed."}

    if not eval_args.run_priming_eval or not eval_args.priming_eval_dir_path:
        logger.info("Skipping priming evaluation (not enabled by args or path not provided).")
        return {}

    priming_dir = Path(eval_args.priming_eval_dir_path)
    if not priming_dir.is_dir():
        logger.error(f"Priming evaluation directory not found: {priming_dir}")
        return {"error": f"Priming directory not found: {priming_dir}"}

    # Output directory for this specific evaluation run (already created by main)
    eval_specific_output_dir = Path(eval_args.output_dir)
    # Make raw priming CSV name specific to the checkpoint step
    raw_priming_csv_path = eval_specific_output_dir / f"priming_results_raw_step_{checkpoint_numeric_step}.csv"
    logger.info(f"Raw priming results CSV will be written/overwritten at: {raw_priming_csv_path}")

    csv_files_in_priming_dir = sorted(list(priming_dir.glob('*.csv')))
    if not csv_files_in_priming_dir:
        logger.warning(f"No *.csv files found in priming directory: {priming_dir}")
        return {}

    all_corpus_summary_metrics = {}  # To store summary metrics for each corpus file
    neptune_metrics_to_log_this_step = {}  # Aggregated metrics for Neptune logging
    logger.info(f"Found {len(csv_files_in_priming_dir)} CSV files for priming evaluation in {priming_dir}.")

    original_model_training_mode = model.training  # Save original mode
    model.eval()  # Ensure model is in evaluation mode

    # Setup CSV writer for raw results
    csv_file_handle, csv_writer_obj = None, None
    try:
        # 'w' mode creates a new CSV for this specific checkpoint's evaluation run
        csv_file_handle = open(raw_priming_csv_path, 'w', newline='', encoding='utf-8')
        csv_writer_obj = csv.writer(csv_file_handle)
        # MODIFIED CSV HEADER
        csv_header = [
            "eval_step", "corpus_file", "target_structure", "item_index",
            "pe", "logp_con", "logp_incon", "logp_baseline",
            "logp_con_random_baseline", "logp_incon_random_baseline"
        ]
        csv_writer_obj.writerow(csv_header)
    except IOError as e:
        logger.error(f"Failed to open or write header to raw priming CSV {raw_priming_csv_path}: {e}")
        if csv_file_handle: csv_file_handle.close()  # Attempt to close if opened
        return {"error": f"Raw priming CSV write error: {e}"}

    # Process each priming corpus file
    for corpus_csv_path in csv_files_in_priming_dir:
        corpus_filename = corpus_csv_path.name
        logger.info(
            f"--- Running Priming Evaluation for Corpus: {corpus_filename} (Checkpoint Step {checkpoint_numeric_step}) ---")

        current_corpus_dataloader = None
        try:
            current_corpus_dataloader = create_priming_dataloader(
                csv_path=str(corpus_csv_path), tokenizer=tokenizer,
                batch_size=eval_args.priming_per_device_eval_batch_size,
                delimiter=eval_args.priming_delimiter, num_workers=eval_args.num_workers,
                pin_memory=True, max_samples=eval_args.priming_eval_max_samples_per_file,
                seed=eval_args.seed  # Seed for dataloader shuffling/sampling if applicable
            )
        except Exception as e:
            logger.error(f"Dataloader creation failed for priming corpus {corpus_filename}: {e}", exc_info=True)
            all_corpus_summary_metrics[corpus_filename] = {"error": f"Dataloader creation failed: {e}"}
            continue  # Skip to next corpus file

        if current_corpus_dataloader is None or len(current_corpus_dataloader.dataset) == 0:
            logger.warning(f"Dataloader for priming corpus {corpus_filename} is None or empty. Skipping this corpus.")
            all_corpus_summary_metrics[corpus_filename] = {"error": "Dataloader was None or empty."}
            continue

        try:
            # Run the native priming evaluation
            # MODIFIED: Pass random_seed
            corpus_summary_metrics, corpus_raw_item_results = run_native_priming_eval(
                model=model, priming_dataloader=current_corpus_dataloader, device=device,
                tokenizer=tokenizer, use_amp=eval_args.use_amp,
                random_seed=eval_args.seed  # Pass the main evaluation seed here
            )
            all_corpus_summary_metrics[corpus_filename] = corpus_summary_metrics
            logger.info(f"Priming Summary Metrics for {corpus_filename}: {corpus_summary_metrics}")

            # Aggregate metrics for Neptune logging
            if NEPTUNE_AVAILABLE and neptune_run_obj:
                # Sanitize corpus_filename for use in Neptune path
                sanitized_corpus_name = corpus_filename.replace('.', '_').replace('/', '_')
                neptune_log_prefix = f"eval_metrics/priming/{sanitized_corpus_name}"
                # Filter for finite numeric metrics to log
                finite_numeric_metrics = {k: v for k, v in corpus_summary_metrics.items() if
                                          isinstance(v, (int, float)) and math.isfinite(v)}
                for metric_key, metric_value in finite_numeric_metrics.items():
                    neptune_metrics_to_log_this_step[f"{neptune_log_prefix}/{metric_key}"] = metric_value

            # Write Raw Item Results to the single CSV file for this checkpoint evaluation
            if csv_writer_obj and corpus_raw_item_results:
                raw_items_written_for_corpus = 0
                try:
                    for target_structure_key, per_structure_results_list in corpus_raw_item_results.items():
                        for item_idx, item_data_dict in enumerate(per_structure_results_list):
                            if isinstance(item_data_dict, dict):  # Ensure item_data is a dictionary
                                pe_val = item_data_dict.get('pe', float('nan'))
                                logp_con_val = item_data_dict.get('logp_con', float('nan'))
                                logp_incon_val = item_data_dict.get('logp_incon', float('nan'))
                                # MODIFIED: Get new baseline values
                                logp_baseline_val = item_data_dict.get('logp_baseline', float('nan'))
                                logp_con_rb_val = item_data_dict.get('logp_con_random_baseline', float('nan'))
                                logp_incon_rb_val = item_data_dict.get('logp_incon_random_baseline', float('nan'))

                                # MODIFIED: Write new values to CSV
                                csv_writer_obj.writerow([
                                    checkpoint_numeric_step, corpus_filename, target_structure_key, item_idx,
                                    f"{pe_val:.6f}" if not math.isnan(pe_val) else 'NaN',
                                    f"{logp_con_val:.6f}" if not math.isnan(logp_con_val) else 'NaN',
                                    f"{logp_incon_val:.6f}" if not math.isnan(logp_incon_val) else 'NaN',
                                    f"{logp_baseline_val:.6f}" if not math.isnan(logp_baseline_val) else 'NaN',
                                    f"{logp_con_rb_val:.6f}" if not math.isnan(logp_con_rb_val) else 'NaN',
                                    f"{logp_incon_rb_val:.6f}" if not math.isnan(logp_incon_rb_val) else 'NaN'
                                ])
                                raw_items_written_for_corpus += 1
                            else:
                                logger.warning(
                                    f"Skipping item in raw priming results for {corpus_filename}, {target_structure_key}, index {item_idx} due to unexpected data type: {type(item_data_dict)}")
                    if csv_file_handle: csv_file_handle.flush()  # Flush after processing each corpus file's results
                    logger.info(
                        f"Successfully wrote {raw_items_written_for_corpus} raw priming results to CSV for corpus {corpus_filename}.")
                except Exception as e_csv_write:
                    logger.error(
                        f"Error occurred while writing raw priming results to CSV for {corpus_filename}: {e_csv_write}",
                        exc_info=True)
        except Exception as e_eval_run:
            logger.error(f"Priming evaluation run failed catastrophically for corpus {corpus_filename}: {e_eval_run}",
                         exc_info=True)
            all_corpus_summary_metrics[corpus_filename] = {"error": f"Core evaluation run failed: {e_eval_run}"}
        finally:
            del current_corpus_dataloader;
            gc.collect()  # Explicitly delete and collect garbage

    # Log all aggregated priming metrics for this checkpoint step to Neptune *once*
    if NEPTUNE_AVAILABLE and neptune_run_obj and neptune_metrics_to_log_this_step:
        logger.info(
            f"Logging {len(neptune_metrics_to_log_this_step)} aggregated priming metrics to Neptune for checkpoint step {checkpoint_numeric_step}...")
        try:
            for full_metric_path, value_to_log in neptune_metrics_to_log_this_step.items():
                # Use append for time-series data, using the numeric checkpoint_numeric_step
                neptune_run_obj[full_metric_path].append(value_to_log, step=checkpoint_numeric_step)
            logger.info(
                f"Successfully logged aggregated priming metrics to Neptune for checkpoint step {checkpoint_numeric_step}.")
        except Exception as e_neptune_log:
            logger.warning(
                f"Neptune logging for aggregated priming metrics failed (checkpoint step {checkpoint_numeric_step}): {e_neptune_log}")
    elif NEPTUNE_AVAILABLE and neptune_run_obj:
        logger.info(
            f"No valid priming summary metrics were aggregated for Neptune logging at checkpoint step {checkpoint_numeric_step}.")

    # Close the main CSV file for raw results
    if csv_file_handle:
        try:
            csv_file_handle.close();
            logger.info(f"Closed raw priming results CSV file: {raw_priming_csv_path}")
        except Exception as e_close:
            logger.error(f"Error closing raw priming results CSV file {raw_priming_csv_path}: {e_close}")

    if original_model_training_mode: model.train()  # Restore original model mode
    logger.info(f"--- Finished All Priming Evaluations for Checkpoint Step {checkpoint_numeric_step} ---")
    return all_corpus_summary_metrics


# --- Argument Parser for evaluate.py ---
def parse_eval_args():
    """Parses command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate a trained GPT-2 like model checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the specific checkpoint directory to evaluate.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save all evaluation results (logs, JSON summary, raw CSVs).")
    parser.add_argument("--run_standard_eval", action="store_true", default=False,
                        help="Run standard perplexity evaluation.")
    parser.add_argument("--run_priming_eval", action="store_true", default=False,
                        help="Run priming evaluation from a directory of CSVs.")
    parser.add_argument("--validation_dataset_path", type=str, default=None,
                        help="Path to the validation Arrow dataset (for --run_standard_eval).")
    parser.add_argument("--priming_eval_dir_path", type=str, default=None,
                        help="Directory containing priming CSVs (for --run_priming_eval).")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16,
                        help="Batch size for standard perplexity evaluation.")
    parser.add_argument("--priming_per_device_eval_batch_size", type=int, default=None,
                        help="Batch size for priming evaluation. Defaults to --per_device_eval_batch_size.")
    parser.add_argument("--eval_max_samples", type=int, default=50000,
                        help="Max samples for standard evaluation (perplexity). <= 0 uses full dataset.")
    parser.add_argument("--priming_eval_max_samples_per_file", type=int, default=1000,
                        help="Max samples from each priming CSV. <= 0 uses all.")
    parser.add_argument("--priming_delimiter", type=str, default=".", help="Delimiter used in priming CSV files.")
    parser.add_argument("--model_class_name", type=str, default="GPT2LMHeadModel",
                        help="Name of the Hugging Face model class (e.g., GPT2LMHeadModel).")
    parser.add_argument("--base_model_name", type=str, default="gpt2",
                        help="Base model identifier for fallback tokenizer/config loading if not in checkpoint.")
    parser.add_argument("--use_amp", action="store_true",
                        help="Enable Automatic Mixed Precision (AMP) for evaluation (if model trained with it).")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (e.g., dataset sampling, priming randomization).")  # Updated help text
    parser.add_argument("--neptune_project", type=str, default=None,
                        help="Neptune project name (e.g., 'your-workspace/your-project').")
    parser.add_argument("--neptune_run_id", type=str, default=None,
                        help="Existing Neptune run ID to log to (rarely used by monitor for new evals).")
    parser.add_argument("--neptune_api_token", type=str, default=None,
                        help="Neptune API token (alternative to NEPTUNE_API_TOKEN env var).")
    parser.add_argument("--neptune_tags", type=str, nargs='+', default=None,
                        help="Optional Neptune tags for newly created evaluation runs.")
    parser.add_argument("--neptune_run_name", type=str, default=None,
                        help="Specific name for the Neptune run (monitor usually provides this).")
    parser.add_argument("--checkpoint_label", type=str, default=None,
                        help="Descriptive label for the checkpoint (e.g., 'checkpoint-1000', 'final_model'). Used for logging and file naming.")

    args = parser.parse_args()

    if args.priming_per_device_eval_batch_size is None:
        args.priming_per_device_eval_batch_size = args.per_device_eval_batch_size

    # --- Argument Validation ---
    if not Path(args.checkpoint_path).is_dir(): parser.error(f"Checkpoint directory not found: {args.checkpoint_path}")
    if args.run_standard_eval and not args.validation_dataset_path: parser.error(
        "--validation_dataset_path is required when --run_standard_eval is set.")
    if args.run_priming_eval and not args.priming_eval_dir_path: parser.error(
        "--priming_eval_dir_path is required when --run_priming_eval is set.")
    if args.validation_dataset_path and not Path(args.validation_dataset_path).is_dir(): parser.error(
        f"Validation dataset directory not found: {args.validation_dataset_path}")
    if args.priming_eval_dir_path and not Path(args.priming_eval_dir_path).is_dir(): parser.error(
        f"Priming evaluation directory not found: {args.priming_eval_dir_path}")

    # Ensure output directory exists (create if not)
    try:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        parser.error(f"Failed to create output directory {args.output_dir}: {e}")

    # Determine checkpoint_step (numeric) and checkpoint_step_label (string)
    if args.checkpoint_label:  # Prioritize explicitly passed label
        args.checkpoint_step_label = args.checkpoint_label
        try:  # Attempt to get numeric step if label is like checkpoint-XXXX
            args.checkpoint_numeric_step = int(args.checkpoint_label.split('-')[-1])
        except ValueError:  # Label is not like checkpoint-XXXX (e.g., "final_model")
            args.checkpoint_numeric_step = -1  # Indicates non-numeric label, or last step if known
            # If it's 'final_model', one might try to get the actual last step from training_args.json if available
            # For now, -1 is a placeholder for non-numeric steps in time-series.
    else:  # Fallback to deriving from checkpoint_path
        args.checkpoint_step_label = Path(args.checkpoint_path).name
        try:
            args.checkpoint_numeric_step = int(args.checkpoint_step_label.split('-')[-1])
        except (ValueError, IndexError):
            args.checkpoint_numeric_step = -1  # Fallback if path name doesn't contain step

    # If checkpoint_numeric_step is still -1 and label is 'final_model', it's truly the final model.
    # The `evaluation_monitor` could pass the actual last numeric step if it knows it for 'final_model'.
    # Otherwise, logging to Neptune with step -1 might place it at the beginning of time-series charts.

    return args


# --- Main Evaluation Execution ---
def main():
    global logger, tqdm  # Make tqdm available after import
    try:
        from tqdm.auto import tqdm as imported_tqdm
        tqdm = imported_tqdm
    except ImportError:
        print("Warning: tqdm.auto not installed. Progress bars will be basic or disabled.", file=sys.stderr)

        def fallback_tqdm(iterable, *args, **kwargs):  # Fallback tqdm
            disable = kwargs.pop('disable', False)
            if not disable and sys.stdout.isatty(): print(f"Processing {kwargs.get('desc', 'items')}...")
            return iterable

        tqdm = fallback_tqdm

    eval_args = parse_eval_args()

    # Setup logging to a file specific to this checkpoint's evaluation
    log_file_name = f"evaluate_log_{eval_args.checkpoint_step_label}.txt"
    log_file_path = Path(eval_args.output_dir) / log_file_name
    setup_logging(log_file=str(log_file_path))  # global logger is initialized/updated here
    logger = logging.getLogger(__name__)  # Ensure global logger var refers to this module's logger

    logger.info(
        f"***** Starting Evaluation Script for Checkpoint: '{eval_args.checkpoint_step_label}' (Numeric Step: {eval_args.checkpoint_numeric_step}) *****")
    logger.info(f"Full Evaluation Arguments: {vars(eval_args)}")

    device = get_device()  # Get device after logging is setup
    set_seed(eval_args.seed)  # Set seed for all random operations, including priming if used

    # --- Neptune Setup ---
    neptune_run_obj = None  # Initialize Neptune run object
    if NEPTUNE_AVAILABLE and eval_args.neptune_project:
        logger.info("Attempting to initialize Neptune for evaluation logging...")
        neptune_api_token_to_use = eval_args.neptune_api_token or os.getenv('NEPTUNE_API_TOKEN')
        if not neptune_api_token_to_use:
            logger.warning(
                "Neptune project specified, but no API token found (checked --neptune_api_token arg and NEPTUNE_API_TOKEN env var). Neptune logging will be disabled.")
        else:
            try:
                neptune_connection_mode = "async"  # Recommended for long-running scripts

                # Attempt to get linked training run name from environment (set by evaluation_monitor.sbatch)
                env_linked_training_run_name = os.getenv('SINGULARITYENV_NEPTUNE_TRAINING_RUN_NAME') or os.getenv(
                    'NEPTUNE_TRAINING_RUN_NAME')

                # Prepare tags for the Neptune run
                current_neptune_tags = ['evaluation', eval_args.checkpoint_step_label]  # Base tags
                if eval_args.neptune_tags:  # Add tags passed via arguments
                    current_neptune_tags.extend(eval_args.neptune_tags)
                if env_linked_training_run_name and f"train_ref:{env_linked_training_run_name}" not in current_neptune_tags:
                    current_neptune_tags.append(f"train_ref:{env_linked_training_run_name}")
                current_neptune_tags = sorted(list(set(current_neptune_tags)))  # Deduplicate and sort

                if eval_args.neptune_run_id:  # If an existing run ID is provided (less common for monitor)
                    logger.info(f"Attempting to connect to existing Neptune run ID: {eval_args.neptune_run_id}")
                    neptune_run_obj = neptune.init_run(project=eval_args.neptune_project,
                                                       api_token=neptune_api_token_to_use,
                                                       with_id=eval_args.neptune_run_id, mode=neptune_connection_mode)
                    # Log that this evaluation is part of this existing run
                    neptune_run_obj[f"evaluation_runs/{eval_args.checkpoint_step_label}/args"] = vars(eval_args)
                else:  # Create a new Neptune run for this specific evaluation
                    # Construct a descriptive run name, prioritizing argument if provided by monitor
                    run_name_for_neptune = eval_args.neptune_run_name  # From monitor: "eval_${SHARED_RUN_ID}_${checkpoint_name}"
                    if not run_name_for_neptune:  # Fallback name if monitor didn't provide one
                        shared_run_id_env = os.getenv('SHARED_RUN_ID', 'unknown_run')  # From orchestrator (via monitor)
                        run_name_for_neptune = f"Eval_{shared_run_id_env}_{eval_args.checkpoint_step_label}".strip('_')

                    logger.info(
                        f"Creating NEW Neptune run with Name: '{run_name_for_neptune}', Tags: {current_neptune_tags}")
                    neptune_run_obj = neptune.init_run(project=eval_args.neptune_project,
                                                       api_token=neptune_api_token_to_use,
                                                       name=run_name_for_neptune, tags=current_neptune_tags,
                                                       mode=neptune_connection_mode)
                    neptune_run_obj["evaluation/script_args"] = vars(eval_args)  # Log args for this new run
                    if env_linked_training_run_name:  # Store the reference to the training run
                        neptune_run_obj["details/linked_training_run_name"] = env_linked_training_run_name

                if neptune_run_obj: logger.info(
                    f"Neptune initialized successfully. Run URL: {neptune_run_obj.get_url()}")

            except Exception as e_neptune_init:
                logger.error(
                    f"Neptune initialization failed: {e_neptune_init}. Neptune logging disabled for this evaluation.",
                    exc_info=True)
                neptune_run_obj = None  # Ensure it's None if init fails
    else:
        logger.info(
            "Neptune logging disabled (either Neptune client not available or --neptune_project not specified).")

    # --- Load Model and Tokenizer ---
    logger.info(f"--- Loading Model and Tokenizer from Checkpoint: {eval_args.checkpoint_path} ---")
    try:
        # Simple map for model class names, extend if more models are used
        model_class_name_map = {"GPT2LMHeadModel": GPT2LMHeadModel}
        model_constructor = model_class_name_map.get(eval_args.model_class_name,
                                                     GPT2LMHeadModel)  # Default to GPT2LMHeadModel

        model, tokenizer, config = load_model_for_evaluation(
            model_constructor, eval_args.checkpoint_path, eval_args.base_model_name
        )
        model.to(device)
        model.eval()  # Ensure model is explicitly in evaluation mode
        logger.info(
            f"Model '{eval_args.model_class_name}' loaded successfully to device '{device}' and set to eval mode.")
    except Exception as e_model_load:
        logger.critical(
            f"FATAL ERROR: Failed to load model from checkpoint '{eval_args.checkpoint_path}': {e_model_load}",
            exc_info=True)
        if NEPTUNE_AVAILABLE and neptune_run_obj:  # Log critical error to Neptune if possible
            try:
                neptune_run_obj[
                    f"evaluation_status/{eval_args.checkpoint_step_label}/critical_error"] = f"Model load failed: {traceback.format_exc()}"
                neptune_run_obj.stop()  # Attempt to stop the run
            except Exception:
                pass  # Best effort
        sys.exit(1)  # Exit if model cannot be loaded

    # --- Prepare Standard Evaluation Dataloader (if requested) ---
    std_eval_dataloader = None
    if eval_args.run_standard_eval:
        logger.info("--- Preparing Standard Evaluation (Perplexity) Dataloader ---")
        if not eval_args.validation_dataset_path:
            logger.error(
                "Standard perplexity evaluation requested, but --validation_dataset_path was not provided. Skipping.")
        else:
            try:
                logger.info(f"Loading validation dataset for perplexity from: {eval_args.validation_dataset_path}")
                validation_hf_dataset = load_from_disk(eval_args.validation_dataset_path)
                full_dataset_size = len(validation_hf_dataset)
                logger.info(f"Full validation dataset size: {full_dataset_size:,} sequences.")

                # Apply sampling if eval_max_samples is set
                if eval_args.eval_max_samples is not None and 0 < eval_args.eval_max_samples < full_dataset_size:
                    np_rng = np.random.RandomState(eval_args.seed)  # Use a seeded RNG for consistent sampling
                    sampled_indices = np_rng.choice(full_dataset_size, size=eval_args.eval_max_samples, replace=False)
                    validation_hf_dataset = validation_hf_dataset.select(sampled_indices)
                    logger.info(
                        f"Using a subset for standard perplexity evaluation: {len(validation_hf_dataset):,} samples (seed: {eval_args.seed}).")
                elif eval_args.eval_max_samples is not None and eval_args.eval_max_samples > 0:
                    logger.info(
                        f"--eval_max_samples ({eval_args.eval_max_samples}) is >= dataset size or invalid. Using full validation set ({full_dataset_size:,} samples).")
                else:  # eval_max_samples <= 0 or None
                    logger.info(
                        f"Using full validation set for standard perplexity evaluation ({full_dataset_size:,} samples).")

                sequential_sampler = SequentialSampler(validation_hf_dataset)
                # Ensure tokenizer has pad_token for collator (should be handled by load_model_for_evaluation)
                lm_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
                std_eval_dataloader = DataLoader(
                    validation_hf_dataset, sampler=sequential_sampler, batch_size=eval_args.per_device_eval_batch_size,
                    num_workers=eval_args.num_workers, pin_memory=True, collate_fn=lm_data_collator
                )
                logger.info("Standard Perplexity Evaluation DataLoader prepared successfully.")
            except Exception as e_std_data:
                logger.error(f"Failed to load or prepare standard evaluation (perplexity) data: {e_std_data}",
                             exc_info=True)
                std_eval_dataloader = None  # Ensure it's None on error
    else:
        logger.info("Standard perplexity evaluation skipped as per arguments (--run_standard_eval not set).")

    # --- Run Evaluations ---
    # Store all evaluation results in a dictionary
    all_evaluation_results = {
        "checkpoint_label": eval_args.checkpoint_step_label,
        "checkpoint_numeric_step": eval_args.checkpoint_numeric_step  # Numeric step for time-series
    }
    overall_eval_start_time = time.time()

    # Standard (Perplexity) Evaluation
    if eval_args.run_standard_eval:
        if std_eval_dataloader:
            standard_eval_metrics = evaluate_standard(eval_args, model, std_eval_dataloader, device)
            all_evaluation_results["standard_perplexity_summary"] = standard_eval_metrics
            # Log to Neptune if run object exists and metrics were generated
            if NEPTUNE_AVAILABLE and neptune_run_obj and standard_eval_metrics:
                try:
                    loss_val = standard_eval_metrics.get("loss", float('nan'))
                    ppl_val = standard_eval_metrics.get("perplexity", float('nan'))
                    # Log with the numeric checkpoint_numeric_step for time-series plotting
                    if math.isfinite(loss_val): neptune_run_obj[f"eval_metrics/standard_loss"].append(loss_val,
                                                                                                      step=eval_args.checkpoint_numeric_step)
                    if math.isfinite(ppl_val): neptune_run_obj[f"eval_metrics/perplexity"].append(ppl_val,
                                                                                                  step=eval_args.checkpoint_numeric_step)
                    logger.info(
                        f"Logged standard perplexity evaluation metrics to Neptune (for numeric step {eval_args.checkpoint_numeric_step}).")
                except Exception as e_neptune_std_log:
                    logger.warning(
                        f"Neptune logging for standard perplexity eval metrics failed (numeric step {eval_args.checkpoint_numeric_step}): {e_neptune_std_log}")
        else:  # Dataloader failed to initialize
            logger.warning(
                "Standard perplexity evaluation was requested but dataloader failed to initialize. Skipping this evaluation.")
            all_evaluation_results["standard_perplexity_summary"] = {"error": "Dataloader failed to initialize."}

    # Priming Evaluation
    if eval_args.run_priming_eval:
        priming_evaluation_summary_metrics = run_priming_evaluation_on_directory(
            eval_args, model, tokenizer, device, neptune_run_obj, eval_args.checkpoint_numeric_step
        )
        all_evaluation_results["priming_evaluation_summary"] = priming_evaluation_summary_metrics
        # Note: Neptune logging for priming metrics is handled *inside* run_priming_evaluation_on_directory

    # --- Finalize and Save Summary ---
    total_script_duration_seconds = time.time() - overall_eval_start_time
    logger.info(
        f"Total evaluation script execution time: {total_script_duration_seconds:.2f} seconds for checkpoint '{eval_args.checkpoint_step_label}'.")
    all_evaluation_results["total_evaluation_duration_seconds"] = total_script_duration_seconds
    if NEPTUNE_AVAILABLE and neptune_run_obj:
        try:  # Log duration to a path that includes the label, not as a time-series
            neptune_run_obj[
                f"evaluation_details/{eval_args.checkpoint_step_label}/duration_seconds"] = total_script_duration_seconds
        except Exception:
            pass  # Best effort

    # Save the comprehensive summary of all evaluations run for this checkpoint
    json_summary_filename = f"evaluation_summary_{eval_args.checkpoint_step_label}.json"
    json_summary_filepath = Path(eval_args.output_dir) / json_summary_filename
    logger.info(f"--- Saving Comprehensive Evaluation Summary for Checkpoint '{eval_args.checkpoint_step_label}' ---")

    # Check if any actual results were generated beyond the initial labels
    if len(all_evaluation_results) > 2:  # i.e., more than just checkpoint_label and checkpoint_numeric_step
        logger.info(
            f"Final Summary Metrics for '{eval_args.checkpoint_step_label}': {json.dumps(all_evaluation_results, indent=2)}")
        try:
            with open(json_summary_filepath, "w", encoding='utf-8') as f_summary:
                json.dump(all_evaluation_results, f_summary, indent=4)
            logger.info(f"Comprehensive evaluation summary saved to: {json_summary_filepath}")
            # Optionally log the entire summary dictionary to Neptune under the specific checkpoint label
            if NEPTUNE_AVAILABLE and neptune_run_obj:
                try:
                    neptune_run_obj[
                        f"evaluation_details/{eval_args.checkpoint_step_label}/full_summary_dict"] = all_evaluation_results
                except Exception as ne_log_summary_dict:
                    logger.warning(
                        f"Failed to log final comprehensive evaluation summary dictionary to Neptune: {ne_log_summary_dict}")
        except Exception as e_save_json:
            logger.error(f"Failed to save or serialize comprehensive evaluation summary JSON: {e_save_json}",
                         exc_info=True)
    else:
        logger.warning(
            f"No actual evaluation results (beyond labels) were generated for checkpoint '{eval_args.checkpoint_step_label}'. Summary JSON will be minimal or not saved if empty.")

    # Reminder about raw priming CSV location
    if eval_args.run_priming_eval:
        priming_csv_path_final = Path(
            eval_args.output_dir) / f"priming_results_raw_step_{eval_args.checkpoint_numeric_step}.csv"
        logger.info(f"Raw priming results (if generated) were saved to: {priming_csv_path_final}")

    # --- Cleanup Neptune Run ---
    if NEPTUNE_AVAILABLE and neptune_run_obj:
        try:
            neptune_run_obj.sync()  # Ensure all data is sent
            neptune_run_obj.stop()
            logger.info("Neptune run stopped successfully.")
        except Exception as e_neptune_stop:
            logger.error(f"Neptune stop operation failed: {e_neptune_stop}")

    logger.info(f"***** Evaluation Script Finished for Checkpoint: '{eval_args.checkpoint_step_label}' *****")


if __name__ == "__main__":
    # This basicConfig is a fallback. setup_logging in main will override for the specific run.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()