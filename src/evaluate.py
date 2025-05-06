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
from typing import Any, Dict, List, Optional, Tuple # Added Tuple

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
from torch.cuda.amp import autocast # For evaluate_standard AMP context

# Priming evaluation imports (assuming they are in the python path)
try:
    from priming_evaluation.data_loader import create_priming_dataloader
    # Assuming the modified version returning (summary, raw)
    from priming_evaluation.evaluator import run_native_priming_eval
except ImportError as e:
    print(f"Error: Failed to import priming_evaluation modules: {e}", file=sys.stderr)
    print("Ensure priming_evaluation package is installed and accessible.", file=sys.stderr)
    # We don't exit here, but priming eval will fail later if requested
    create_priming_dataloader = None
    run_native_priming_eval = None

# Optional Neptune import
try:
    import neptune
    NEPTUNE_AVAILABLE = True
except ImportError:
    neptune = None # Define as None if not available
    NEPTUNE_AVAILABLE = False
    print("Neptune.ai library not found, Neptune logging will be disabled for evaluation.")

# --- Globals ---
logger = None

# --- Helper Functions (Adapted from train.py) ---

def get_device():
    """Gets the appropriate device for PyTorch computations."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    elif torch.cuda.is_available():
        # For evaluation, usually run on a single GPU, default to cuda:0
        # Adjust if distributed evaluation is intended
        device = torch.device("cuda:0")
        print(f"Using CUDA GPU: 0 - {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def setup_logging(log_level=logging.INFO, log_file=None):
    """Configures basic logging, optionally to a file."""
    global logger
    fmt = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s"
    dfmt = "%Y-%m-%d %H:%M:%S"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        try:
            # Ensure directory exists
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file, mode='a')) # Append mode
        except OSError as e:
            print(f"Warning: Could not create log file handler for {log_file}: {e}")

    logging.basicConfig(level=log_level, format=fmt, datefmt=dfmt, handlers=handlers, force=True)
    logger = logging.getLogger(__name__) # Get the logger
    logger.info("Logging setup complete for evaluation script.")

def set_seed(seed_value):
    """Sets random seeds."""
    import random
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    if logger: logger.info(f"Set random seed: {seed_value}")
    else: print(f"Set random seed: {seed_value}")


def load_model_for_evaluation(model_class, checkpoint_path: str, base_model_name="gpt2"):
    """Loads model, tokenizer, and config from checkpoint for evaluation."""
    # This function can be used directly from train.py if it's placed in a shared utility module
    # For now, copied here for self-containment.
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_dir():
        logger.error(f"Checkpoint directory not found: {ckpt_path}")
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    logger.info(f"Loading tokenizer and config from checkpoint: {ckpt_path}")
    try:
        # Try loading tokenizer and config directly from the checkpoint directory
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=True)
        config = AutoConfig.from_pretrained(ckpt_path)
        logger.info("Loaded tokenizer and config from checkpoint directory.")
    except OSError:
        # Fallback to loading from the base model name if not found in checkpoint
        logger.warning(f"Tokenizer/config not found in checkpoint {ckpt_path}. "
                       f"Falling back to loading from base model: {base_model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
            config = AutoConfig.from_pretrained(base_model_name)
        except Exception as e:
            logger.error(f"Fallback loading of tokenizer/config failed for {base_model_name}: {e}", exc_info=True)
            raise

    logger.info(f"Loading model weights from checkpoint: {ckpt_path}")
    try:
        # Load model weights from the checkpoint directory using the potentially modified config
        model = model_class.from_pretrained(ckpt_path, config=config)
        logger.info("Successfully loaded model weights.")
    except Exception as e:
        logger.error(f"Failed to load model weights from checkpoint {ckpt_path}: {e}", exc_info=True)
        raise

    # Ensure tokenizer has a pad token, crucial for collation
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set tokenizer pad_token to eos_token ('{tokenizer.eos_token}')")
        else:
            # Avoid adding tokens during evaluation if possible, try finding one
            # This part might need refinement based on the tokenizer.
            # If absolutely necessary, add one, but it might affect perplexity slightly
            # if the model wasn't trained with it (though unlikely if base model had one).
            logger.warning("Tokenizer has no pad_token and no eos_token. Using pad_token_id 0 if possible.")
            if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is None:
                 tokenizer.pad_token_id = 0 # A common default, but verify

    return model, tokenizer, config


# --- Evaluation Functions (Adapted from train.py) ---

def evaluate_standard(args, model, eval_dataloader, device):
    """Runs standard evaluation (perplexity). Assumes single-node/GPU for now."""
    # Simplified version assuming single GPU evaluation (no DDP aggregation needed here)
    # If distributed eval is needed, the DDP aggregation logic from train.py is required.

    if eval_dataloader is None:
        logger.warning("Standard evaluation dataloader is None. Skipping standard eval.")
        return {}

    original_mode = model.training
    model.eval()

    total_loss = 0.0
    total_items = 0

    logger.info("Starting standard evaluation...")
    progress_bar = tqdm(eval_dataloader, desc="Eval (Std)", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            try:
                batch_on_device = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            except RuntimeError as e:
                logger.error(f"Error moving standard eval batch to device {device}: {e}")
                continue

            try:
                # Use AMP context based on args and device type
                amp_enabled = args.use_amp and device.type == 'cuda'
                with autocast(enabled=amp_enabled):
                    outputs = model(**batch_on_device)
                    loss = outputs.loss
            except Exception as e:
                logger.error(f"Error during standard eval forward pass: {e}", exc_info=True)
                continue

            if loss is not None and torch.isfinite(loss):
                # outputs.loss is already averaged over batch items by HF LMHeadModel
                # Accumulate sum of losses weighted by batch size for correct overall average
                num_items_in_batch = batch_on_device['input_ids'].size(0)
                total_loss += loss.detach().item() * num_items_in_batch
                total_items += num_items_in_batch
                progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
            else:
                logger.warning(f"Non-finite loss detected: {loss.item() if loss is not None else 'None'}. Skipping batch contribution.")

    # Calculate final metrics
    metrics = {}
    if total_items > 0:
        final_avg_loss = total_loss / total_items
        try:
            perplexity = math.exp(final_avg_loss) if final_avg_loss < 700 else float('inf')
        except (OverflowError, ValueError):
            perplexity = float('inf') if final_avg_loss > 0 else float('nan')

        logger.info(f"Standard Evaluation Results: Average Loss = {final_avg_loss:.4f}, Perplexity = {perplexity:.4f}, Total Items = {total_items}")
        metrics = {"loss": final_avg_loss, "perplexity": perplexity, "total_items": total_items}
    else:
        logger.warning("Standard evaluation completed, but total_items processed is zero. No metrics calculated.")
        metrics = {"loss": float('nan'), "perplexity": float('nan'), "total_items": 0}

    if original_mode: model.train() # Restore mode
    progress_bar.close()
    return metrics


def run_priming_evaluation_on_directory(
    eval_args, # Pass evaluation-specific args
    model,
    tokenizer,
    device,
    run, # Neptune run object (can be None)
    checkpoint_step: int # Pass the step number explicitly
):
    """
    Finds CSVs, creates dataloaders, runs eval, aggregates summary results,
    appends raw per-item results to a persistent CSV file in the specified output directory,
    and logs aggregated summary metrics to Neptune *once* per call.
    Returns a dictionary of summary metrics per file.
    MODIFIED for evaluate.py context.
    """

    # Ensure priming libraries were imported
    if create_priming_dataloader is None or run_native_priming_eval is None:
         logger.error("Priming evaluation libraries not available. Skipping priming eval.")
         return {"error": "Priming library import failed."}

    if not eval_args.run_priming_eval or not eval_args.priming_eval_dir_path:
        logger.info("Skipping priming evaluation (not enabled or path not provided).")
        return {}

    priming_dir = Path(eval_args.priming_eval_dir_path)
    if not priming_dir.is_dir():
        logger.error(f"Priming evaluation directory not found: {priming_dir}")
        return {"error": f"Priming directory not found: {priming_dir}"}

    # Use the specific output directory provided for this evaluation run
    eval_output_dir = Path(eval_args.output_dir)
    try:
        eval_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create evaluation output directory {eval_output_dir}: {e}. Cannot save priming CSV.")
        return {"error": f"Failed to create output directory {eval_output_dir}"}

    csv_files = sorted(list(priming_dir.glob('*.csv')))
    if not csv_files:
        logger.warning(f"No *.csv files found in priming directory: {priming_dir}")
        return {}

    # --- CSV Setup ---
    # Save CSV within the evaluation-specific output directory
    csv_output_path = eval_output_dir / "priming_results_raw.csv"
    logger.info(f"Raw priming results CSV will be appended to: {csv_output_path}")

    all_priming_summary_results = {} # Store the aggregated metrics per file for return value
    neptune_logs_for_this_step = {} # Dict to store logs for this step

    logger.info(f"Found {len(csv_files)} CSVs for priming eval in {priming_dir}.")
    original_mode = model.training

    # --- CSV File Handling ---
    csv_writer = None
    csv_file_handle = None
    try:
        # Use 'a' mode to append if multiple eval scripts run targeting the same output dir (less likely here)
        # Or 'w' mode to overwrite/create fresh for this specific eval run
        csv_file_handle = open(csv_output_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file_handle)
        header = ["eval_step", "corpus_file", "target_structure", "item_index", "pe", "logp_con", "logp_incon"]
        csv_writer.writerow(header)
        logger.info(f"Opened and wrote header to priming results CSV: {csv_output_path}")
    except IOError as e:
        logger.error(f"Failed to open or write header to CSV {csv_output_path}: {e}")
        csv_writer = None
        if csv_file_handle:
            try: csv_file_handle.close()
            except Exception: pass
        csv_file_handle = None

    # --- Process each CSV file ---
    model.eval() # Ensure model is in eval mode
    for csv_path in csv_files:
        csv_filename = csv_path.name
        logger.info(f"--- Running Priming Eval for: {csv_filename} (Checkpoint Step {checkpoint_step}) ---")

        priming_dataloader_single = None
        try:
            priming_dataloader_single = create_priming_dataloader(
                csv_path=str(csv_path),
                tokenizer=tokenizer,
                batch_size=eval_args.priming_per_device_eval_batch_size,
                delimiter=eval_args.priming_delimiter,
                num_workers=eval_args.num_workers,
                pin_memory=True,
                max_samples=eval_args.priming_eval_max_samples_per_file,
                seed=eval_args.seed
            )
        except Exception as e:
            logger.error(f"Dataloader creation failed for {csv_filename}: {e}", exc_info=True)
            all_priming_summary_results[csv_filename] = {"error": f"Dataloader creation failed: {e}"}
            continue

        if priming_dataloader_single is None or len(priming_dataloader_single.dataset) == 0:
             logger.warning(f"Dataloader for {csv_filename} is None or empty. Skipping.")
             all_priming_summary_results[csv_filename] = {"error": "Dataloader None or empty."}
             continue

        # --- Run the evaluation ---
        try:
            priming_summary_metrics, priming_raw_results = run_native_priming_eval(
                model=model,
                priming_dataloader=priming_dataloader_single,
                device=device,
                tokenizer=tokenizer,
                use_amp=eval_args.use_amp # Use AMP flag from eval args
            )

            # Store summary results for return value
            all_priming_summary_results[csv_filename] = priming_summary_metrics
            logger.info(f"Priming Summary Metrics for {csv_filename}: {priming_summary_metrics}")

            # --- Aggregate metrics for Neptune logging ---
            if run: # Check if Neptune run object exists
                 log_prefix = f"eval/priming/{csv_filename.replace('.', '_').replace('/','_')}" # Sanitize name
                 metrics_to_log = {k: v for k, v in priming_summary_metrics.items() if isinstance(v, (int, float)) and math.isfinite(v)}
                 if metrics_to_log:
                     for k, v in metrics_to_log.items():
                         neptune_logs_for_this_step[f"{log_prefix}/{k}"] = v

            # --- Write Raw Results to CSV ---
            if csv_writer and priming_raw_results:
                logger.info(f"Writing {sum(len(v) for v in priming_raw_results.values()):,} raw priming results to CSV for {csv_filename}...")
                items_written_count = 0
                try:
                    for target_structure, results_list in priming_raw_results.items():
                        for idx, item_data in enumerate(results_list):
                            if isinstance(item_data, dict):
                                pe_val = item_data.get('pe', float('nan'))
                                logp_con_val = item_data.get('logp_con', float('nan'))
                                logp_incon_val = item_data.get('logp_incon', float('nan'))
                                row = [
                                    checkpoint_step, csv_filename, target_structure, idx,
                                    f"{pe_val:.6f}" if not math.isnan(pe_val) else 'NaN',
                                    f"{logp_con_val:.6f}" if not math.isnan(logp_con_val) else 'NaN',
                                    f"{logp_incon_val:.6f}" if not math.isnan(logp_incon_val) else 'NaN'
                                ]
                                csv_writer.writerow(row)
                                items_written_count += 1
                            else:
                                logger.warning(f"Skipping invalid item data format in raw results for {target_structure} index {idx} in {csv_filename}: Type={type(item_data)}")
                    if csv_file_handle: csv_file_handle.flush() # Flush after each file
                    logger.info(f"Finished writing {items_written_count} rows to CSV for {csv_filename}.")
                except Exception as e:
                     logger.error(f"Error occurred while writing raw results to CSV for {csv_filename}: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Priming evaluation run failed for {csv_filename}: {e}", exc_info=True)
            all_priming_summary_results[csv_filename] = {"error": f"Evaluation run failed: {e}"}
        finally:
            del priming_dataloader_single
            gc.collect()

    # --- Log aggregated metrics AFTER the loop ---
    if run and neptune_logs_for_this_step:
        logger.info(f"Logging {len(neptune_logs_for_this_step)} aggregated priming summary metrics to Neptune for step {checkpoint_step}...")
        try:
            # Use append API for logging time-series data if appropriate, or direct assignment
            # Using append assumes you might run eval for the same checkpoint multiple times
            # For a single eval run, direct assignment might be okay, but append is safer
            for metric_path, value in neptune_logs_for_this_step.items():
                 # Use the explicit checkpoint_step provided to the function
                 run[metric_path].append(value, step=checkpoint_step)
            logger.info(f"Finished logging aggregated metrics to Neptune for step {checkpoint_step}.")
        except Exception as e:
             logger.warning(f"Neptune logging failed for aggregated priming summary metrics at step {checkpoint_step}: {e}")
    elif run:
        logger.info(f"No priming summary metrics were aggregated for Neptune logging at step {checkpoint_step}.")

    # --- Close CSV File ---
    if csv_file_handle:
        try:
            csv_file_handle.close()
            logger.info("Closed priming results CSV file.")
        except Exception as e:
            logger.error(f"Error closing CSV file {csv_output_path}: {e}")

    if original_mode: model.train() # Restore mode
    logger.info(f"--- Finished All Priming Evaluations for Checkpoint Step {checkpoint_step} ---")

    return all_priming_summary_results


# --- Argument Parser for evaluate.py ---

def parse_eval_args():
    """Parses command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate a trained GPT-2 like model checkpoint.")

    # === Required Paths ===
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the specific checkpoint directory to evaluate.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation results (JSON summary, priming CSV).")

    # === Evaluation Control ===
    parser.add_argument("--run_standard_eval", action="store_true", default=False, help="Run standard perplexity evaluation.")
    parser.add_argument("--run_priming_eval", action="store_true", default=False, help="Run priming evaluation from directory.")

    # === Dataset Paths (Required if corresponding eval is run) ===
    parser.add_argument("--validation_dataset_path", type=str, default=None, help="Path to the validation Arrow dataset (needed for --run_standard_eval).")
    parser.add_argument("--priming_eval_dir_path", type=str, default=None, help="Directory containing priming CSVs (needed for --run_priming_eval).")

    # === Evaluation Parameters ===
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Standard eval batch size per device.")
    parser.add_argument("--priming_per_device_eval_batch_size", type=int, default=None, help="Priming eval batch size. Defaults to --per_device_eval_batch_size.")
    parser.add_argument("--eval_max_samples", type=int, default=50000, help="Maximum number of samples for standard evaluation. Default: 50,000. <= 0 uses full dataset.")
    parser.add_argument("--priming_eval_max_samples_per_file", type=int, default=1000, help="Maximum number of samples from each priming CSV. <= 0 uses all.")
    parser.add_argument("--priming_delimiter", type=str, default=".", help="Delimiter in priming CSVs.")

    # === Model & Hardware ===
    parser.add_argument("--model_class_name", type=str, default="GPT2LMHeadModel", help="Name of the Hugging Face model class (e.g., GPT2LMHeadModel).")
    parser.add_argument("--base_model_name", type=str, default="gpt2", help="Base model identifier for fallback tokenizer/config loading.")
    parser.add_argument("--use_amp", action="store_true", help="Enable AMP for evaluation (useful if model trained with AMP).")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (sampling).")

    # === Neptune Logging (Optional) ===
    parser.add_argument("--neptune_project", type=str, default=None, help="Neptune project name (e.g., 'user/project'). If set, attempts to log results.")
    parser.add_argument("--neptune_run_id", type=str, default=None, help="Existing Neptune run ID to log to. If None and project is set, creates a new run.")
    parser.add_argument("--neptune_api_token", type=str, default=None, help="Neptune API token (if not set via environment variable NEPTUNE_API_TOKEN).")
    parser.add_argument("--neptune_tags", type=str, nargs='+', default=None, help="Optional Neptune tags for new runs (e.g., 'evaluation', 'checkpoint-1000').")


    args = parser.parse_args()

    # Set defaults based on other args
    if args.priming_per_device_eval_batch_size is None:
        args.priming_per_device_eval_batch_size = args.per_device_eval_batch_size

    # --- Validation ---
    if not Path(args.checkpoint_path).is_dir():
        parser.error(f"Checkpoint directory not found: {args.checkpoint_path}")
    if args.run_standard_eval and not args.validation_dataset_path:
        parser.error("--validation_dataset_path is required when --run_standard_eval is set.")
    if args.run_priming_eval and not args.priming_eval_dir_path:
        parser.error("--priming_eval_dir_path is required when --run_priming_eval is set.")
    if args.validation_dataset_path and not Path(args.validation_dataset_path).is_dir():
         parser.error(f"Validation dataset directory not found: {args.validation_dataset_path}")
    if args.priming_eval_dir_path and not Path(args.priming_eval_dir_path).is_dir():
         parser.error(f"Priming evaluation directory not found: {args.priming_eval_dir_path}")

    # Ensure output directory exists (create if not)
    try:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        parser.error(f"Failed to create output directory {args.output_dir}: {e}")

    # Extract checkpoint step from path for logging/reporting
    try:
        # Assumes format like ".../checkpoint-STEP"
        args.checkpoint_step = int(Path(args.checkpoint_path).name.split('-')[-1])
    except (ValueError, IndexError):
        print(f"Warning: Could not automatically determine checkpoint step from path: {args.checkpoint_path}. Using step -1 for reporting.")
        args.checkpoint_step = -1 # Use -1 to indicate unknown step

    return args

# --- Main Evaluation Execution ---

def main():
    eval_args = parse_eval_args()

    # Setup logging (log to a file in the output dir)
    log_file = Path(eval_args.output_dir) / f"evaluate_log_step_{eval_args.checkpoint_step}.txt"
    setup_logging(log_file=str(log_file))
    global logger # Ensure logger is assigned
    logger = logging.getLogger(__name__)

    logger.info(f"***** Starting Evaluation Script *****")
    logger.info(f"Evaluation Arguments: {vars(eval_args)}")

    # Setup Device and Seed
    device = get_device()
    set_seed(eval_args.seed)

    # --- Neptune Setup ---
    run = None
    if NEPTUNE_AVAILABLE and eval_args.neptune_project:
        logger.info("Attempting to initialize Neptune...")
        # Allow overriding token via arg, fallback to env var
        api_token = eval_args.neptune_api_token or os.getenv('NEPTUNE_API_TOKEN')
        if not api_token:
            logger.warning("Neptune project specified, but no API token found (checked arg --neptune_api_token and env NEPTUNE_API_TOKEN). Neptune disabled.")
        else:
            try:
                neptune_mode = "async" # Use async mode for potentially long evaluations
                if eval_args.neptune_run_id:
                    logger.info(f"Connecting to existing Neptune run: {eval_args.neptune_run_id}")
                    run = neptune.init_run(
                        project=eval_args.neptune_project,
                        api_token=api_token,
                        with_id=eval_args.neptune_run_id,
                        mode=neptune_mode
                    )
                    # Optionally log evaluation parameters to the existing run
                    run[f"evaluation/step_{eval_args.checkpoint_step}/args"] = vars(eval_args)
                else:
                    logger.info(f"Creating NEW Neptune run for evaluation of step {eval_args.checkpoint_step}")
                    # Add checkpoint step to default tags if creating new run
                    default_tags = ['evaluation', f'checkpoint-{eval_args.checkpoint_step}']
                    if eval_args.neptune_tags:
                        default_tags.extend(eval_args.neptune_tags)

                    run = neptune.init_run(
                        project=eval_args.neptune_project,
                        api_token=api_token,
                        name=f"Eval_Step_{eval_args.checkpoint_step}",
                        tags=default_tags,
                        mode=neptune_mode
                    )
                    # Log evaluation parameters for the new run
                    run["evaluation/args"] = vars(eval_args)
                logger.info(f"Neptune initialized. Run URL: {run.get_url()}")
            except Exception as e:
                logger.error(f"Neptune initialization failed: {e}. Neptune logging disabled.", exc_info=True)
                run = None
    else:
        logger.info("Neptune logging disabled.")

    # --- Load Model ---
    logger.info(f"--- Loading Model Checkpoint: {eval_args.checkpoint_path} ---")
    try:
        # Dynamically get model class if needed, but default to GPT2LMHeadModel
        model_class = GPT2LMHeadModel # Replace with dynamic import if needed based on config/args
        model, tokenizer, config = load_model_for_evaluation(
            model_class,
            eval_args.checkpoint_path,
            eval_args.base_model_name
        )
        model.to(device)
        model.eval() # Ensure model is in eval mode
        logger.info(f"Model loaded successfully to {device}.")
    except Exception as e:
        logger.critical(f"Fatal Error: Failed to load model checkpoint: {e}", exc_info=True)
        if run:
            try: run[f"evaluation/step_{eval_args.checkpoint_step}/error"] = f"Model load failed: {e}"; run.stop()
            except Exception: pass
        sys.exit(1)

    # --- Prepare Standard Eval Dataloader ---
    std_eval_dataloader = None
    if eval_args.run_standard_eval:
        logger.info("--- Preparing Standard Evaluation Dataloader ---")
        if not eval_args.validation_dataset_path:
            logger.error("Standard evaluation requested but no validation_dataset_path provided.")
        else:
            try:
                logger.info(f"Loading validation data: {eval_args.validation_dataset_path}")
                ds = load_from_disk(eval_args.validation_dataset_path)
                original_size = len(ds)
                logger.info(f"Full Eval dataset size: {original_size:,} sequences")

                # Sampling Logic (copied from train.py)
                if eval_args.eval_max_samples is not None and eval_args.eval_max_samples > 0 and eval_args.eval_max_samples < original_size:
                    logger.info(f"Sampling {eval_args.eval_max_samples:,} sequences (seed: {eval_args.seed}).")
                    rng = np.random.RandomState(eval_args.seed)
                    sampled_indices = rng.choice(original_size, size=eval_args.eval_max_samples, replace=False)
                    ds = ds.select(sampled_indices)
                    logger.info(f"Using subset for Eval: {len(ds):,} sequences")
                elif eval_args.eval_max_samples is not None and eval_args.eval_max_samples > 0:
                     logger.info(f"Eval_max_samples >= dataset size. Using full validation set.")
                else:
                     logger.info("Eval_max_samples <= 0. Using full validation set.")

                sampler = SequentialSampler(ds)
                data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
                std_eval_dataloader = DataLoader(
                    ds,
                    sampler=sampler,
                    batch_size=eval_args.per_device_eval_batch_size,
                    num_workers=eval_args.num_workers,
                    pin_memory=True,
                    collate_fn=data_collator
                )
                logger.info("Standard Eval DataLoader created.")
            except Exception as e:
                logger.error(f"Failed to load/prepare standard evaluation data: {e}", exc_info=True)
                std_eval_dataloader = None # Ensure it's None on error
    else:
        logger.info("Skipping standard evaluation data loading.")


    # --- Run Evaluations ---
    all_results = {}
    eval_start_time = time.time()

    # Standard Evaluation
    if eval_args.run_standard_eval:
        if std_eval_dataloader:
            std_metrics = evaluate_standard(eval_args, model, std_eval_dataloader, device)
            all_results["standard_summary"] = std_metrics
            if run and std_metrics:
                try:
                    loss_val = std_metrics.get("loss", float('nan')); ppl_val = std_metrics.get("perplexity", float('nan'))
                    step = eval_args.checkpoint_step
                    if math.isfinite(loss_val): run[f"evaluation/step_{step}/standard_loss"].append(loss_val, step=step)
                    if math.isfinite(ppl_val): run[f"evaluation/step_{step}/perplexity"].append(ppl_val, step=step)
                    logger.info(f"Logged standard eval metrics to Neptune for step {step}.")
                except Exception as e:
                    logger.warning(f"Neptune standard eval log failed for step {eval_args.checkpoint_step}: {e}")
        else:
            logger.warning("Standard evaluation requested but dataloader failed to initialize. Skipping.")
            all_results["standard_summary"] = {"error": "Dataloader failed"}


    # Priming Evaluation
    if eval_args.run_priming_eval:
        prime_metrics_summary = run_priming_evaluation_on_directory(
            eval_args=eval_args,
            model=model,
            tokenizer=tokenizer,
            device=device,
            run=run, # Pass Neptune run object
            checkpoint_step=eval_args.checkpoint_step # Pass step explicitly
        )
        all_results["priming_summary"] = prime_metrics_summary
        # Note: Neptune logging for priming is handled *inside* run_priming_evaluation_on_directory


    eval_duration = time.time() - eval_start_time
    logger.info(f"Total evaluation time: {eval_duration:.2f} seconds")
    if run:
         try: run[f"evaluation/step_{eval_args.checkpoint_step}/duration_seconds"] = eval_duration
         except Exception: pass


    # --- Save Final Summary Results ---
    summary_file_path = Path(eval_args.output_dir) / f"evaluation_summary_step_{eval_args.checkpoint_step}.json"
    logger.info(f"--- Saving Evaluation Summary ---")
    if all_results:
        logger.info(f"Final Summary Metrics: {all_results}")
        try:
            with open(summary_file_path, "w", encoding='utf-8') as f:
                json.dump(all_results, f, indent=4)
            logger.info(f"Evaluation summary results saved to: {summary_file_path}")
            # Log summary dict to Neptune
            if run:
                 try: run[f"evaluation/step_{eval_args.checkpoint_step}/summary_results"] = all_results
                 except Exception as ne: logger.warning(f"Failed to log final eval summary results dict to Neptune: {ne}")
        except IOError as e:
            logger.error(f"Failed to save evaluation summary JSON: {e}")
        except TypeError as e:
            logger.error(f"Failed to serialize evaluation summary to JSON: {e}")
    else:
        logger.warning("No evaluation results were generated to save.")

    # Remind user where raw priming CSV is (if run)
    if eval_args.run_priming_eval:
        csv_path = Path(eval_args.output_dir) / 'priming_results_raw.csv'
        logger.info(f"Raw priming results (if generated) were saved to: {csv_path}")


    # --- Cleanup ---
    if run:
        try:
            run.stop()
            logger.info("Neptune run stopped.")
        except Exception as e:
            logger.error(f"Neptune stop failed: {e}")

    logger.info(f"***** Evaluation Script Finished for Checkpoint Step {eval_args.checkpoint_step} *****")


if __name__ == "__main__":
    # Basic logging setup in case main fails early
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    # Import tqdm here for the progress bars used in eval functions
    try:
        from tqdm.auto import tqdm
    except ImportError:
        print("Warning: tqdm not installed. Progress bars will be disabled.")
        # Create a dummy tqdm if needed, or functions should handle disable=True
        def tqdm(iterable, *args, **kwargs): return iterable

    main()