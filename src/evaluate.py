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
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime  # For sentinel file content
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel priming loading

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
    BatchEncoding,
    PreTrainedModel  # Added for type hinting
)
from datasets import load_from_disk
from torch.cuda.amp import autocast  # For evaluate_standard AMP context

# Priming evaluation imports
try:
    # Assuming priming_evaluation is in the python path (e.g. src is in PYTHONPATH or installed)
    from priming_evaluation.data_loader import load_and_process_priming_data, create_priming_dataloader
    from priming_evaluation.evaluator import run_native_priming_eval

    PRIMING_LIBS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Failed to import priming_evaluation modules: {e}. Priming evaluation will be skipped.",
          file=sys.stderr)
    load_and_process_priming_data = None
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
tqdm = None  # Will be imported or polyfilled in main_script_entry


# --- Helper Functions (Existing ones like get_device, setup_logging, set_seed are kept) ---
def get_device():
    """Gets the appropriate device for PyTorch computations."""
    global logger
    dev_str = "cpu"
    dev_name = "CPU"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():  # Check if built is crucial
        dev_str = "mps"
        dev_name = "MPS"
    elif torch.cuda.is_available():
        dev_str = "cuda:0"  # Default to first GPU for evaluation
        try:
            dev_name = torch.cuda.get_device_name(0)
        except Exception:
            dev_name = "CUDA GPU"

    device = torch.device(dev_str)

    log_msg_device = f"Using device: {device} ({dev_name})"
    if logger:
        logger.info(log_msg_device)
    else:
        print(log_msg_device)
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
    log_msg_seed = f"Set random seed: {seed_value}"
    if logger:
        logger.info(log_msg_seed)
    else:
        print(log_msg_seed)


def load_base_model_and_tokenizer(eval_args, model_class):
    """Loads base model architecture (no weights) and tokenizer."""
    global logger
    logger.info(f"Loading base tokenizer from: {eval_args.base_model_name_or_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(eval_args.base_model_name_or_path, use_fast=True)
    except Exception as e:
        logger.error(f"Failed to load tokenizer from base '{eval_args.base_model_name_or_path}': {e}", exc_info=True)
        raise

    logger.info(f"Loading base configuration from: {eval_args.base_model_name_or_path}")
    try:
        config = AutoConfig.from_pretrained(eval_args.base_model_name_or_path)
    except Exception as e:
        logger.error(f"Failed to load config from base '{eval_args.base_model_name_or_path}': {e}", exc_info=True)
        raise

    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set tokenizer pad_token to its eos_token ('{tokenizer.eos_token}')")
        else:
            logger.warning("Tokenizer has no pad_token and no eos_token. This might cause issues.")

    logger.info(f"Initializing model structure ({model_class.__name__}) with configuration (no weights loaded yet).")
    try:
        # Initialize model from config only, weights will be loaded per checkpoint
        model = model_class(config=config)
        logger.info("Successfully initialized model structure from config.")
    except Exception as e:
        logger.error(f"Failed to initialize model from config: {e}", exc_info=True)
        raise
    return model, tokenizer, config


def load_checkpoint_weights(model: PreTrainedModel, checkpoint_path: str):
    """Loads weights from a specific checkpoint into the provided model."""
    global logger
    ckpt_path_obj = Path(checkpoint_path)
    if not ckpt_path_obj.is_dir():
        logger.error(f"Checkpoint directory for weight loading not found: {ckpt_path_obj}")
        return False

    # Try to load from a PyTorch normal saved model file (e.g. pytorch_model.bin)
    # This is what `from_pretrained` would look for.
    # If your checkpoints are saved differently (e.g. custom state dicts), adjust this.
    try:
        logger.info(f"Attempting to load weights from checkpoint: {ckpt_path_obj}")
        # We create a temporary model from pretrained to get its state_dict
        # This is a common pattern if the checkpoint is a full HF model save
        # A more direct way if you only save state_dict: torch.load(ckpt_path_obj / "model_state.pt")
        # For now, assume standard HF save format in the checkpoint directory
        temp_model_for_weights = type(model).from_pretrained(ckpt_path_obj)
        model.load_state_dict(temp_model_for_weights.state_dict())
        del temp_model_for_weights  # Free memory
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logger.info(f"Successfully loaded weights from {ckpt_path_obj} into the model.")
        return True
    except Exception as e:
        logger.error(f"Failed to load weights from checkpoint {ckpt_path_obj}: {e}", exc_info=True)
        return False


# evaluate_standard remains largely the same, but takes the model directly
def evaluate_standard(args, model, eval_dataloader, device, current_checkpoint_label):
    """Runs standard evaluation (perplexity)."""
    global logger, tqdm
    if eval_dataloader is None:
        logger.warning(f"[{current_checkpoint_label}] Standard evaluation dataloader is None. Skipping.")
        return {}

    original_mode = model.training
    model.eval()
    total_loss = 0.0
    total_items = 0

    logger.info(f"[{current_checkpoint_label}] Starting standard perplexity evaluation...")
    progress_bar = tqdm(eval_dataloader, desc=f"Eval (Std PPL) for {current_checkpoint_label}", leave=False,
                        disable=not sys.stdout.isatty())

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            try:
                batch_on_device = {k: v.to(device, non_blocking=True) for k, v in batch.items() if
                                   isinstance(v, torch.Tensor)}
            except RuntimeError as e:
                logger.error(
                    f"[{current_checkpoint_label}] Error moving std eval batch {batch_idx} to device {device}: {e}")
                continue

            try:
                amp_enabled = args.use_amp and device.type == 'cuda'
                with autocast(enabled=amp_enabled):
                    outputs = model(**batch_on_device)
                    loss = outputs.loss
            except Exception as e:
                logger.error(
                    f"[{current_checkpoint_label}] Error during std eval forward pass on batch {batch_idx}: {e}",
                    exc_info=True)
                continue

            if loss is not None and torch.isfinite(loss):
                num_items_in_batch = batch_on_device['input_ids'].size(0)
                total_loss += loss.detach().item() * num_items_in_batch
                total_items += num_items_in_batch
                if batch_idx % 50 == 0:
                    progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
            else:
                logger.warning(
                    f"[{current_checkpoint_label}] Non-finite loss in std eval (batch {batch_idx}): {loss.item() if loss is not None else 'None'}.")

    metrics = {}
    if total_items > 0:
        final_avg_loss = total_loss / total_items
        try:
            perplexity = math.exp(min(final_avg_loss, 700)) if final_avg_loss > 0 else float('nan')
        except (OverflowError, ValueError):
            perplexity = float('inf') if final_avg_loss > 0 else float('nan')

        logger.info(
            f"[{current_checkpoint_label}] Std Eval Results: Avg Loss = {final_avg_loss:.4f}, PPL = {perplexity:.4f}, Items = {total_items}")
        metrics = {"loss": final_avg_loss, "perplexity": perplexity, "total_items": total_items}
    else:
        logger.warning(f"[{current_checkpoint_label}] Std eval completed, but total_items processed is zero.")
        metrics = {"loss": float('nan'), "perplexity": float('nan'), "total_items": 0}

    if original_mode: model.train()
    progress_bar.close()
    return metrics


# run_priming_evaluation_on_directory now takes the model, and the processed_data_cache
def run_priming_evaluation_on_directory(
        eval_args, model, tokenizer, device, neptune_run_obj,
        current_checkpoint_numeric_step, current_checkpoint_label,
        current_checkpoint_output_dir: Path,  # Specific output dir for this checkpoint's priming results
        processed_priming_data_cache: Dict[str, List[Dict[str, Any]]]  # In-memory cache
):
    """
    Finds CSVs, creates dataloaders (using cache & parallel loading), runs priming eval,
    aggregates summary results, writes raw per-item results to a CSV for *this checkpoint*,
    and logs to Neptune.
    """
    global logger, tqdm
    if not PRIMING_LIBS_AVAILABLE:
        logger.error(f"[{current_checkpoint_label}] Priming libs not available. Skipping priming eval.")
        return {"error": "Priming library import failed."}

    if not eval_args.run_priming_eval or not eval_args.priming_eval_dir_path:
        logger.info(f"[{current_checkpoint_label}] Skipping priming eval (not enabled or path not provided).")
        return {}

    priming_dir = Path(eval_args.priming_eval_dir_path)
    if not priming_dir.is_dir():
        logger.error(f"[{current_checkpoint_label}] Priming eval directory not found: {priming_dir}")
        return {"error": f"Priming directory not found: {priming_dir}"}

    # Ensure the specific output directory for this checkpoint's priming results exists
    current_checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    raw_priming_csv_path = current_checkpoint_output_dir / f"priming_results_raw_{current_checkpoint_label}.csv"
    logger.info(f"[{current_checkpoint_label}] Raw priming results CSV: {raw_priming_csv_path}")

    all_csv_files_in_priming_dir = sorted(list(priming_dir.glob('*.csv')))
    if not all_csv_files_in_priming_dir:
        logger.warning(f"[{current_checkpoint_label}] No *.csv files found in priming directory: {priming_dir}")
        return {}

    all_corpus_summary_metrics = {}
    neptune_metrics_to_log_this_step = {}
    logger.info(f"[{current_checkpoint_label}] Found {len(all_csv_files_in_priming_dir)} CSVs for priming eval.")

    original_model_training_mode = model.training
    model.eval()

    # --- Parallel Loading of Priming Data ---
    corpus_data_for_dataloaders: Dict[str, List[Dict[str, Any]]] = {}  # filename -> processed_data

    def _load_and_process_single_corpus(corpus_path_obj: Path) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        corpus_path_str = str(corpus_path_obj)
        corpus_filename = corpus_path_obj.name

        if corpus_path_str in processed_priming_data_cache:
            logger.debug(f"[{current_checkpoint_label}] Using cached processed data for {corpus_filename}")
            return corpus_filename, processed_priming_data_cache[corpus_path_str]

        logger.debug(f"[{current_checkpoint_label}] Loading and processing priming data for {corpus_filename}...")
        try:
            # Assuming load_and_process_priming_data is from your priming_evaluation.data_loader
            # It should take Path, tokenizer, and delimiter
            _data = load_and_process_priming_data(
                csv_path=corpus_path_obj,
                tokenizer=tokenizer,  # Pass tokenizer if your loader uses it for validation/etc.
                delimiter=eval_args.priming_delimiter
            )
            if _data:
                processed_priming_data_cache[corpus_path_str] = _data  # Cache it
            return corpus_filename, _data
        except Exception as e_load:
            logger.error(f"[{current_checkpoint_label}] Error loading/processing {corpus_filename}: {e_load}",
                         exc_info=True)
            return corpus_filename, None

    num_cores_for_loading = min(os.cpu_count() or 1, len(all_csv_files_in_priming_dir), 8)  # Cap at 8 for sanity
    logger.info(
        f"[{current_checkpoint_label}] Using up to {num_cores_for_loading} workers for parallel priming corpus loading.")

    with ThreadPoolExecutor(max_workers=num_cores_for_loading) as executor:
        future_to_corpus_path = {
            executor.submit(_load_and_process_single_corpus, fp): fp for fp in all_csv_files_in_priming_dir
        }
        for future in tqdm(as_completed(future_to_corpus_path), total=len(all_csv_files_in_priming_dir),
                           desc=f"Loading Priming Corpuses for {current_checkpoint_label}", leave=False):
            # corpus_path_obj_completed = future_to_corpus_path[future]
            try:
                fname, data_list = future.result()
                if data_list:
                    corpus_data_for_dataloaders[fname] = data_list
                else:
                    logger.warning(f"[{current_checkpoint_label}] No data returned after processing {fname}")
            except Exception as exc_future:
                # corpus_name_exc = Path(future_to_corpus_path[future]).name
                logger.error(
                    f"[{current_checkpoint_label}] Priming corpus loading generated an exception: {exc_future}",
                    exc_info=True)
    # --- End Parallel Loading ---

    if not corpus_data_for_dataloaders:
        logger.warning(
            f"[{current_checkpoint_label}] No priming data loaded after parallel processing. Skipping priming eval for this checkpoint.")
        if original_model_training_mode: model.train()
        return {"error": "No priming data successfully loaded."}

    csv_file_handle, csv_writer_obj = None, None
    try:
        csv_file_handle = open(raw_priming_csv_path, 'w', newline='', encoding='utf-8')
        csv_writer_obj = csv.writer(csv_file_handle)
        csv_header = ["eval_step_label", "corpus_file", "target_structure", "item_index", "pe", "logp_con",
                      "logp_incon"]
        csv_writer_obj.writerow(csv_header)
    except IOError as e_csv:
        logger.error(f"[{current_checkpoint_label}] Failed to open raw priming CSV {raw_priming_csv_path}: {e_csv}")
        if csv_file_handle: csv_file_handle.close()
        if original_model_training_mode: model.train()
        return {"error": f"Raw priming CSV write error: {e_csv}"}

    for corpus_filename_key in sorted(corpus_data_for_dataloaders.keys()):  # Process in defined order
        processed_data = corpus_data_for_dataloaders[corpus_filename_key]

        logger.info(f"--- [{current_checkpoint_label}] Running Priming Eval for Corpus: {corpus_filename_key} ---")
        current_corpus_dataloader = None
        try:
            # Create dataset and dataloader from the (potentially cached and parallel-loaded) processed_data
            # The create_priming_dataloader needs to be adapted to take processed_data list directly,
            # or we construct the PrimingEvaluationDataset here.
            # For simplicity, let's assume create_priming_dataloader can take `processed_data`
            # and a dummy path for logging, or we slightly refactor it.
            # Let's try to call your existing create_priming_dataloader logic,
            # but it expects a csv_path. We'll need to simulate parts of it.

            # Modification: Directly use the processed_data
            if not processed_data:
                logger.warning(
                    f"[{current_checkpoint_label}] No processed data for {corpus_filename_key}. Skipping dataloader creation.")
                all_corpus_summary_metrics[corpus_filename_key] = {"error": "No processed data available."}
                continue

            from priming_evaluation.data_loader import PrimingEvaluationDataset  # Assuming this class exists
            from functools import partial
            from priming_evaluation.data_loader import collate_priming_eval_batch  # Assuming this exists

            # Manually create dataset and dataloader if create_priming_dataloader is too tied to file paths
            priming_dataset = PrimingEvaluationDataset(processed_data)
            if len(priming_dataset) == 0:
                logger.warning(
                    f"[{current_checkpoint_label}] Dataset for {corpus_filename_key} is empty. Max samples: {eval_args.priming_eval_max_samples_per_file}.")
                all_corpus_summary_metrics[corpus_filename_key] = {"error": "Dataset empty after processing/sampling."}
                continue

            # Apply sampling if max_samples is set (this was in original create_priming_dataloader)
            # This sampling should ideally happen in _load_and_process_single_corpus or just after it.
            # For now, let's assume load_and_process_priming_data handles sampling if max_samples is a param to it.
            # Or, apply it here if `processed_data` is the full set.
            # Let's assume `eval_args.priming_eval_max_samples_per_file` would be handled by `load_and_process_priming_data`
            # or if not, it needs to be added there or here.
            # For now, let's simplify and assume `processed_data` is already sampled if needed.

            collate_fn_partial = partial(collate_priming_eval_batch, tokenizer=tokenizer,
                                         join_string=eval_args.priming_delimiter + " ",
                                         max_length=getattr(tokenizer, 'model_max_length', None))

            current_corpus_dataloader = DataLoader(
                priming_dataset,
                sampler=SequentialSampler(priming_dataset),
                batch_size=eval_args.priming_per_device_eval_batch_size,
                collate_fn=collate_fn_partial,
                num_workers=eval_args.num_workers,
                pin_memory=True,  # Good for GPU
                shuffle=False
            )
            logger.info(
                f"[{current_checkpoint_label}] Priming DataLoader created for {corpus_filename_key} with {len(priming_dataset)} items.")

        except Exception as e_dl:
            logger.error(f"[{current_checkpoint_label}] Dataloader creation failed for {corpus_filename_key}: {e_dl}",
                         exc_info=True)
            all_corpus_summary_metrics[corpus_filename_key] = {"error": f"Dataloader creation failed: {e_dl}"}
            continue

        if current_corpus_dataloader is None or len(
                current_corpus_dataloader.dataset) == 0:  # Redundant check if above handles it
            logger.warning(
                f"[{current_checkpoint_label}] Dataloader for {corpus_filename_key} is None or empty. Skipping.")
            all_corpus_summary_metrics[corpus_filename_key] = {"error": "Dataloader None/empty."}
            continue

        try:
            corpus_summary_metrics, corpus_raw_item_results = run_native_priming_eval(
                model=model, priming_dataloader=current_corpus_dataloader, device=device,
                tokenizer=tokenizer, use_amp=eval_args.use_amp
            )
            all_corpus_summary_metrics[corpus_filename_key] = corpus_summary_metrics
            logger.info(
                f"[{current_checkpoint_label}] Priming Summary for {corpus_filename_key}: {corpus_summary_metrics}")

            if NEPTUNE_AVAILABLE and neptune_run_obj:
                sanitized_corpus_name = corpus_filename_key.replace('.', '_').replace('/', '_')
                neptune_log_prefix = f"eval_metrics/priming/{sanitized_corpus_name}"
                finite_numeric_metrics = {k: v for k, v in corpus_summary_metrics.items() if
                                          isinstance(v, (int, float)) and math.isfinite(v)}
                for metric_key, metric_value in finite_numeric_metrics.items():
                    # Log with the *numeric step* of the checkpoint for time-series
                    neptune_run_obj[f"{neptune_log_prefix}/{metric_key}"].append(metric_value,
                                                                                 step=current_checkpoint_numeric_step)

            if csv_writer_obj and corpus_raw_item_results:
                items_written_count = 0
                for target_structure, results_list in corpus_raw_item_results.items():
                    for item_idx, item_data in enumerate(results_list):
                        if isinstance(item_data, dict):
                            pe = item_data.get('pe', float('nan'))
                            lpc = item_data.get('logp_con', float('nan'))
                            lpi = item_data.get('logp_incon', float('nan'))
                            csv_writer_obj.writerow([
                                current_checkpoint_label, corpus_filename_key, target_structure, item_idx,
                                f"{pe:.6f}" if not math.isnan(pe) else 'NaN',
                                f"{lpc:.6f}" if not math.isnan(lpc) else 'NaN',
                                f"{lpi:.6f}" if not math.isnan(lpi) else 'NaN'
                            ])
                            items_written_count += 1
                        else:
                            logger.warning(
                                f"Skipping non-dict item in raw results for {corpus_filename_key}, {target_structure}")
                if csv_file_handle: csv_file_handle.flush()
                logger.info(
                    f"[{current_checkpoint_label}] Wrote {items_written_count} raw priming results to CSV for {corpus_filename_key}.")

        except Exception as e_eval:
            logger.error(f"[{current_checkpoint_label}] Priming eval run failed for {corpus_filename_key}: {e_eval}",
                         exc_info=True)
            all_corpus_summary_metrics[corpus_filename_key] = {"error": f"Core eval run failed: {e_eval}"}
        finally:
            del current_corpus_dataloader
            gc.collect()

    if csv_file_handle:
        try:
            csv_file_handle.close()
            logger.info(f"[{current_checkpoint_label}] Closed raw priming results CSV: {raw_priming_csv_path}")
        except Exception as e_close_csv:
            logger.error(
                f"[{current_checkpoint_label}] Error closing raw priming CSV {raw_priming_csv_path}: {e_close_csv}")

    if original_model_training_mode: model.train()
    logger.info(f"--- [{current_checkpoint_label}] Finished All Priming Evals ---")
    return all_corpus_summary_metrics


# --- Argument Parser for evaluate.py (Modified) ---
def parse_eval_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM checkpoints, with multi-checkpoint and watch capabilities.")

    # Checkpoint Specification (mutually exclusive with watch mode for initial scan logic)
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to a single checkpoint directory to evaluate (if not using --checkpoint_dir or --watch_mode).")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory containing multiple checkpoint folders (e.g., 'checkpoint-1000', 'final_model'). Evaluates all found and ready.")

    # Watch Mode
    parser.add_argument("--watch_mode", action="store_true",
                        help="Enable watch mode to continuously monitor --checkpoint_dir for new, ready checkpoints.")
    parser.add_argument("--watch_interval_seconds", type=int, default=300,
                        help="Polling interval in seconds for watch mode.")
    parser.add_argument("--checkpoint_ready_sentinel", type=str, default="EVAL_READY.txt",
                        help="Filename within a checkpoint directory indicating it's ready for evaluation.")

    # General Paths & Config
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base directory to save all evaluation results. Subdirs for each checkpoint will be created here.")
    parser.add_argument("--base_model_name_or_path", type=str, default="gpt2",  # Renamed for clarity
                        help="Base model identifier (e.g., 'gpt2') or path to load initial tokenizer, config, and model architecture.")
    parser.add_argument("--model_class_name", type=str, default="GPT2LMHeadModel",
                        help="Name of the Hugging Face model class (e.g., GPT2LMHeadModel).")

    # Evaluation Types
    parser.add_argument("--run_standard_eval", action="store_true", default=False,
                        help="Run standard perplexity evaluation.")
    parser.add_argument("--validation_dataset_path", type=str, default=None, help="Path for --run_standard_eval.")
    parser.add_argument("--eval_max_samples", type=int, default=50000,
                        help="Max samples for standard eval. <= 0 for full.")

    parser.add_argument("--run_priming_eval", action="store_true", default=False, help="Run priming evaluation.")
    parser.add_argument("--priming_eval_dir_path", type=str, default=None, help="Directory with priming CSVs.")
    parser.add_argument("--priming_eval_max_samples_per_file", type=int, default=1000,
                        help="Max samples per priming CSV. <=0 for all.")
    parser.add_argument("--priming_delimiter", type=str, default=".", help="Delimiter in priming CSVs.")

    # Batch Sizes & Workers
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size for standard eval.")
    parser.add_argument("--priming_per_device_eval_batch_size", type=int, default=None,
                        help="Batch size for priming. Defaults to per_device_eval_batch_size.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")

    # Misc
    parser.add_argument("--use_amp", action="store_true", help="Enable AMP for evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Neptune (these are mostly for the orchestrator to pass through)
    parser.add_argument("--neptune_project", type=str, default=None)
    parser.add_argument("--neptune_run_id", type=str, default=None,
                        help="Log to an existing Neptune run (rarely used by this script directly).")
    parser.add_argument("--neptune_api_token", type=str, default=None)
    parser.add_argument("--neptune_tags", type=str, nargs='+', default=None)
    # parser.add_argument("--neptune_run_name", type=str, default=None) # The eval_job.sh will construct this. This script makes sub-logs.

    # This arg is now determined dynamically per checkpoint
    # parser.add_argument("--checkpoint_label", type=str, default=None, help="Descriptive label for the checkpoint")

    args = parser.parse_args()

    if args.priming_per_device_eval_batch_size is None:
        args.priming_per_device_eval_batch_size = args.per_device_eval_batch_size

    # Validations
    if not args.checkpoint_path and not args.checkpoint_dir:
        parser.error("Either --checkpoint_path (for single) or --checkpoint_dir (for multi/watch) must be specified.")
    if args.watch_mode and not args.checkpoint_dir:
        parser.error("--watch_mode requires --checkpoint_dir to be specified.")
    if args.checkpoint_path and args.checkpoint_dir:
        logger.warning(
            "--checkpoint_path and --checkpoint_dir both specified. --checkpoint_path will be evaluated first, then --checkpoint_dir (if not in watch mode). Consider using only one.")

    if args.run_standard_eval and not args.validation_dataset_path: parser.error(
        "--validation_dataset_path required for --run_standard_eval.")
    if args.run_priming_eval and not args.priming_eval_dir_path: parser.error(
        "--priming_eval_dir_path required for --run_priming_eval.")
    if args.validation_dataset_path and not Path(args.validation_dataset_path).is_dir(): parser.error(
        f"Validation dataset dir not found: {args.validation_dataset_path}")
    if args.priming_eval_dir_path and not Path(args.priming_eval_dir_path).is_dir(): parser.error(
        f"Priming eval dir not found: {args.priming_eval_dir_path}")

    try:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        parser.error(f"Failed to create output_dir {args.output_dir}: {e}")
    return args


def get_checkpoint_numeric_step(checkpoint_name: str) -> int:
    """Extracts numeric step from checkpoint name like 'checkpoint-12345', or -1 for 'final_model' or others."""
    if checkpoint_name == "final_model":
        return -2  # Special value for final_model, distinct from error/unknown -1
    try:
        return int(checkpoint_name.split('-')[-1])
    except (ValueError, IndexError):
        return -1  # Fallback for non-numeric or unparseable


def find_ready_checkpoints(target_dir: Path, sentinel_filename: str, already_processed_checkpoints: Set[str]) -> List[
    Path]:
    """Scans target_dir for new, ready checkpoint subdirectories."""
    global logger
    ready_checkpoints = []
    if not target_dir.is_dir():
        logger.warning(f"Target directory for checkpoint scanning does not exist: {target_dir}")
        return ready_checkpoints

    for item in target_dir.iterdir():
        if item.is_dir() and item.name not in already_processed_checkpoints:
            if (item.name.startswith("checkpoint-") or item.name == "final_model"):
                sentinel_file = item / sentinel_filename
                if sentinel_file.is_file():
                    ready_checkpoints.append(item)
                else:
                    logger.debug(
                        f"Checkpoint '{item.name}' found, but sentinel '{sentinel_filename}' is missing. Will re-check later if in watch mode.")

    # Sort by numeric step, 'final_model' usually last or handled specially
    def sort_key(p: Path):
        num_step = get_checkpoint_numeric_step(p.name)
        if num_step == -2: return float('inf')  # final_model at the end
        if num_step == -1: return float('inf') - 1  # other non-numeric before final_model
        return num_step

    ready_checkpoints.sort(key=sort_key)
    if ready_checkpoints:
        logger.info(f"Found {len(ready_checkpoints)} new, ready checkpoints: {[p.name for p in ready_checkpoints]}")
    return ready_checkpoints


# --- Main Evaluation Execution (Heavily Modified) ---
def main_script_entry():
    global logger, tqdm, NEPTUNE_AVAILABLE, neptune  # Make tqdm available after import
    try:
        from tqdm.auto import tqdm as imported_tqdm
        tqdm = imported_tqdm
    except ImportError:
        print("Warning: tqdm.auto not installed. Progress bars will be basic or disabled.", file=sys.stderr)

        def fallback_tqdm(iterable, *args, **kwargs):
            disable = kwargs.pop('disable', False)
            if not disable and sys.stdout.isatty(): print(f"Processing {kwargs.get('desc', 'items')}...")
            return iterable

        tqdm = fallback_tqdm

    eval_args = parse_eval_args()

    # Log file for the entire run of this script (multi-checkpoint)
    # Individual checkpoint logs might go into their subdirs if needed, but this is the main one.
    main_log_file_name = f"evaluate_multi_checkpoint_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    main_log_file_path = Path(eval_args.output_dir) / main_log_file_name
    setup_logging(log_file=str(main_log_file_path))
    logger = logging.getLogger(__name__)

    logger.info("***** Starting Multi-Checkpoint Evaluation Script *****")
    logger.info(f"Full Evaluation Arguments: {vars(eval_args)}")

    device = get_device()
    set_seed(eval_args.seed)

    # --- Neptune Setup (for the entire multi-checkpoint run) ---
    # Individual checkpoints will log their metrics under this single run if possible.
    neptune_run_obj = None
    if NEPTUNE_AVAILABLE and eval_args.neptune_project:
        logger.info("Attempting to initialize Neptune for this evaluation session...")
        neptune_api_token_to_use = eval_args.neptune_api_token or os.getenv('NEPTUNE_API_TOKEN')
        if not neptune_api_token_to_use:
            logger.warning("Neptune project specified, but no API token. Neptune logging disabled.")
        else:
            try:
                # Name for this multi-checkpoint evaluation run (e.g., from Slurm job name or shared ID)
                # The eval_job.sh should set SINGULARITYENV_NEPTUNE_TRAINING_RUN_NAME which includes SHARED_RUN_ID
                # We can use that to make a unique name for this multi-eval job.
                shared_run_id_from_env = os.getenv('SHARED_RUN_ID',
                                                   f"multi_eval_{datetime.now().strftime('%Y%m%d_%H%M')}")
                neptune_eval_session_name = f"eval_session_{shared_run_id_from_env}"

                current_neptune_tags = ['evaluation_session']
                if eval_args.neptune_tags: current_neptune_tags.extend(eval_args.neptune_tags)
                # Link to training run if name is available (e.g. from orchestrator)
                linked_training_run_name = os.getenv('SINGULARITYENV_NEPTUNE_TRAINING_RUN_NAME')
                if linked_training_run_name:
                    current_neptune_tags.append(f"train_ref:{linked_training_run_name}")

                neptune_run_obj = neptune.init_run(
                    project=eval_args.neptune_project,
                    api_token=neptune_api_token_to_use,
                    name=neptune_eval_session_name,
                    tags=sorted(list(set(current_neptune_tags))),
                    mode="async"  # Default mode
                )
                neptune_run_obj["evaluation_session/script_args"] = vars(eval_args)
                if linked_training_run_name:
                    neptune_run_obj["evaluation_session/details/linked_training_run_name"] = linked_training_run_name
                logger.info(f"Neptune initialized for session. Run URL: {neptune_run_obj.get_url()}")
            except Exception as e_neptune:
                logger.error(f"Neptune initialization failed: {e_neptune}. Logging disabled.", exc_info=True)
                neptune_run_obj = None
    else:
        logger.info("Neptune logging disabled (client not available or project not specified).")

    # --- Load Base Model Structure and Tokenizer ---
    logger.info(f"--- Loading Base Model Structure and Tokenizer ---")
    try:
        model_class_map = {"GPT2LMHeadModel": GPT2LMHeadModel}  # Extend as needed
        model_constructor = model_class_map.get(eval_args.model_class_name, GPT2LMHeadModel)

        # Load base architecture (no weights) and tokenizer
        base_model, tokenizer, _ = load_base_model_and_tokenizer(eval_args, model_constructor)
        base_model.to(device)
        base_model.eval()  # Ensure eval mode
        logger.info(f"Base model '{eval_args.model_class_name}' structure loaded to '{device}' and set to eval mode.")
    except Exception as e_model_struct:
        logger.critical(f"FATAL: Failed to load base model structure/tokenizer: {e_model_struct}", exc_info=True)
        if neptune_run_obj:
            neptune_run_obj[
                "evaluation_session/critical_error"] = f"Base model/tokenizer load failed: {traceback.format_exc()}"
            neptune_run_obj.stop()
        sys.exit(1)

    # --- Prepare Standard Evaluation Dataloader (once, if enabled) ---
    std_eval_dataloader = None
    if eval_args.run_standard_eval:
        logger.info("--- Preparing Standard Evaluation (Perplexity) Dataloader ---")
        if not eval_args.validation_dataset_path:
            logger.error("Std perplexity eval requested, but --validation_dataset_path missing. Skipping.")
        else:
            try:
                validation_hf_dataset = load_from_disk(eval_args.validation_dataset_path)
                # Apply sampling
                if eval_args.eval_max_samples and 0 < eval_args.eval_max_samples < len(validation_hf_dataset):
                    np_rng = np.random.RandomState(eval_args.seed)
                    sampled_indices = np_rng.choice(len(validation_hf_dataset), size=eval_args.eval_max_samples,
                                                    replace=False)
                    validation_hf_dataset = validation_hf_dataset.select(sampled_indices)
                    logger.info(f"Using subset for std perplexity: {len(validation_hf_dataset)} samples.")

                lm_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
                std_eval_dataloader = DataLoader(
                    validation_hf_dataset, sampler=SequentialSampler(validation_hf_dataset),
                    batch_size=eval_args.per_device_eval_batch_size, num_workers=eval_args.num_workers,
                    pin_memory=True, collate_fn=lm_data_collator
                )
                logger.info("Standard Perplexity Eval DataLoader prepared.")
            except Exception as e_std_data:
                logger.error(f"Failed to load/prepare std eval data: {e_std_data}", exc_info=True)
                std_eval_dataloader = None

    # --- Main Processing Logic ---
    processed_checkpoints: Set[str] = set()
    all_checkpoints_summary_results = {}  # Store summary from each processed checkpoint
    processed_priming_data_cache: Dict[str, List[Dict[str, Any]]] = {}  # In-memory cache for priming data
    final_model_successfully_processed = False
    overall_start_time = time.time()

    try:
        # Initial scan based on args
        initial_checkpoints_to_process: List[Path] = []
        scan_directory = None
        if eval_args.checkpoint_dir:
            scan_directory = Path(eval_args.checkpoint_dir)
        elif eval_args.watch_mode:  # Should have checkpoint_dir if watch_mode is true (validated in parse_args)
            logger.error(
                "Watch mode enabled but checkpoint_dir is None. This should not happen due to arg validation.")  # Should be caught by parser

        if eval_args.checkpoint_path:  # Single, specific checkpoint takes precedence for first eval
            cp_path = Path(eval_args.checkpoint_path)
            if cp_path.is_dir():
                sentinel = cp_path / eval_args.checkpoint_ready_sentinel
                if sentinel.is_file():
                    initial_checkpoints_to_process.append(cp_path)
                    logger.info(f"Prioritizing single specified checkpoint: {cp_path.name}")
                else:
                    logger.warning(
                        f"Specified checkpoint {cp_path.name} is missing sentinel '{eval_args.checkpoint_ready_sentinel}'. Skipping it for now.")
            else:
                logger.warning(f"Specified checkpoint_path {cp_path} is not a directory. Skipping.")

        if scan_directory:  # If --checkpoint_dir was given, scan it (respects --watch_mode for subsequent runs)
            logger.info(f"Performing initial scan of directory: {scan_directory}")
            initial_checkpoints_to_process.extend(
                find_ready_checkpoints(scan_directory, eval_args.checkpoint_ready_sentinel, processed_checkpoints)
            )
            # Deduplicate if checkpoint_path was also in checkpoint_dir
            initial_checkpoints_to_process = sorted(list(set(initial_checkpoints_to_process)),
                                                    key=lambda p: get_checkpoint_numeric_step(p.name))

        # Process initial list
        if initial_checkpoints_to_process:
            logger.info(f"--- Processing {len(initial_checkpoints_to_process)} initially found ready checkpoints ---")
            for ckpt_path_obj in initial_checkpoints_to_process:
                if ckpt_path_obj.name in processed_checkpoints: continue  # Should be handled by find_ready_checkpoints

                ckpt_label = ckpt_path_obj.name
                ckpt_numeric_step = get_checkpoint_numeric_step(ckpt_label)
                logger.info(f"===== Processing Checkpoint: {ckpt_label} (Step: {ckpt_numeric_step}) =====")

                current_ckpt_output_dir = Path(eval_args.output_dir) / ckpt_label
                current_ckpt_output_dir.mkdir(parents=True, exist_ok=True)

                if not load_checkpoint_weights(base_model, str(ckpt_path_obj)):
                    logger.error(f"Failed to load weights for {ckpt_label}. Skipping this checkpoint.")
                    all_checkpoints_summary_results[ckpt_label] = {"error": "Failed to load weights."}
                    if neptune_run_obj:
                        neptune_run_obj[f"evaluation_status/{ckpt_label}/error"] = "Weight loading failed"
                    continue  # Skip to next checkpoint

                checkpoint_specific_results = {
                    "checkpoint_label": ckpt_label,
                    "checkpoint_numeric_step": ckpt_numeric_step
                }

                if eval_args.run_standard_eval and std_eval_dataloader:
                    std_metrics = evaluate_standard(eval_args, base_model, std_eval_dataloader, device, ckpt_label)
                    checkpoint_specific_results["standard_perplexity_summary"] = std_metrics
                    if neptune_run_obj and std_metrics:
                        loss_val = std_metrics.get("loss")
                        ppl_val = std_metrics.get("perplexity")
                        if loss_val is not None and math.isfinite(loss_val): neptune_run_obj[
                            f"eval_metrics/standard_loss"].append(loss_val, step=ckpt_numeric_step)
                        if ppl_val is not None and math.isfinite(ppl_val): neptune_run_obj[
                            f"eval_metrics/perplexity"].append(ppl_val, step=ckpt_numeric_step)

                if eval_args.run_priming_eval:
                    priming_summary = run_priming_evaluation_on_directory(
                        eval_args, base_model, tokenizer, device, neptune_run_obj,
                        ckpt_numeric_step, ckpt_label, current_ckpt_output_dir,
                        processed_priming_data_cache
                    )
                    checkpoint_specific_results["priming_evaluation_summary"] = priming_summary
                    # Neptune logging for priming is handled inside run_priming_evaluation_on_directory

                summary_json_path = current_ckpt_output_dir / f"evaluation_summary_{ckpt_label}.json"
                with open(summary_json_path, "w", encoding='utf-8') as f_sum:
                    json.dump(checkpoint_specific_results, f_sum, indent=4)
                logger.info(f"Saved checkpoint summary to: {summary_json_path}")
                if neptune_run_obj:  # Log the summary dict to Neptune as well
                    neptune_run_obj[f"evaluation_details/{ckpt_label}/full_summary_dict"] = checkpoint_specific_results

                all_checkpoints_summary_results[ckpt_label] = checkpoint_specific_results
                processed_checkpoints.add(ckpt_label)
                if ckpt_label == "final_model":
                    final_model_successfully_processed = True

                # Clear some memory if possible
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Watch Mode Loop
        if eval_args.watch_mode and scan_directory:  # scan_directory must be set for watch mode
            logger.info(
                f"--- Entering Watch Mode (Dir: {scan_directory}, Interval: {eval_args.watch_interval_seconds}s) ---")
            while True:
                if final_model_successfully_processed:
                    logger.info("'final_model' has been successfully processed. Exiting watch mode.")
                    # Check for orchestrator's training completion signal as well
                    # (e.g. SHARED_OUTPUT_DIR_HOST / TRAINING_COMPLETED.txt)
                    # This logic might be better in the Slurm script if it needs to kill the job.
                    # For now, `evaluate.py` exits its watch loop.
                    training_complete_sentinel_path = scan_directory / "TRAINING_COMPLETED.txt"  # Adjust if needed
                    if training_complete_sentinel_path.exists():
                        logger.info(f"'{training_complete_sentinel_path}' detected. Confirming end of run.")
                    break

                newly_ready_checkpoints = find_ready_checkpoints(
                    scan_directory, eval_args.checkpoint_ready_sentinel, processed_checkpoints
                )

                if newly_ready_checkpoints:
                    logger.info(f"--- Watch Mode: Found {len(newly_ready_checkpoints)} new ready checkpoints ---")
                    # Process this new list (similar loop as above)
                    for ckpt_path_obj in newly_ready_checkpoints:
                        # Double check, though find_ready_checkpoints should handle it
                        if ckpt_path_obj.name in processed_checkpoints: continue

                        ckpt_label = ckpt_path_obj.name
                        ckpt_numeric_step = get_checkpoint_numeric_step(ckpt_label)
                        logger.info(
                            f"===== Processing Checkpoint (Watch): {ckpt_label} (Step: {ckpt_numeric_step}) =====")

                        current_ckpt_output_dir = Path(eval_args.output_dir) / ckpt_label
                        current_ckpt_output_dir.mkdir(parents=True, exist_ok=True)

                        if not load_checkpoint_weights(base_model, str(ckpt_path_obj)):
                            logger.error(f"Failed to load weights for {ckpt_label} (Watch). Skipping.")
                            all_checkpoints_summary_results[ckpt_label] = {"error": "Failed to load weights."}
                            if neptune_run_obj: neptune_run_obj[
                                f"evaluation_status/{ckpt_label}/error"] = "Weight loading failed"
                            continue

                        checkpoint_specific_results = {
                            "checkpoint_label": ckpt_label,
                            "checkpoint_numeric_step": ckpt_numeric_step
                        }
                        if eval_args.run_standard_eval and std_eval_dataloader:
                            std_metrics = evaluate_standard(eval_args, base_model, std_eval_dataloader, device,
                                                            ckpt_label)
                            checkpoint_specific_results["standard_perplexity_summary"] = std_metrics
                            if neptune_run_obj and std_metrics:
                                loss_val = std_metrics.get("loss")
                                ppl_val = std_metrics.get("perplexity")
                                if loss_val is not None and math.isfinite(loss_val): neptune_run_obj[
                                    f"eval_metrics/standard_loss"].append(loss_val, step=ckpt_numeric_step)
                                if ppl_val is not None and math.isfinite(ppl_val): neptune_run_obj[
                                    f"eval_metrics/perplexity"].append(ppl_val, step=ckpt_numeric_step)

                        if eval_args.run_priming_eval:
                            priming_summary = run_priming_evaluation_on_directory(
                                eval_args, base_model, tokenizer, device, neptune_run_obj,
                                ckpt_numeric_step, ckpt_label, current_ckpt_output_dir,
                                processed_priming_data_cache
                            )
                            checkpoint_specific_results["priming_evaluation_summary"] = priming_summary

                        summary_json_path = current_ckpt_output_dir / f"evaluation_summary_{ckpt_label}.json"
                        with open(summary_json_path, "w", encoding='utf-8') as f_sum:
                            json.dump(checkpoint_specific_results, f_sum, indent=4)
                        logger.info(f"Saved checkpoint summary to: {summary_json_path}")
                        if neptune_run_obj: neptune_run_obj[
                            f"evaluation_details/{ckpt_label}/full_summary_dict"] = checkpoint_specific_results

                        all_checkpoints_summary_results[ckpt_label] = checkpoint_specific_results
                        processed_checkpoints.add(ckpt_label)
                        if ckpt_label == "final_model":
                            final_model_successfully_processed = True
                            if final_model_successfully_processed:  # Break inner loop to re-check outer while condition
                                logger.info(
                                    "'final_model' processed during watch. Will exit watch mode on next iteration.")
                                break

                        gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()

                    if final_model_successfully_processed:  # If final_model was in the batch, re-check outer loop condition
                        continue

                else:  # No new checkpoints found in this iteration
                    logger.info(
                        f"Watch Mode: No new ready checkpoints found. Sleeping for {eval_args.watch_interval_seconds}s...")

                time.sleep(eval_args.watch_interval_seconds)
            logger.info("--- Exited Watch Mode ---")
        else:  # Not in watch mode
            logger.info("--- Single-pass evaluation complete (Watch Mode was not enabled or applicable) ---")

    except KeyboardInterrupt:
        logger.warning("Evaluation script interrupted by user (KeyboardInterrupt).")
        if neptune_run_obj: neptune_run_obj["evaluation_session/status"] = "interrupted_keyboard"
    except Exception as e_main:
        logger.critical(f"An unhandled exception occurred in the main evaluation flow: {e_main}", exc_info=True)
        if neptune_run_obj: neptune_run_obj[
            "evaluation_session/critical_error"] = f"Main flow exception: {traceback.format_exc()}"
    finally:
        total_script_duration_seconds = time.time() - overall_start_time
        logger.info(f"Total evaluation script execution time: {total_script_duration_seconds:.2f} seconds.")
        logger.info(f"Processed {len(processed_checkpoints)} unique checkpoints: {sorted(list(processed_checkpoints))}")

        # Save a final summary of all checkpoints processed in this run
        overall_summary_path = Path(
            eval_args.output_dir) / f"all_evaluated_checkpoints_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(overall_summary_path, "w", encoding='utf-8') as f_all_sum:
                json.dump(all_checkpoints_summary_results, f_all_sum, indent=4)
            logger.info(f"Saved overall summary of all processed checkpoints to: {overall_summary_path}")
            if neptune_run_obj:
                neptune_run_obj["evaluation_session/all_checkpoints_summary_dict"] = all_checkpoints_summary_results
                neptune_run_obj["evaluation_session/details/total_duration_seconds"] = total_script_duration_seconds
                neptune_run_obj["evaluation_session/details/total_checkpoints_processed"] = len(processed_checkpoints)
        except Exception as e_save_all_json:
            logger.error(f"Failed to save overall summary JSON: {e_save_all_json}")

        if NEPTUNE_AVAILABLE and neptune_run_obj:
            try:
                neptune_run_obj.sync()
                neptune_run_obj.stop()
                logger.info("Neptune run stopped successfully.")
            except Exception as e_neptune_stop:
                logger.error(f"Neptune stop operation failed: {e_neptune_stop}")

        logger.info(f"***** Evaluation Script Finished *****")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main_script_entry()