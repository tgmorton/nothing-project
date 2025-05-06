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
import re # For sorting checkpoint directories
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union # Added Union

# Standard ML/data imports
import numpy as np
import torch
import torch.linalg # Added for norm calculations
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,  # Assuming this is the model class used
    PreTrainedTokenizer,
    BatchEncoding,
    PretrainedConfig
)
from datasets import load_from_disk, Dataset # Added Dataset for type hinting
from torch.cuda.amp import autocast # For evaluate_standard AMP context

# Priming evaluation imports
PRIMING_EVAL_AVAILABLE = False
try:
    from priming_evaluation.data_loader import create_priming_dataloader
    from priming_evaluation.evaluator import run_native_priming_eval
    PRIMING_EVAL_AVAILABLE = True
except ImportError as e:
    # Log warning later if user tries to run priming eval
    create_priming_dataloader = None
    run_native_priming_eval = None

# Optional Neptune import
NEPTUNE_AVAILABLE = False
try:
    import neptune
    from neptune.sdk.run import Run as NeptuneRun # Type hint
    NEPTUNE_AVAILABLE = True
except ImportError:
    neptune = None # Define as None if not available
    NeptuneRun = None # Define type hint as None


# --- Globals ---
logger = None

# --- Helper Functions ---

def get_device() -> torch.device:
    """Gets the appropriate device for PyTorch computations."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0") # Default to first GPU for eval
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")
    return device

def setup_logging(log_level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Configures basic logging, optionally to a file."""
    global logger
    fmt = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s"
    dfmt = "%Y-%m-%d %H:%M:%S"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Use 'w' mode to create a fresh log for each eval script run
            # If running in batch mode, logs for all checkpoints go here.
            handlers.append(logging.FileHandler(log_path, mode='w'))
        except OSError as e:
            print(f"Warning: Could not create log file handler for {log_file}: {e}")

    # force=True ensures reconfiguration works if called multiple times
    logging.basicConfig(level=log_level, format=fmt, datefmt=dfmt, handlers=handlers, force=True)
    logger = logging.getLogger(__name__) # Get the logger for this module
    logger.info("Logging setup complete for evaluation script.")

def set_seed(seed_value: int) -> None:
    """Sets random seeds for reproducibility."""
    import random
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    if logger: logger.info(f"Set random seed: {seed_value}")
    else: print(f"Set random seed: {seed_value}")

def load_model_tokenizer_config(
    model_class: type,
    checkpoint_path: str,
    base_model_name: str = "gpt2",
    attn_implementation: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None
) -> Tuple[torch.nn.Module, PreTrainedTokenizer, PretrainedConfig]:
    """Loads model, tokenizer, and config from checkpoint for evaluation."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_dir():
        if logger: logger.error(f"Checkpoint directory not found: {ckpt_path}")
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    if logger: logger.info(f"Loading tokenizer and config from checkpoint: {ckpt_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=True)
        config = AutoConfig.from_pretrained(ckpt_path)
        if logger: logger.info("Loaded tokenizer and config from checkpoint directory.")
    except OSError:
        if logger: logger.warning(f"Tokenizer/config not found in {ckpt_path}. Falling back to: {base_model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
            config = AutoConfig.from_pretrained(base_model_name)
        except Exception as e:
            if logger: logger.error(f"Fallback loading failed for {base_model_name}: {e}", exc_info=True)
            raise

    # Handle pad token *before* loading model potentially
    new_tokens_added = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            if logger: logger.info(f"Set tokenizer pad_token to eos_token ('{tokenizer.eos_token}')")
        else:
            added = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            if added > 0:
                 if logger: logger.warning(f"Added '[PAD]' as pad_token to tokenizer ({added} new special token).")
                 new_tokens_added = True

    # --- Apply eval-specific loading args to config if provided ---
    # This allows overriding checkpoint config for evaluation if needed
    if attn_implementation:
         setattr(config, 'attn_implementation', attn_implementation)
         if logger: logger.info(f"Setting config attn_implementation='{attn_implementation}' for loading.")
    # dtype is handled by from_pretrained argument below

    if logger: logger.info(f"Loading model weights from checkpoint: {ckpt_path}")
    try:
        model_load_kwargs = {"config": config}
        if torch_dtype:
            model_load_kwargs["torch_dtype"] = torch_dtype
            if logger: logger.info(f"Requesting model dtype: {torch_dtype}")
        # Pass attn_implementation again here; from_pretrained priority might vary by HF version
        if attn_implementation:
             model_load_kwargs["attn_implementation"] = attn_implementation


        model = model_class.from_pretrained(ckpt_path, **model_load_kwargs)

        # Resize embeddings if a new pad token was added (essential!)
        if new_tokens_added:
            model.resize_token_embeddings(len(tokenizer))
            if logger: logger.info(f"Resized model embeddings to: {len(tokenizer)}")

        if logger: logger.info(f"Successfully loaded model '{model.__class__.__name__}' weights.")
        if logger:
             try:
                  impl = getattr(model.config, '_attn_implementation', 'default')
                  logger.info(f"Model running with attention implementation: {impl}")
             except AttributeError: pass

    except Exception as e:
        if logger: logger.error(f"Failed to load model weights from {ckpt_path}: {e}", exc_info=True)
        raise

    return model, tokenizer, config

def load_training_state(state_path: str, device: torch.device) -> Optional[Dict[str, Any]]:
    """Loads the training state dict from a .pt file."""
    state_file = Path(state_path)
    if not state_file.is_file():
        if logger: logger.warning(f"Training state file not found: {state_file}. Cannot analyze optimizer state.")
        return None
    try:
        if logger: logger.info(f"Loading training state from: {state_file}")
        # Load onto CPU first for analysis, avoids putting large optimizer state on GPU
        state = torch.load(state_file, map_location='cpu', weights_only=False)
        if logger: logger.info(f"Successfully loaded training state. Keys: {list(state.keys())}")
        return state
    except Exception as e:
        if logger: logger.error(f"Failed to load training state from {state_file}: {e}", exc_info=True)
        return None

def analyze_training_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyzes optimizer and scaler state, returning key statistics."""
    stats: Dict[str, Any] = {"optimizer_stats": {}, "scaler_stats": {}, "other_saved_stats": {}}
    if not state:
        return {"error": "Training state not provided."}

    # --- Optimizer Analysis ---
    if "optimizer" in state:
        opt_state = state["optimizer"]
        opt_stats = {}
        try:
            # Get LR/WD from the first parameter group
            if "param_groups" in opt_state and opt_state["param_groups"]:
                opt_stats["learning_rate"] = opt_state["param_groups"][0].get("lr", float('nan'))
                opt_stats["weight_decay"] = opt_state["param_groups"][0].get("weight_decay", float('nan'))
            else:
                 opt_stats["param_groups"] = "Not found"

            # Analyze AdamW states (exp_avg, exp_avg_sq) if present
            if "state" in opt_state and opt_state["state"]:
                all_exp_avg = []
                all_exp_avg_sq = []
                num_params_in_opt_state = 0
                for param_id, param_state in opt_state["state"].items():
                    num_params_in_opt_state += 1
                    # Ensure tensors are floats for analysis, handle potential different dtypes
                    if "exp_avg" in param_state:
                        all_exp_avg.append(param_state["exp_avg"].detach().float().flatten())
                    if "exp_avg_sq" in param_state:
                        all_exp_avg_sq.append(param_state["exp_avg_sq"].detach().float().flatten())

                opt_stats["params_in_state"] = num_params_in_opt_state

                # Calculate global stats
                if all_exp_avg:
                    full_exp_avg = torch.cat(all_exp_avg)
                    opt_stats["exp_avg_l2_norm"] = torch.linalg.norm(full_exp_avg).item()
                    opt_stats["exp_avg_mean"] = torch.mean(full_exp_avg).item()
                    opt_stats["exp_avg_std"] = torch.std(full_exp_avg).item()
                    del full_exp_avg
                if all_exp_avg_sq:
                    full_exp_avg_sq = torch.cat(all_exp_avg_sq)
                    opt_stats["exp_avg_sq_l2_norm"] = torch.linalg.norm(full_exp_avg_sq).item()
                    opt_stats["exp_avg_sq_mean"] = torch.mean(full_exp_avg_sq).item()
                    opt_stats["exp_avg_sq_std"] = torch.std(full_exp_avg_sq).item()
                    del full_exp_avg_sq

                del all_exp_avg, all_exp_avg_sq
            else:
                opt_stats["adam_state"] = "Not found or empty"

            stats["optimizer_stats"] = opt_stats

        except Exception as e:
            if logger: logger.warning(f"Could not fully analyze optimizer state: {e}", exc_info=True)
            stats["optimizer_stats"]["error"] = str(e)
    else:
        stats["optimizer_stats"] = "Not found in training state"

    # --- Scaler Analysis ---
    if "scaler" in state and state["scaler"]:
        scaler_state = state["scaler"]
        scaler_stats = {}
        try:
            # Common keys for GradScaler state dict
            scaler_stats["scale"] = scaler_state.get("_scale", scaler_state.get("scale", float('nan')))
            scaler_stats["growth_factor"] = scaler_state.get("_growth_factor", scaler_state.get("growth_factor", float('nan')))
            scaler_stats["backoff_factor"] = scaler_state.get("_backoff_factor", scaler_state.get("backoff_factor", float('nan')))
            scaler_stats["growth_interval"] = scaler_state.get("_growth_interval", scaler_state.get("growth_interval", float('nan')))
            # Found inf summary (key might vary across torch versions)
            inf_check = scaler_state.get("_found_inf_per_device", scaler_state.get("found_inf", None))
            if inf_check is not None:
                 # Try to convert to basic types for JSON
                 try: scaler_stats["found_inf_summary"] = str(inf_check.item()) if hasattr(inf_check, 'item') else str(inf_check)
                 except: scaler_stats["found_inf_summary"] = "Could not serialize"

            stats["scaler_stats"] = scaler_stats

        except Exception as e:
             if logger: logger.warning(f"Could not fully analyze scaler state: {e}", exc_info=True)
             stats["scaler_stats"]["error"] = str(e)
    else:
        stats["scaler_stats"] = "Not found or disabled in training state"

    # --- Other Saved State ---
    other_stats = {}
    if "global_step" in state:
        other_stats["global_step_saved"] = state["global_step"]
    if "epoch" in state:
        other_stats["epoch_saved"] = state["epoch"]
    if "args" in state:
         saved_args = state["args"]
         if isinstance(saved_args, argparse.Namespace): saved_args = vars(saved_args)
         # Convert Path objects and non-serializable items to strings
         other_stats["training_args"] = {k: str(v) if isinstance(v, Path) else v for k, v in saved_args.items()}
         # Quick check for basic types for JSON safety
         for k, v in other_stats["training_args"].items():
             if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                  other_stats["training_args"][k] = f"<Non-serializable: {type(v).__name__}>"

    stats["other_saved_stats"] = other_stats

    gc.collect()
    return stats

# --- Evaluation Functions (evaluate_standard, run_priming_evaluation_on_directory) ---
# (Copied from previous version, but added GPU memory logging to evaluate_standard
# and ensured priming CSV filename includes step number)

def evaluate_standard(args: argparse.Namespace, model: torch.nn.Module, eval_dataloader: DataLoader, device: torch.device) -> Dict[str, Any]:
    """Runs standard evaluation (perplexity)."""
    if eval_dataloader is None:
        if logger: logger.warning("Standard evaluation dataloader is None. Skipping standard eval.")
        return {"error": "Dataloader is None"}

    original_mode = model.training
    model.eval()
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)

    total_loss = 0.0
    total_items = 0
    oom_count = 0

    if logger: logger.info("Starting standard evaluation...")
    progress_bar = tqdm(eval_dataloader, desc="Eval (Std)", leave=False, disable=(logger is None or logger.getEffectiveLevel() > logging.INFO))

    with torch.no_grad():
        for batch in progress_bar:
            try:
                batch_on_device = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    oom_count += 1
                    if logger: logger.error(f"OOM error moving batch to {device}. Skipping. (Count: {oom_count})")
                    gc.collect(); torch.cuda.empty_cache()
                    if oom_count > 5:
                        if logger: logger.error("Too many OOM errors, aborting standard eval loop.")
                        progress_bar.close()
                        return {"error": "Too many OOM errors during data transfer."}
                    continue
                else:
                    if logger: logger.error(f"Error moving batch to {device}: {e}")
                    continue
            except Exception as e:
                 if logger: logger.error(f"Unexpected error moving batch to {device}: {e}")
                 continue

            try:
                amp_enabled = args.use_amp and device.type == 'cuda'
                amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
                with autocast(enabled=amp_enabled, dtype=amp_dtype if amp_enabled else None):
                    outputs = model(**batch_on_device)
                    loss = outputs.loss
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    oom_count += 1
                    if logger: logger.error(f"OOM error during forward pass. Skipping. (Count: {oom_count})")
                    gc.collect(); torch.cuda.empty_cache()
                    if oom_count > 5:
                        if logger: logger.error("Too many OOM errors, aborting standard eval loop.")
                        progress_bar.close()
                        return {"error": "Too many OOM errors during forward pass."}
                    continue
                else:
                    if logger: logger.error(f"Runtime error during forward pass: {e}", exc_info=True)
                    continue
            except Exception as e:
                if logger: logger.error(f"Unexpected error during forward pass: {e}", exc_info=True)
                continue

            if loss is not None and torch.isfinite(loss):
                num_items_in_batch = batch_on_device['input_ids'].size(0)
                total_loss += loss.detach().item() * num_items_in_batch
                total_items += num_items_in_batch
                progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
            else:
                if logger: logger.warning(f"Non-finite loss detected: {loss.item() if loss is not None else 'None'}. Skipping.")

    progress_bar.close()

    # Calculate final metrics
    metrics = {}
    if total_items > 0:
        final_avg_loss = total_loss / total_items
        try:
            perplexity = math.exp(min(final_avg_loss, 700))
        except (OverflowError, ValueError):
            perplexity = float('inf') if final_avg_loss > 0 else float('nan')

        if logger: logger.info(f"Standard Evaluation Results: Average Loss = {final_avg_loss:.4f}, Perplexity = {perplexity:.4f}, Total Items = {total_items}, OOM Count = {oom_count}")
        metrics = {"loss": final_avg_loss, "perplexity": perplexity, "total_items": total_items, "oom_count": oom_count}
    else:
        if logger: logger.warning("Standard evaluation completed, but total_items processed is zero.")
        metrics = {"loss": float('nan'), "perplexity": float('nan'), "total_items": 0, "oom_count": oom_count, "error": "No items processed"}

    # Log peak memory usage
    if device.type == 'cuda':
         peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
         if logger: logger.info(f"Peak GPU memory allocated during standard eval: {peak_mem_gb:.3f} GB")
         metrics["peak_gpu_mem_gb"] = peak_mem_gb
         torch.cuda.reset_peak_memory_stats(device) # Reset for next potential stage

    if original_mode: model.train()
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return metrics

def run_priming_evaluation_on_directory(
    eval_args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    run: Optional[NeptuneRun],
    checkpoint_step: int
) -> Dict[str, Any]:
    """Finds CSVs, creates dataloaders, runs eval, aggregates results."""
    if not PRIMING_EVAL_AVAILABLE:
         if logger: logger.error("Priming evaluation libraries not available. Skipping priming eval.")
         return {"error": "Priming library import failed."}

    if not eval_args.run_priming_eval or not eval_args.priming_eval_dir_path:
        if logger: logger.info("Skipping priming evaluation (not enabled or path not provided).")
        return {}

    priming_dir = Path(eval_args.priming_eval_dir_path)
    if not priming_dir.is_dir():
        if logger: logger.error(f"Priming evaluation directory not found: {priming_dir}")
        return {"error": f"Priming directory not found: {priming_dir}"}

    # Output dir is specific to this checkpoint step
    eval_output_dir = Path(eval_args.output_dir) / f"checkpoint-{checkpoint_step}"
    try:
        eval_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        if logger: logger.error(f"Failed to create priming output dir {eval_output_dir}: {e}. Cannot save priming CSV.")
        return {"error": f"Failed to create output directory {eval_output_dir}"}

    csv_files = sorted(list(priming_dir.glob('*.csv')))
    if not csv_files:
        if logger: logger.warning(f"No *.csv files found in priming directory: {priming_dir}")
        return {}

    # --- CSV Setup ---
    # Save CSV within the step-specific evaluation output directory
    csv_output_path = eval_output_dir / f"priming_results_raw_step_{checkpoint_step}.csv"
    if logger: logger.info(f"Raw priming results CSV will be saved to: {csv_output_path}")

    all_priming_summary_results = {}
    neptune_logs_for_this_step = {}

    if logger: logger.info(f"Found {len(csv_files)} CSVs for priming eval in {priming_dir}.")
    original_mode = model.training
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device) # Reset memory counter

    # --- CSV File Handling ---
    csv_file_handle = None
    csv_writer = None
    try:
        # Use 'w' mode to create fresh for this step's eval
        csv_file_handle = open(csv_output_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file_handle)
        header = ["eval_step", "corpus_file", "target_structure", "item_index", "pe", "logp_con", "logp_incon"]
        csv_writer.writerow(header)
        # if logger: logger.info(f"Opened and wrote header to priming results CSV: {csv_output_path}") # Verbose
    except IOError as e:
        if logger: logger.error(f"Failed to open or write header to CSV {csv_output_path}: {e}")
        if csv_file_handle: csv_file_handle.close()
        return {"error": f"Failed to open priming CSV {csv_output_path}"}


    # --- Process each CSV file ---
    model.eval() # Ensure model is in eval mode
    oom_count = 0
    total_priming_items = 0
    start_time_priming = time.time()

    for csv_path in csv_files:
        csv_filename = csv_path.name
        if logger: logger.info(f"--- Running Priming Eval for: {csv_filename} ---")

        priming_dataloader_single = None
        try:
            priming_dataloader_single = create_priming_dataloader(
                csv_path=str(csv_path), tokenizer=tokenizer,
                batch_size=eval_args.priming_per_device_eval_batch_size,
                delimiter=eval_args.priming_delimiter, num_workers=eval_args.num_workers,
                pin_memory=True, max_samples=eval_args.priming_eval_max_samples_per_file,
                seed=eval_args.seed
            )
        except Exception as e:
            if logger: logger.error(f"Dataloader creation failed for {csv_filename}: {e}", exc_info=True)
            all_priming_summary_results[csv_filename] = {"error": f"Dataloader creation failed: {e}"}
            continue

        if priming_dataloader_single is None or len(priming_dataloader_single.dataset) == 0:
             if logger: logger.warning(f"Dataloader for {csv_filename} is None or empty. Skipping.")
             all_priming_summary_results[csv_filename] = {"error": "Dataloader None or empty."}
             continue

        # --- Run the evaluation for this file ---
        try:
            # Assume run_native_priming_eval handles its own AMP context if needed internally
            priming_summary_metrics, priming_raw_results = run_native_priming_eval(
                model=model, priming_dataloader=priming_dataloader_single, device=device,
                tokenizer=tokenizer, use_amp=eval_args.use_amp
            )

            all_priming_summary_results[csv_filename] = priming_summary_metrics
            if logger: logger.info(f"Priming Summary ({csv_filename}): {priming_summary_metrics}")
            total_priming_items += priming_summary_metrics.get('num_items', 0)

            # Aggregate metrics for Neptune logging
            if run:
                 log_prefix = f"eval/priming/{csv_filename.replace('.', '_').replace('/','_')}"
                 metrics_to_log = {k: v for k, v in priming_summary_metrics.items() if isinstance(v, (int, float)) and math.isfinite(v)}
                 if metrics_to_log:
                     for k, v in metrics_to_log.items():
                         neptune_logs_for_this_step[f"{log_prefix}/{k}"] = v

            # Write Raw Results to CSV
            if csv_writer and priming_raw_results:
                items_written_count = 0
                try:
                    for target_structure, results_list in priming_raw_results.items():
                        for idx, item_data in enumerate(results_list):
                            if isinstance(item_data, dict):
                                pe = item_data.get('pe', float('nan'))
                                logp_con = item_data.get('logp_con', float('nan'))
                                logp_incon = item_data.get('logp_incon', float('nan'))
                                row = [checkpoint_step, csv_filename, target_structure, idx,
                                       f"{pe:.6f}", f"{logp_con:.6f}", f"{logp_incon:.6f}"]
                                csv_writer.writerow(row)
                                items_written_count += 1
                    # if logger: logger.debug(f"Wrote {items_written_count} raw rows for {csv_filename}.") # Debug
                except Exception as e:
                     if logger: logger.error(f"Error writing raw results to CSV for {csv_filename}: {e}", exc_info=True)

        except RuntimeError as e:
             if 'out of memory' in str(e).lower():
                 oom_count += 1
                 if logger: logger.error(f"OOM error during priming eval for {csv_filename}. Skipping file. (Count: {oom_count})")
                 gc.collect(); torch.cuda.empty_cache()
                 all_priming_summary_results[csv_filename] = {"error": "OOM during evaluation"}
                 if oom_count > 5:
                      if logger: logger.error("Too many OOM errors, aborting priming eval.")
                      if csv_file_handle: csv_file_handle.close() # Close file before returning
                      return {"error": "Too many OOM errors during priming evaluation."}
                 continue # Skip to next file
             else:
                 if logger: logger.error(f"Priming eval run failed for {csv_filename}: {e}", exc_info=True)
                 all_priming_summary_results[csv_filename] = {"error": f"Evaluation run failed: {e}"}
        except Exception as e:
            if logger: logger.error(f"Priming eval run failed for {csv_filename}: {e}", exc_info=True)
            all_priming_summary_results[csv_filename] = {"error": f"Evaluation run failed: {e}"}
        finally:
            del priming_dataloader_single # Explicitly delete dataloader
            gc.collect()

    # --- Finalize after loop ---
    if csv_file_handle: csv_file_handle.close()

    priming_duration = time.time() - start_time_priming
    if logger: logger.info(f"Finished priming evaluation loop. Duration: {priming_duration:.2f}s. Total Items: {total_priming_items}. OOM Count: {oom_count}")

    # Log aggregated metrics to Neptune
    if run and neptune_logs_for_this_step:
        if logger: logger.info(f"Logging {len(neptune_logs_for_this_step)} aggregated priming summary metrics to Neptune for step {checkpoint_step}...")
        try:
            for metric_path, value in neptune_logs_for_this_step.items():
                 run[metric_path].append(value, step=checkpoint_step)
            # Log overall stats
            run[f"evaluation/step_{checkpoint_step}/priming_total_items"].append(total_priming_items, step=checkpoint_step)
            run[f"evaluation/step_{checkpoint_step}/priming_duration_seconds"].append(priming_duration, step=checkpoint_step)
            run[f"evaluation/step_{checkpoint_step}/priming_oom_count"].append(oom_count, step=checkpoint_step)
            if logger: logger.info(f"Finished logging aggregated metrics to Neptune for step {checkpoint_step}.")
        except Exception as e:
             if logger: logger.warning(f"Neptune logging failed for priming summary at step {checkpoint_step}: {e}")
    elif run:
        if logger: logger.info(f"No priming summary metrics were aggregated for Neptune logging at step {checkpoint_step}.")

    # Add overall stats to returned dict
    all_priming_summary_results["_global_summary_"] = {
         "total_items": total_priming_items,
         "duration_seconds": priming_duration,
         "oom_count": oom_count
    }

    # Log peak memory usage
    if device.type == 'cuda':
         peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
         if logger: logger.info(f"Peak GPU memory allocated during priming eval: {peak_mem_gb:.3f} GB")
         all_priming_summary_results["_global_summary_"]["peak_gpu_mem_gb"] = peak_mem_gb
         torch.cuda.reset_peak_memory_stats(device)

    if original_mode: model.train()
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if logger: logger.info(f"--- Finished All Priming Evaluations for Checkpoint Step {checkpoint_step} ---")

    return all_priming_summary_results


# --- Argument Parser ---

def parse_eval_args() -> argparse.Namespace:
    """Parses command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate trained model checkpoints.")

    # === Path Arguments (Either single checkpoint or directory) ===
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a specific checkpoint directory to evaluate.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to a directory containing multiple checkpoint-* subdirs to evaluate.")
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory to save evaluation results (JSON summaries, priming CSVs).")
    parser.add_argument("--training_state_filename", type=str, default="training_state.pt", help="Filename of the training state file within checkpoint dirs.")

    # === Evaluation Control ===
    parser.add_argument("--run_standard_eval", action="store_true", default=False, help="Run standard perplexity evaluation.")
    parser.add_argument("--run_priming_eval", action="store_true", default=False, help="Run priming evaluation from directory.")
    parser.add_argument("--analyze_training_state", action="store_true", default=False, help="Load and analyze training_state.pt for optimizer/scaler stats.")

    # === Dataset Paths ===
    parser.add_argument("--validation_dataset_path", type=str, default=None, help="Path to validation Arrow dataset (needed for --run_standard_eval).")
    parser.add_argument("--priming_eval_dir_path", type=str, default=None, help="Directory containing priming CSVs (needed for --run_priming_eval).")

    # === Evaluation Parameters ===
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Standard eval batch size per device.")
    parser.add_argument("--priming_per_device_eval_batch_size", type=int, default=None, help="Priming eval batch size. Defaults to --per_device_eval_batch_size.")
    parser.add_argument("--eval_max_samples", type=int, default=50000, help="Max samples for standard eval. <= 0 uses full dataset.")
    parser.add_argument("--priming_eval_max_samples_per_file", type=int, default=1000, help="Max samples per priming CSV. <= 0 uses all.")
    parser.add_argument("--priming_delimiter", type=str, default=".", help="Delimiter in priming CSVs.")

    # === Model & Hardware ===
    parser.add_argument("--model_class_name", type=str, default="GPT2LMHeadModel", help="Name of the Hugging Face model class.")
    parser.add_argument("--base_model_name", type=str, default="gpt2", help="Base model identifier for fallback loading.")
    parser.add_argument("--use_amp", action="store_true", help="Enable AMP autocast for evaluation forward pass.")
    # Added args to control eval model loading
    parser.add_argument("--eval_use_flash_attention_2", action="store_true", help="Attempt to load model using Flash Attention 2 for evaluation.")
    parser.add_argument("--eval_torch_dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"], help="Torch dtype for model loading (auto detects based on hardware/AMP/Flash).")

    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # === Neptune Logging ===
    parser.add_argument("--neptune_project", type=str, default=None, help="Neptune project name (e.g., 'user/project').")
    parser.add_argument("--neptune_run_id", type=str, default=None, help="Existing Neptune run ID to log to. If None and project is set, creates a new run for the evaluation batch.")
    parser.add_argument("--neptune_api_token", type=str, default=None, help="Neptune API token (or use env var NEPTUNE_API_TOKEN).")
    parser.add_argument("--neptune_tags", type=str, nargs='+', default=None, help="Optional Neptune tags for new runs.")

    args = parser.parse_args()

    # --- Validation ---
    if not args.checkpoint_path and not args.checkpoint_dir:
        parser.error("Either --checkpoint_path (for single evaluation) or --checkpoint_dir (for batch evaluation) must be provided.")
    if args.checkpoint_path and args.checkpoint_dir:
        parser.error("Provide either --checkpoint_path OR --checkpoint_dir, not both.")
    if args.checkpoint_path and not Path(args.checkpoint_path).is_dir():
         parser.error(f"Specified checkpoint path not found or not a directory: {args.checkpoint_path}")
    if args.checkpoint_dir and not Path(args.checkpoint_dir).is_dir():
         parser.error(f"Specified checkpoint directory not found: {args.checkpoint_dir}")

    # Validate dataset paths based on requested evaluations
    if args.run_standard_eval and not args.validation_dataset_path:
        parser.error("--validation_dataset_path is required when --run_standard_eval is set.")
    if args.run_priming_eval and not args.priming_eval_dir_path:
        parser.error("--priming_eval_dir_path is required when --run_priming_eval is set.")
    if args.validation_dataset_path and not Path(args.validation_dataset_path).is_dir():
         parser.error(f"Validation dataset directory not found: {args.validation_dataset_path}")
    if args.priming_eval_dir_path and not Path(args.priming_eval_dir_path).is_dir():
         parser.error(f"Priming evaluation directory not found: {args.priming_eval_dir_path}")

    # Ensure output directory exists
    try:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        parser.error(f"Failed to create output directory {args.output_dir}: {e}")

    # Set priming batch size default
    if args.priming_per_device_eval_batch_size is None:
        args.priming_per_device_eval_batch_size = args.per_device_eval_batch_size

    return args

# --- Neptune Run Manager ---
class NeptuneRunManager:
    """Handles Neptune run initialization and stopping."""
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run: Optional[NeptuneRun] = None
        self._initialize()

    def _initialize(self):
        if not NEPTUNE_AVAILABLE or not self.args.neptune_project:
            if logger: logger.info("Neptune logging disabled (library not available or project not set).")
            return

        api_token = self.args.neptune_api_token or os.getenv('NEPTUNE_API_TOKEN')
        if not api_token:
            if logger: logger.warning("Neptune project specified, but no API token found. Neptune disabled.")
            return

        try:
            neptune_mode = "async"
            # In batch mode, we usually want one run for the whole batch.
            # If run_id is provided, use it. Otherwise, create one new run.
            if self.args.neptune_run_id:
                if logger: logger.info(f"Connecting to existing Neptune run: {self.args.neptune_run_id}")
                self.run = neptune.init_run(
                    project=self.args.neptune_project, api_token=api_token,
                    with_id=self.args.neptune_run_id, mode=neptune_mode
                )
            else:
                if logger: logger.info(f"Creating NEW Neptune run for evaluation batch...")
                default_tags = ['evaluation']
                # Add mode tag (single/batch)
                mode_tag = 'batch_eval' if self.args.checkpoint_dir else 'single_eval'
                default_tags.append(mode_tag)
                if self.args.neptune_tags: default_tags.extend(self.args.neptune_tags)

                self.run = neptune.init_run(
                    project=self.args.neptune_project, api_token=api_token,
                    name=f"Eval_{mode_tag}_{Path(self.args.output_dir).name}", # Generic name for batch
                    tags=default_tags, mode=neptune_mode
                )
                # Log evaluation parameters once for the new run
                self.run["evaluation/batch_args"] = vars(self.args)

            if logger: logger.info(f"Neptune initialized. Run URL: {self.run.get_url()}")

        except Exception as e:
            if logger: logger.error(f"Neptune initialization failed: {e}. Neptune logging disabled.", exc_info=True)
            self.run = None

    def get_run(self) -> Optional[NeptuneRun]:
        return self.run

    def stop(self):
        if self.run:
            try:
                self.run.stop()
                if logger: logger.info("Neptune run stopped.")
            except Exception as e:
                if logger: logger.error(f"Neptune stop failed: {e}")
            self.run = None

# --- Core Evaluation Logic for One Checkpoint ---

def run_evaluation_for_checkpoint(
    checkpoint_path: Union[str, Path],
    eval_args: argparse.Namespace,
    device: torch.device,
    run_manager: NeptuneRunManager
) -> Dict[str, Any]:
    """Loads models, runs evaluations, analyzes state for a single checkpoint."""
    ckpt_path = Path(checkpoint_path)
    results = {}
    model = None
    tokenizer = None
    config = None
    training_state = None
    run = run_manager.get_run() # Get Neptune run object

    # --- Determine Checkpoint Step ---
    try:
        checkpoint_step = int(ckpt_path.name.split('-')[-1])
    except (ValueError, IndexError):
        if logger: logger.warning(f"Could not determine step from path: {ckpt_path}. Using step -1.")
        checkpoint_step = -1

    if logger: logger.info(f"\n===== Evaluating Checkpoint Step: {checkpoint_step} ({ckpt_path.name}) =====")

    # --- Determine Model Loading Dtype ---
    torch_dtype = None
    if eval_args.eval_torch_dtype != "auto":
        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        torch_dtype = dtype_map.get(eval_args.eval_torch_dtype)
    elif eval_args.eval_use_flash_attention_2 or eval_args.use_amp:
        # Default to bf16 if available and using AMP/Flash, else fp16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            if logger: logger.info("Auto dtype: Using bfloat16 (AMP/Flash requested, supported).")
        else:
            torch_dtype = torch.float16
            if logger: logger.info("Auto dtype: Using float16 (AMP/Flash requested, bfloat16 not supported).")
    else:
         if logger: logger.info("Auto dtype: Using default (float32).")

    # --- Load Model ---
    model_load_start = time.time()
    try:
        model_class = GPT2LMHeadModel # TODO: Make dynamic if needed
        attn_impl = "flash_attention_2" if eval_args.eval_use_flash_attention_2 else None

        model, tokenizer, config = load_model_tokenizer_config(
            model_class, str(ckpt_path), eval_args.base_model_name,
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype
        )
        model.to(device)
        model.eval()
        param_count = sum(p.numel() for p in model.parameters())
        trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results["model_params"] = {"total": param_count, "trainable": trainable_param_count}
        if logger: logger.info(f"Model loaded to {device}. Params: {param_count:,} (Trainable: {trainable_param_count:,}). Time: {time.time() - model_load_start:.2f}s")
        # Save config used
        try: results["model_config"] = config.to_dict()
        except Exception: results["model_config"] = "Error serializing config"

        if device.type == 'cuda':
             peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
             results["model_params"]["peak_gpu_mem_load_gb"] = peak_mem_gb
             if logger: logger.info(f"Peak GPU memory after load: {peak_mem_gb:.3f} GB")
             torch.cuda.reset_peak_memory_stats(device) # Reset for eval stages


    except Exception as e:
        if logger: logger.critical(f"Fatal Error loading checkpoint {checkpoint_step}: {e}", exc_info=True)
        results["error"] = f"Model load failed: {e}"
        if run:
            try: run[f"evaluation/step_{checkpoint_step}/error"] = results["error"]
            except Exception: pass
        # Cleanup potentially partially loaded items
        del model, tokenizer, config
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return results # Return immediately on critical load error

    # --- Analyze Training State ---
    if eval_args.analyze_training_state:
        training_state_path = ckpt_path / eval_args.training_state_filename
        training_state = load_training_state(str(training_state_path), device='cpu') # Load to CPU
        if training_state:
            analysis_start = time.time()
            results["training_state_stats"] = analyze_training_state(training_state)
            if logger: logger.info(f"Training state analyzed. Time: {time.time() - analysis_start:.2f}s")
            # Log specific stats to Neptune if available
            if run:
                try:
                    opt_stats = results["training_state_stats"].get("optimizer_stats", {})
                    scaler_stats = results["training_state_stats"].get("scaler_stats", {})
                    other_stats = results["training_state_stats"].get("other_saved_stats", {})

                    ns_opt = f"evaluation/step_{checkpoint_step}/optimizer"
                    if "learning_rate" in opt_stats: run[f"{ns_opt}/learning_rate"].append(opt_stats["learning_rate"], step=checkpoint_step)
                    if "exp_avg_l2_norm" in opt_stats: run[f"{ns_opt}/exp_avg_norm"].append(opt_stats["exp_avg_l2_norm"], step=checkpoint_step)
                    if "exp_avg_sq_l2_norm" in opt_stats: run[f"{ns_opt}/exp_avg_sq_norm"].append(opt_stats["exp_avg_sq_l2_norm"], step=checkpoint_step)

                    ns_scaler = f"evaluation/step_{checkpoint_step}/scaler"
                    if "scale" in scaler_stats: run[f"{ns_scaler}/scale"].append(scaler_stats["scale"], step=checkpoint_step)

                    ns_other = f"evaluation/step_{checkpoint_step}/saved_state"
                    if "global_step_saved" in other_stats: run[f"{ns_other}/global_step"].append(other_stats["global_step_saved"], step=checkpoint_step)

                except Exception as e:
                    if logger: logger.warning(f"Neptune logging failed for training state stats (Step {checkpoint_step}): {e}")
        else:
            results["training_state_stats"] = {"error": "Failed to load or state file not found."}

    # --- Prepare Standard Eval Dataloader ---
    std_eval_dataloader = None
    if eval_args.run_standard_eval:
        if not eval_args.validation_dataset_path:
            if logger: logger.error("Standard eval requested but validation_dataset_path missing.")
            results["standard_summary"] = {"error": "validation_dataset_path missing"}
        else:
            data_load_start = time.time()
            try:
                if logger: logger.info(f"Loading validation data: {eval_args.validation_dataset_path}")
                ds = load_from_disk(eval_args.validation_dataset_path)
                original_size = len(ds)
                if logger: logger.info(f"Full Eval dataset size: {original_size:,} sequences")

                # Sampling Logic
                max_samples = eval_args.eval_max_samples
                if max_samples is not None and max_samples > 0 and max_samples < original_size:
                    if logger: logger.info(f"Sampling {max_samples:,} sequences (seed: {eval_args.seed}).")
                    rng = np.random.RandomState(eval_args.seed)
                    indices = rng.choice(original_size, size=max_samples, replace=False)
                    ds = ds.select(indices)
                    if logger: logger.info(f"Using subset for Eval: {len(ds):,} sequences")
                else:
                     if logger: logger.info("Using full validation set for standard eval.")

                sampler = SequentialSampler(ds)
                data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
                std_eval_dataloader = DataLoader(
                    ds, sampler=sampler, batch_size=eval_args.per_device_eval_batch_size,
                    num_workers=eval_args.num_workers, pin_memory=True, collate_fn=data_collator
                )
                if logger: logger.info(f"Standard Eval DataLoader created. Time: {time.time() - data_load_start:.2f}s")
            except Exception as e:
                if logger: logger.error(f"Failed to load/prepare standard evaluation data: {e}", exc_info=True)
                results["standard_summary"] = {"error": f"Dataloader creation failed: {e}"}
                std_eval_dataloader = None


    # --- Run Evaluations ---
    eval_stage_start = time.time()

    # Standard Evaluation
    if eval_args.run_standard_eval:
        if std_eval_dataloader:
            std_metrics = evaluate_standard(eval_args, model, std_eval_dataloader, device)
            results["standard_summary"] = std_metrics
            if run and std_metrics and "error" not in std_metrics:
                try:
                    loss = std_metrics.get("loss", float('nan'))
                    ppl = std_metrics.get("perplexity", float('nan'))
                    peak_mem = std_metrics.get("peak_gpu_mem_gb", float('nan'))
                    step = checkpoint_step # Use determined step
                    if math.isfinite(loss): run[f"evaluation/step_{step}/standard_loss"].append(loss, step=step)
                    if math.isfinite(ppl): run[f"evaluation/step_{step}/perplexity"].append(ppl, step=step)
                    if math.isfinite(peak_mem): run[f"evaluation/step_{step}/standard_peak_gpu_gb"].append(peak_mem, step=step)
                    # if logger: logger.info(f"Logged standard eval metrics to Neptune for step {step}.") # Verbose
                except Exception as e:
                    if logger: logger.warning(f"Neptune standard eval log failed for step {checkpoint_step}: {e}")
        else:
             # Error already logged if dataloader failed
             if "standard_summary" not in results: # Add error if not already present
                  results["standard_summary"] = {"error": "Dataloader not available"}

    # Priming Evaluation
    if eval_args.run_priming_eval:
        prime_metrics = run_priming_evaluation_on_directory(
            eval_args=eval_args, model=model, tokenizer=tokenizer, device=device,
            run=run, checkpoint_step=checkpoint_step
        )
        results["priming_summary"] = prime_metrics
        # Note: Neptune logging for priming is handled inside the function

    eval_stage_duration = time.time() - eval_stage_start
    results["evaluation_duration_seconds"] = eval_stage_duration
    if logger: logger.info(f"Evaluation stages completed. Duration: {eval_stage_duration:.2f}s")
    if run:
         try: run[f"evaluation/step_{checkpoint_step}/eval_stages_duration_seconds"].append(eval_stage_duration, step=checkpoint_step)
         except Exception: pass

    # --- Save Checkpoint-Specific Summary ---
    # Create step-specific subdirectory in output_dir
    step_output_dir = Path(eval_args.output_dir) / f"checkpoint-{checkpoint_step}"
    try:
        step_output_dir.mkdir(parents=True, exist_ok=True)
        summary_file_path = step_output_dir / f"evaluation_summary_step_{checkpoint_step}.json"
        if logger: logger.info(f"--- Saving Evaluation Summary for Step {checkpoint_step} ---")
        if results:
            # if logger: logger.info(f"Summary Metrics: {results}") # Can be very verbose
            try:
                # Custom encoder to handle numpy types if necessary
                class NpEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer): return int(obj)
                        if isinstance(obj, np.floating): return float(obj)
                        if isinstance(obj, np.ndarray): return obj.tolist()
                        if isinstance(obj, (torch.Tensor)): return obj.tolist() # Handle tensors
                        return super(NpEncoder, self).default(obj)

                with open(summary_file_path, "w", encoding='utf-8') as f:
                    json.dump(results, f, indent=4, cls=NpEncoder)
                if logger: logger.info(f"Evaluation summary saved to: {summary_file_path}")
                # Optionally log summary dict to Neptune
                if run:
                     try: run[f"evaluation/step_{checkpoint_step}/summary_dict"] = results
                     except Exception as ne:
                          # Try logging a simplified version if the full dict fails
                          try:
                              simple_results = {"standard": results.get("standard_summary"), "priming": results.get("priming_summary",{}).get("_global_summary_")}
                              run[f"evaluation/step_{checkpoint_step}/summary_dict_simple"] = simple_results
                          except Exception:
                               if logger: logger.warning(f"Failed to log even simplified eval summary dict to Neptune (Step {checkpoint_step}): {ne}")

            except IOError as e:
                if logger: logger.error(f"Failed to save summary JSON for step {checkpoint_step}: {e}")
            except TypeError as e:
                if logger: logger.error(f"Failed to serialize summary to JSON for step {checkpoint_step}: {e}")
        else:
            if logger: logger.warning(f"No evaluation results generated for step {checkpoint_step}.")

    except OSError as e:
        if logger: logger.error(f"Failed to create output subdirectory for step {checkpoint_step}: {e}")


    # --- Cleanup for this checkpoint ---
    del model, tokenizer, config, training_state, std_eval_dataloader
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if logger: logger.info(f"Finished evaluation for step {checkpoint_step}. Cleared memory.")

    return results

# --- Main Execution ---

def main():
    eval_args = parse_eval_args()

    # Setup logging (main log file for the whole run)
    log_file_name = f"evaluate_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    if eval_args.checkpoint_path: # Single run
         log_file_name = f"evaluate_log_step_{Path(eval_args.checkpoint_path).name}.txt"
    log_file_path = Path(eval_args.output_dir) / log_file_name
    setup_logging(log_file=str(log_file_path))
    global logger
    logger = logging.getLogger(__name__)

    mode = "Batch" if eval_args.checkpoint_dir else "Single"
    logger.info(f"***** Starting Evaluation Script ({mode} Mode) *****")
    logger.info(f"Evaluation Arguments: {vars(eval_args)}")

    # Setup Device and Seed
    device = get_device()
    set_seed(eval_args.seed)

    # Initialize Neptune Run Manager
    neptune_manager = NeptuneRunManager(eval_args)

    # --- Determine Checkpoints to Evaluate ---
    checkpoint_paths_to_evaluate = []
    if eval_args.checkpoint_path:
        checkpoint_paths_to_evaluate.append(Path(eval_args.checkpoint_path))
        logger.info(f"Running evaluation for single checkpoint: {eval_args.checkpoint_path}")
    else: # Batch mode using checkpoint_dir
        base_ckpt_dir = Path(eval_args.checkpoint_dir)
        logger.info(f"Scanning for checkpoint directories in: {base_ckpt_dir}")
        found_checkpoints = list(base_ckpt_dir.glob("checkpoint-*"))
        # Filter out non-directories and sort numerically by step
        numeric_checkpoints = {}
        for ckpt in found_checkpoints:
            if ckpt.is_dir():
                match = re.search(r"checkpoint-(\d+)$", ckpt.name)
                if match:
                    step = int(match.group(1))
                    numeric_checkpoints[step] = ckpt
        if not numeric_checkpoints:
            logger.error(f"No valid 'checkpoint-STEP' directories found in {base_ckpt_dir}. Exiting.")
            sys.exit(1)

        sorted_steps = sorted(numeric_checkpoints.keys())
        checkpoint_paths_to_evaluate = [numeric_checkpoints[step] for step in sorted_steps]
        logger.info(f"Found {len(checkpoint_paths_to_evaluate)} checkpoints to evaluate:")
        for path in checkpoint_paths_to_evaluate: logger.info(f"  - {path.name}")

    # --- Run Evaluation Loop ---
    overall_start_time = time.time()
    all_batch_results = {}
    num_evaluated = 0
    num_failed = 0

    for i, ckpt_path in enumerate(checkpoint_paths_to_evaluate):
        logger.info(f"--- Starting evaluation for checkpoint {i+1}/{len(checkpoint_paths_to_evaluate)}: {ckpt_path.name} ---")
        try:
            ckpt_results = run_evaluation_for_checkpoint(
                checkpoint_path=ckpt_path,
                eval_args=eval_args,
                device=device,
                run_manager=neptune_manager
            )
            all_batch_results[ckpt_path.name] = ckpt_results
            if ckpt_results.get("error"):
                 num_failed += 1
            else:
                 num_evaluated += 1
        except Exception as e:
            logger.error(f"Unhandled critical error during evaluation of {ckpt_path.name}: {e}", exc_info=True)
            all_batch_results[ckpt_path.name] = {"critical_error": str(e)}
            num_failed += 1
            # Try to continue to the next checkpoint

        # Explicit cleanup between checkpoints in batch mode
        if len(checkpoint_paths_to_evaluate) > 1:
             logger.info(f"--- Cleaning up GPU memory after evaluating {ckpt_path.name} ---")
             gc.collect()
             if torch.cuda.is_available(): torch.cuda.empty_cache()


    overall_duration = time.time() - overall_start_time
    logger.info(f"\n===== Evaluation Run Summary =====")
    logger.info(f"Mode: {mode}")
    logger.info(f"Checkpoints Processed: {len(checkpoint_paths_to_evaluate)}")
    logger.info(f"Successfully Evaluated: {num_evaluated}")
    logger.info(f"Failed/Skipped: {num_failed}")
    logger.info(f"Total Duration: {overall_duration:.2f} seconds")
    logger.info(f"Results saved in: {eval_args.output_dir}")
    logger.info(f"Main log file: {log_file_path}")

    # Stop Neptune run at the very end
    neptune_manager.stop()

    logger.info(f"***** Evaluation Script Finished ({mode} Mode) *****")


if __name__ == "__main__":
    # Basic logging setup in case main fails early
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    try:
        from tqdm.auto import tqdm
    except ImportError:
        print("Warning: tqdm not installed. Progress bars will be disabled.")
        def tqdm(iterable, *args, **kwargs): return iterable

    main()
