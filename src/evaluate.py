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
tqdm_module = None  # Will be imported or polyfilled in main


# --- Helper Functions ---

def get_device():
    """Gets the appropriate device for PyTorch computations."""
    global logger
    dev_str = "cpu"
    dev_name = "CPU"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): # More robust MPS check
        dev_str = "mps"
        dev_name = "MPS"
    elif torch.cuda.is_available():
        dev_str = "cuda:0"
        try:
            dev_name = torch.cuda.get_device_name(0)
        except Exception:
            dev_name = "CUDA GPU"

    device = torch.device(dev_str)

    log_func = logger.info if logger else print
    log_func(f"Using device: {device} ({dev_name})")
    return device


def setup_logging(log_level_str="INFO", log_file=None):
    """Configures basic logging, optionally to a file."""
    global logger
    numeric_log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    fmt = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - [%(funcName)s] - %(message)s" # Added funcName
    dfmt = "%Y-%m-%d %H:%M:%S"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file, mode='a'))
        except OSError as e:
            print(f"Warning: Could not create log file handler for {log_file}: {e}")

    logging.basicConfig(level=numeric_log_level, format=fmt, datefmt=dfmt, handlers=handlers, force=True)
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete for evaluation script at level {log_level_str.upper()}.")
    # Set transformers library log level to WARNING to reduce noise, unless our level is DEBUG
    if numeric_log_level > logging.DEBUG:
        logging.getLogger("transformers").setLevel(logging.WARNING)


def set_seed(seed_value):
    """Sets random seeds."""
    import random
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    log_func = logger.info if logger else print
    log_func(f"Set random seed: {seed_value}")


def load_model_for_evaluation(model_class, checkpoint_path: str, base_model_name="gpt2"):
    """Loads model, tokenizer, and config from checkpoint for evaluation."""
    global logger
    ckpt_path = Path(checkpoint_path)
    log_func_info = logger.info if logger else lambda msg: print(f"INFO: {msg}")
    log_func_error = logger.error if logger else lambda msg: print(f"ERROR: {msg}")
    log_func_warning = logger.warning if logger else lambda msg: print(f"WARNING: {msg}")


    if not ckpt_path.is_dir():
        log_msg = f"Checkpoint directory not found: {ckpt_path}"
        log_func_error(log_msg)
        raise FileNotFoundError(log_msg)

    log_func_info(f"Loading tokenizer and config from checkpoint: {ckpt_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=True)
        config = AutoConfig.from_pretrained(ckpt_path)
        log_func_info(f"Loaded tokenizer and config successfully from checkpoint directory: {ckpt_path}")
    except OSError:
        log_func_warning(f"Tokenizer/config not found directly in checkpoint {ckpt_path}. "
                         f"Falling back to loading from base model name: {base_model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
            config = AutoConfig.from_pretrained(base_model_name)
        except Exception as e:
            log_func_error(f"Fallback loading of tokenizer/config from base '{base_model_name}' failed: {e}",
                           exc_info=True if logger else False)
            raise

    log_func_info(f"Loading model weights for '{config.model_type}' from checkpoint: {ckpt_path}")
    try:
        model = model_class.from_pretrained(ckpt_path, config=config)
        log_func_info("Successfully loaded model weights from checkpoint.")
    except Exception as e:
        log_func_error(f"Failed to load model weights from checkpoint {ckpt_path}: {e}",
                       exc_info=True if logger else False)
        raise

    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            log_func_info(f"Set tokenizer pad_token to its eos_token ('{tokenizer.eos_token}')")
        else:
            log_func_warning(
                "Tokenizer has no pad_token and no eos_token. This might cause issues with padding during collation.")
    return model, tokenizer, config


def evaluate_standard(args, model, eval_dataloader, device):
    """Runs standard evaluation (perplexity)."""
    global logger, tqdm_module
    if eval_dataloader is None:
        logger.warning("Standard evaluation dataloader is None. Skipping standard perplexity eval.")
        return {}

    original_mode = model.training
    model.eval()
    total_loss = 0.0
    total_items = 0

    logger.info("Starting standard perplexity evaluation...")
    progress_bar = tqdm_module(eval_dataloader, desc="Eval (Std Perplexity)", leave=False, disable=not sys.stdout.isatty())

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            try:
                batch_on_device = {k: v.to(device, non_blocking=True) for k, v in batch.items() if
                                   isinstance(v, torch.Tensor)}
            except RuntimeError as e:
                logger.error(f"Error moving standard eval batch {batch_idx} to device {device}: {e}")
                continue

            try:
                amp_enabled = args.use_amp and device.type == 'cuda'
                with autocast(enabled=amp_enabled):
                    outputs = model(**batch_on_device)
                    loss = outputs.loss
            except Exception as e:
                logger.error(f"Error during standard eval forward pass on batch {batch_idx}: {e}", exc_info=True)
                continue

            if loss is not None and torch.isfinite(loss):
                num_items_in_batch = batch_on_device['input_ids'].size(0)
                total_loss += loss.detach().item() * num_items_in_batch
                total_items += num_items_in_batch
                if hasattr(progress_bar, 'set_postfix'): # Check if set_postfix exists
                    progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
            else:
                logger.warning(
                    f"Non-finite loss detected in standard eval (batch {batch_idx}): {loss.item() if loss is not None else 'None'}. Skipping contribution.")

    metrics = {}
    if total_items > 0:
        final_avg_loss = total_loss / total_items
        try:
            perplexity = math.exp(min(final_avg_loss, 700)) if final_avg_loss > 0 else float('nan')
        except (OverflowError, ValueError):
            perplexity = float('inf') if final_avg_loss > 0 else float('nan')

        logger.info(
            f"Standard Evaluation Results: Average Loss = {final_avg_loss:.4f}, Perplexity = {perplexity:.4f}, Total Items Evaluated = {total_items}")
        metrics = {"loss": final_avg_loss, "perplexity": perplexity, "total_items": total_items}
    else:
        logger.warning("Standard evaluation completed, but total_items processed is zero. No metrics calculated.")
        metrics = {"loss": float('nan'), "perplexity": float('nan'), "total_items": 0}

    if original_mode: model.train()
    if hasattr(progress_bar, 'close'): progress_bar.close()
    return metrics


def run_priming_evaluation_on_directory(
        eval_args, model, tokenizer, device, neptune_run_obj, checkpoint_numeric_step
):
    """
    Finds CSVs, creates dataloaders, runs priming eval, aggregates summary results,
    appends raw per-item results to a persistent CSV, and logs to Neptune.
    """
    global logger, tqdm_module
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

    eval_specific_output_dir = Path(eval_args.output_dir)
    raw_priming_csv_path = eval_specific_output_dir / f"priming_results_raw_step_{checkpoint_numeric_step}.csv"
    logger.info(f"Raw priming results CSV will be written/overwritten at: {raw_priming_csv_path}")

    csv_files_in_priming_dir = sorted(list(priming_dir.glob('*.csv')))
    if not csv_files_in_priming_dir:
        logger.warning(f"No *.csv files found in priming directory: {priming_dir}")
        return {}

    all_corpus_summary_metrics = {}
    neptune_metrics_to_log_this_step = {}
    logger.info(f"Found {len(csv_files_in_priming_dir)} CSV files for priming evaluation in {priming_dir}.")

    original_model_training_mode = model.training
    model.eval()

    csv_file_handle, csv_writer_obj = None, None
    try:
        csv_file_handle = open(raw_priming_csv_path, 'w', newline='', encoding='utf-8')
        csv_writer_obj = csv.writer(csv_file_handle)
        csv_header = [
            "eval_step", "corpus_file", "target_structure", "item_index",
            "pe", "logp_con", "logp_incon", "logp_baseline",
            "logp_con_random_baseline", "logp_incon_random_baseline"
        ]
        csv_writer_obj.writerow(csv_header)
    except IOError as e:
        logger.error(f"Failed to open or write header to raw priming CSV {raw_priming_csv_path}: {e}")
        if csv_file_handle: csv_file_handle.close()
        return {"error": f"Raw priming CSV write error: {e}"}

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
                seed=eval_args.seed,
                max_length=eval_args.priming_max_seq_length # Passed max_length
            )
        except Exception as e:
            logger.error(f"Dataloader creation failed for priming corpus {corpus_filename}: {e}", exc_info=True)
            all_corpus_summary_metrics[corpus_filename] = {"error": f"Dataloader creation failed: {e}"}
            continue

        if current_corpus_dataloader is None or not hasattr(current_corpus_dataloader, 'dataset') or len(current_corpus_dataloader.dataset) == 0:
            logger.warning(f"Dataloader for priming corpus {corpus_filename} is None or empty. Skipping this corpus.")
            all_corpus_summary_metrics[corpus_filename] = {"error": "Dataloader was None or empty."}
            continue

        try:
            corpus_summary_metrics, corpus_raw_item_results = run_native_priming_eval(
                model=model, priming_dataloader=current_corpus_dataloader, device=device,
                tokenizer=tokenizer, use_amp=eval_args.use_amp, # tokenizer passed here
                random_seed=eval_args.seed
            )
            all_corpus_summary_metrics[corpus_filename] = corpus_summary_metrics
            logger.info(f"Priming Summary Metrics for {corpus_filename}: {corpus_summary_metrics}")

            if NEPTUNE_AVAILABLE and neptune_run_obj:
                sanitized_corpus_name = corpus_filename.replace('.', '_').replace('/', '_')
                neptune_log_prefix = f"eval_metrics/priming/{sanitized_corpus_name}"
                finite_numeric_metrics = {k: v for k, v in corpus_summary_metrics.items() if
                                          isinstance(v, (int, float)) and math.isfinite(v)}
                for metric_key, metric_value in finite_numeric_metrics.items():
                    neptune_metrics_to_log_this_step[f"{neptune_log_prefix}/{metric_key}"] = metric_value

            if csv_writer_obj and corpus_raw_item_results:
                raw_items_written_for_corpus = 0
                try:
                    for target_structure_key, per_structure_results_list in corpus_raw_item_results.items():
                        for item_idx, item_data_dict in enumerate(per_structure_results_list):
                            if isinstance(item_data_dict, dict):
                                pe_val = item_data_dict.get('pe', float('nan'))
                                logp_con_val = item_data_dict.get('logp_con', float('nan'))
                                logp_incon_val = item_data_dict.get('logp_incon', float('nan'))
                                logp_baseline_val = item_data_dict.get('logp_baseline', float('nan'))
                                logp_con_rb_val = item_data_dict.get('logp_con_random_baseline', float('nan'))
                                logp_incon_rb_val = item_data_dict.get('logp_incon_random_baseline', float('nan'))

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
                    if csv_file_handle: csv_file_handle.flush()
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
            gc.collect()

    if NEPTUNE_AVAILABLE and neptune_run_obj and neptune_metrics_to_log_this_step:
        logger.info(
            f"Logging {len(neptune_metrics_to_log_this_step)} aggregated priming metrics to Neptune for checkpoint step {checkpoint_numeric_step}...")
        try:
            for full_metric_path, value_to_log in neptune_metrics_to_log_this_step.items():
                neptune_run_obj[full_metric_path].append(value_to_log, step=checkpoint_numeric_step)
            logger.info(
                f"Successfully logged aggregated priming metrics to Neptune for checkpoint step {checkpoint_numeric_step}.")
        except Exception as e_neptune_log:
            logger.warning(
                f"Neptune logging for aggregated priming metrics failed (checkpoint step {checkpoint_numeric_step}): {e_neptune_log}")
    elif NEPTUNE_AVAILABLE and neptune_run_obj:
        logger.info(
            f"No valid priming summary metrics were aggregated for Neptune logging at checkpoint step {checkpoint_numeric_step}.")

    if csv_file_handle:
        try:
            csv_file_handle.close();
            logger.info(f"Closed raw priming results CSV file: {raw_priming_csv_path}")
        except Exception as e_close:
            logger.error(f"Error closing raw priming results CSV file {raw_priming_csv_path}: {e_close}")

    if original_model_training_mode: model.train()
    logger.info(f"--- Finished All Priming Evaluations for Checkpoint Step {checkpoint_numeric_step} ---")
    return all_corpus_summary_metrics


def parse_eval_args():
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
    parser.add_argument("--priming_max_seq_length", type=int, default=None,
                        help="Maximum sequence length for priming evaluation tokenization. If None, uses tokenizer.model_max_length.")
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
    parser.add_argument("--num_workers", type=int, default=0, # Changed default to 0 for easier debugging
                        help="Number of DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--neptune_project", type=str, default=None,
                        help="Neptune project name (e.g., 'your-workspace/your-project').")
    parser.add_argument("--neptune_run_id", type=str, default=None,
                        help="Existing Neptune run ID to log to.")
    parser.add_argument("--neptune_api_token", type=str, default=None,
                        help="Neptune API token (alternative to NEPTUNE_API_TOKEN env var).")
    parser.add_argument("--neptune_tags", type=str, nargs='+', default=None,
                        help="Optional Neptune tags for newly created evaluation runs.")
    parser.add_argument("--neptune_run_name", type=str, default=None,
                        help="Specific name for the Neptune run.")
    parser.add_argument("--checkpoint_label", type=str, default=None,
                        help="Descriptive label for the checkpoint (e.g., 'checkpoint-1000', 'final_model').")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level for the script.")


    args = parser.parse_args()
    if args.priming_per_device_eval_batch_size is None:
        args.priming_per_device_eval_batch_size = args.per_device_eval_batch_size
    if not Path(args.checkpoint_path).is_dir(): parser.error(f"Checkpoint directory not found: {args.checkpoint_path}")
    if args.run_standard_eval and not args.validation_dataset_path: parser.error("--validation_dataset_path is required for --run_standard_eval.")
    if args.run_priming_eval and not args.priming_eval_dir_path: parser.error("--priming_eval_dir_path is required for --run_priming_eval.")
    if args.validation_dataset_path and not Path(args.validation_dataset_path).is_dir(): parser.error(f"Validation dataset directory not found: {args.validation_dataset_path}")
    if args.priming_eval_dir_path and not Path(args.priming_eval_dir_path).is_dir(): parser.error(f"Priming evaluation directory not found: {args.priming_eval_dir_path}")
    try: Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    except OSError as e: parser.error(f"Failed to create output directory {args.output_dir}: {e}")

    if args.checkpoint_label:
        args.checkpoint_step_label = args.checkpoint_label
        try: args.checkpoint_numeric_step = int(args.checkpoint_label.split('-')[-1])
        except ValueError: args.checkpoint_numeric_step = -1
    else:
        args.checkpoint_step_label = Path(args.checkpoint_path).name
        try: args.checkpoint_numeric_step = int(args.checkpoint_step_label.split('-')[-1])
        except (ValueError, IndexError): args.checkpoint_numeric_step = -1
    return args


def main():
    global logger, tqdm_module
    try:
        from tqdm.auto import tqdm as imported_tqdm
        tqdm_module = imported_tqdm
    except ImportError:
        print("Warning: tqdm.auto not installed. Progress bars will be basic or disabled.", file=sys.stderr)
        def fallback_tqdm(iterable, *args, **kwargs):
            disable = kwargs.pop('disable', False)
            desc = kwargs.get('desc', 'items')
            if not disable and sys.stdout.isatty(): print(f"Processing {desc}...")
            # yield from iterable # for Python 3.3+
            for x in iterable: yield x # Compatible with older Pythons
        tqdm_module = fallback_tqdm

    eval_args = parse_eval_args()

    log_file_name = f"evaluate_log_{eval_args.checkpoint_step_label}.txt"
    log_file_path = Path(eval_args.output_dir) / log_file_name
    setup_logging(log_level_str=eval_args.log_level, log_file=str(log_file_path)) # Use arg for log level
    logger = logging.getLogger(__name__)

    logger.info(
        f"***** Starting Evaluation Script for Checkpoint: '{eval_args.checkpoint_step_label}' (Numeric Step: {eval_args.checkpoint_numeric_step}) *****")
    logger.debug(f"Full Evaluation Arguments: {vars(eval_args)}") # Log full args only at DEBUG

    device = get_device()
    set_seed(eval_args.seed)

    neptune_run_obj = None
    if NEPTUNE_AVAILABLE and eval_args.neptune_project:
        logger.info("Attempting to initialize Neptune for evaluation logging...")
        neptune_api_token_to_use = eval_args.neptune_api_token or os.getenv('NEPTUNE_API_TOKEN')
        if not neptune_api_token_to_use:
            logger.warning("Neptune project specified, but no API token found. Neptune logging will be disabled.")
        else:
            try:
                neptune_connection_mode = "async"
                env_linked_training_run_name = os.getenv('SINGULARITYENV_NEPTUNE_TRAINING_RUN_NAME') or os.getenv('NEPTUNE_TRAINING_RUN_NAME')
                current_neptune_tags = ['evaluation', eval_args.checkpoint_step_label]
                if eval_args.neptune_tags: current_neptune_tags.extend(eval_args.neptune_tags)
                if env_linked_training_run_name and f"train_ref:{env_linked_training_run_name}" not in current_neptune_tags:
                    current_neptune_tags.append(f"train_ref:{env_linked_training_run_name}")
                current_neptune_tags = sorted(list(set(current_neptune_tags)))

                if eval_args.neptune_run_id:
                    logger.info(f"Attempting to connect to existing Neptune run ID: {eval_args.neptune_run_id}")
                    neptune_run_obj = neptune.init_run(project=eval_args.neptune_project,
                                                       api_token=neptune_api_token_to_use,
                                                       with_id=eval_args.neptune_run_id, mode=neptune_connection_mode)
                    neptune_run_obj[f"evaluation_runs/{eval_args.checkpoint_step_label}/args"] = vars(eval_args)
                else:
                    run_name_for_neptune = eval_args.neptune_run_name
                    if not run_name_for_neptune:
                        shared_run_id_env = os.getenv('SHARED_RUN_ID', 'unknown_run')
                        run_name_for_neptune = f"Eval_{shared_run_id_env}_{eval_args.checkpoint_step_label}".strip('_')
                    logger.info(f"Creating NEW Neptune run with Name: '{run_name_for_neptune}', Tags: {current_neptune_tags}")
                    neptune_run_obj = neptune.init_run(project=eval_args.neptune_project,
                                                       api_token=neptune_api_token_to_use,
                                                       name=run_name_for_neptune, tags=current_neptune_tags,
                                                       mode=neptune_connection_mode)
                    neptune_run_obj["evaluation/script_args"] = vars(eval_args)
                    if env_linked_training_run_name:
                        neptune_run_obj["details/linked_training_run_name"] = env_linked_training_run_name
                if neptune_run_obj: logger.info(f"Neptune initialized. Run URL: {neptune_run_obj.get_url()}")
            except Exception as e_neptune_init:
                logger.error(f"Neptune initialization failed: {e_neptune_init}. Neptune logging disabled.", exc_info=True)
                neptune_run_obj = None
    else:
        logger.info("Neptune logging disabled.")

    logger.info(f"--- Loading Model and Tokenizer from Checkpoint: {eval_args.checkpoint_path} ---")
    try:
        model_class_name_map = {"GPT2LMHeadModel": GPT2LMHeadModel} # Add other models if needed
        model_constructor = model_class_name_map.get(eval_args.model_class_name, GPT2LMHeadModel)
        model, tokenizer, config = load_model_for_evaluation(
            model_constructor, eval_args.checkpoint_path, eval_args.base_model_name
        )
        model.to(device)
        model.eval()
        logger.info(f"Model '{eval_args.model_class_name}' loaded to device '{device}' and set to eval mode.")
    except Exception as e_model_load:
        logger.critical(f"FATAL: Failed to load model from '{eval_args.checkpoint_path}': {e_model_load}", exc_info=True)
        if NEPTUNE_AVAILABLE and neptune_run_obj:
            try:
                neptune_run_obj[f"evaluation_status/{eval_args.checkpoint_step_label}/critical_error"] = f"Model load failed: {traceback.format_exc()}"
                neptune_run_obj.stop()
            except Exception: pass
        sys.exit(1)

    std_eval_dataloader = None
    if eval_args.run_standard_eval:
        logger.info("--- Preparing Standard Evaluation (Perplexity) Dataloader ---")
        if not eval_args.validation_dataset_path:
            logger.error("--validation_dataset_path not provided for standard eval. Skipping.")
        else:
            try:
                logger.info(f"Loading validation dataset from: {eval_args.validation_dataset_path}")
                validation_hf_dataset = load_from_disk(eval_args.validation_dataset_path)
                full_dataset_size = len(validation_hf_dataset)
                logger.info(f"Full validation dataset size: {full_dataset_size:,} sequences.")

                if eval_args.eval_max_samples is not None and 0 < eval_args.eval_max_samples < full_dataset_size:
                    np_rng = np.random.RandomState(eval_args.seed)
                    sampled_indices = np_rng.choice(full_dataset_size, size=eval_args.eval_max_samples, replace=False)
                    validation_hf_dataset = validation_hf_dataset.select(sampled_indices)
                    logger.info(f"Using subset for perplexity eval: {len(validation_hf_dataset):,} samples (seed: {eval_args.seed}).")
                elif eval_args.eval_max_samples is not None and eval_args.eval_max_samples > 0:
                    logger.info(f"Using full validation set ({full_dataset_size:,} samples) as max_samples >= dataset size.")
                else:
                    logger.info(f"Using full validation set ({full_dataset_size:,} samples).")

                sequential_sampler = SequentialSampler(validation_hf_dataset)
                lm_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
                std_eval_dataloader = DataLoader(
                    validation_hf_dataset, sampler=sequential_sampler, batch_size=eval_args.per_device_eval_batch_size,
                    num_workers=eval_args.num_workers, pin_memory=True, collate_fn=lm_data_collator
                )
                logger.info("Standard Perplexity DataLoader prepared.")
            except Exception as e_std_data:
                logger.error(f"Failed to load/prepare standard eval data: {e_std_data}", exc_info=True)
                std_eval_dataloader = None
    else:
        logger.info("Standard perplexity evaluation skipped (--run_standard_eval not set).")

    all_evaluation_results = {
        "checkpoint_label": eval_args.checkpoint_step_label,
        "checkpoint_numeric_step": eval_args.checkpoint_numeric_step
    }
    overall_eval_start_time = time.time()

    if eval_args.run_standard_eval:
        if std_eval_dataloader:
            standard_eval_metrics = evaluate_standard(eval_args, model, std_eval_dataloader, device)
            all_evaluation_results["standard_perplexity_summary"] = standard_eval_metrics
            if NEPTUNE_AVAILABLE and neptune_run_obj and standard_eval_metrics:
                try:
                    loss_val = standard_eval_metrics.get("loss", float('nan'))
                    ppl_val = standard_eval_metrics.get("perplexity", float('nan'))
                    if math.isfinite(loss_val): neptune_run_obj[f"eval_metrics/standard_loss"].append(loss_val, step=eval_args.checkpoint_numeric_step)
                    if math.isfinite(ppl_val): neptune_run_obj[f"eval_metrics/perplexity"].append(ppl_val, step=eval_args.checkpoint_numeric_step)
                    logger.info(f"Logged standard metrics to Neptune (step {eval_args.checkpoint_numeric_step}).")
                except Exception as e_neptune_std_log:
                    logger.warning(f"Neptune logging for standard metrics failed (step {eval_args.checkpoint_numeric_step}): {e_neptune_std_log}")
        else:
            logger.warning("Standard perplexity eval requested but dataloader failed. Skipping.")
            all_evaluation_results["standard_perplexity_summary"] = {"error": "Dataloader failed."}

    if eval_args.run_priming_eval:
        priming_evaluation_summary_metrics = run_priming_evaluation_on_directory(
            eval_args, model, tokenizer, device, neptune_run_obj, eval_args.checkpoint_numeric_step
        )
        all_evaluation_results["priming_evaluation_summary"] = priming_evaluation_summary_metrics

    total_script_duration_seconds = time.time() - overall_eval_start_time
    logger.info(f"Total evaluation script time: {total_script_duration_seconds:.2f}s for '{eval_args.checkpoint_step_label}'.")
    all_evaluation_results["total_evaluation_duration_seconds"] = total_script_duration_seconds
    if NEPTUNE_AVAILABLE and neptune_run_obj:
        try:
            neptune_run_obj[f"evaluation_details/{eval_args.checkpoint_step_label}/duration_seconds"] = total_script_duration_seconds
        except Exception: pass

    json_summary_filename = f"evaluation_summary_{eval_args.checkpoint_step_label}.json"
    json_summary_filepath = Path(eval_args.output_dir) / json_summary_filename
    logger.info(f"--- Saving Comprehensive Evaluation Summary for '{eval_args.checkpoint_step_label}' ---")

    if len(all_evaluation_results) > 2:
        logger.info(f"Final Summary for '{eval_args.checkpoint_step_label}': {json.dumps(all_evaluation_results, indent=2)}")
        try:
            with open(json_summary_filepath, "w", encoding='utf-8') as f_summary:
                json.dump(all_evaluation_results, f_summary, indent=4)
            logger.info(f"Comprehensive summary saved to: {json_summary_filepath}")
            if NEPTUNE_AVAILABLE and neptune_run_obj:
                try:
                    neptune_run_obj[f"evaluation_details/{eval_args.checkpoint_step_label}/full_summary_dict"] = all_evaluation_results
                except Exception as ne_log_summary_dict:
                    logger.warning(f"Failed to log full summary to Neptune: {ne_log_summary_dict}")
        except Exception as e_save_json:
            logger.error(f"Failed to save summary JSON: {e_save_json}", exc_info=True)
    else:
        logger.warning(f"No actual results generated for '{eval_args.checkpoint_step_label}'. Summary JSON minimal.")

    if eval_args.run_priming_eval:
        priming_csv_path_final = Path(eval_args.output_dir) / f"priming_results_raw_step_{eval_args.checkpoint_numeric_step}.csv"
        logger.info(f"Raw priming results (if any) saved to: {priming_csv_path_final}")

    if NEPTUNE_AVAILABLE and neptune_run_obj:
        try:
            neptune_run_obj.sync()
            neptune_run_obj.stop()
            logger.info("Neptune run stopped.")
        except Exception as e_neptune_stop:
            logger.error(f"Neptune stop failed: {e_neptune_stop}")

    logger.info(f"***** Evaluation Script Finished for Checkpoint: '{eval_args.checkpoint_step_label}' *****")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s") # Fallback
    main()