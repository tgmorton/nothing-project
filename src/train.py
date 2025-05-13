# src/train.py

# === Imports ===
import logging
import argparse
from pathlib import Path
import csv
import os
import sys
import math
import torch
import random
import numpy as np
import json
import time
import traceback
import gc
import subprocess  # For running local eval

# --- ML/data library imports ---
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import (
    GPT2LMHeadModel,
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizer,
    get_scheduler,
    DataCollatorForLanguageModeling
)
# Use torch.amp directly for GradScaler and autocast
import torch.amp
from datasets import load_from_disk
from torch.nn.utils import clip_grad_norm_

# Optional Neptune import
try:
    import neptune

    NEPTUNE_AVAILABLE = True
except ImportError:
    print("Neptune.ai library not found, Neptune logging will be disabled.")
    sys.modules['neptune'] = None  # Ensure neptune is None if not imported
    NEPTUNE_AVAILABLE = False
    neptune = None

# --- Logger Setup ---
logger = None  # Will be initialized in setup_logging


# --- Helper Function Definitions ---

def parse_args():
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train a GPT-2 like model using preprocessed Arrow datasets.")

    # === Essential Paths ===
    parser.add_argument("--train_dataset_path", type=str, required=True, help="Path to the training Arrow dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for checkpoints, logs, final model.")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Base model identifier for tokenizer/config structure (e.g., 'gpt2').")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to RESUME training from.")

    # === Training Hyperparameters ===
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total training epochs (used if max_steps <= 0).")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum number of optimizer steps to train for. Overrides num_train_epochs if > 0.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Train batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Steps for gradient accumulation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Peak learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay coefficient.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="LR scheduler type.",
                        choices=["linear", "cosine", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="LR warmup steps.")
    parser.add_argument("--model_size", type=str, default="10m",
                        help="Model size tag for small model config (e.g., '10m' or '100m').")

    # === Hardware & Precision ===
    parser.add_argument("--use_amp", action="store_true", help="Enable AMP training.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")

    # === Control & Reproducibility ===
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # === Logging & Saving ===
    parser.add_argument("--logging_steps", type=int, default=100, help="Log train metrics every X steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps.")

    # === LOCAL Evaluation Triggering Control (these are used if --local_eval is set) ===
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Trigger local evaluation every X steps (if --local_eval).")
    parser.add_argument("--local_eval", action="store_true",
                        help="Run evaluation script locally as subprocess. If False, no evaluation is triggered by this script.")
    # --submit_eval_script_path is NO LONGER USED by train.py for Slurm submission
    # parser.add_argument("--submit_eval_script_path", type=str, default=None, help="Path to the Slurm script for submitting eval jobs (DEPRECATED in this script).")
    parser.add_argument("--evaluate_script_path", type=str, default="src/evaluate.py",
                        help="Path to evaluate.py (required if --local_eval).")
    parser.add_argument("--validation_dataset_path", type=str, default=None,
                        help="Path to validation Arrow dataset (passed to local eval script if standard eval enabled).")
    parser.add_argument("--priming_eval_dir_path", type=str, default=None,
                        help="Directory containing priming CSVs (passed to local eval script if priming eval enabled).")
    parser.add_argument("--trigger_standard_eval", action="store_true", default=False,
                        help="Trigger/run standard evaluation (if --local_eval).")
    parser.add_argument("--trigger_priming_eval", action="store_true", default=False,
                        help="Trigger/run priming evaluation (if --local_eval).")
    parser.add_argument("--priming_eval_steps", type=int, default=None,
                        help="Trigger/run priming eval every X steps (if --local_eval). Defaults to --eval_steps.")

    # === Neptune Logging ===
    parser.add_argument("--neptune_project", type=str, default=None, help="Neptune project name.")
    parser.add_argument("--neptune_tags", type=str, nargs='+', default=None, help="Optional Neptune tags.")
    parser.add_argument("--neptune_run_name", type=str, default=None, help="Optional Neptune run name.")

    args = parser.parse_args()

    # Set defaults
    if args.priming_eval_steps is None: args.priming_eval_steps = args.eval_steps

    # Validation for Training
    if not args.train_dataset_path: parser.error("--train_dataset_path required for training.")
    if not args.output_dir: parser.error("--output_dir required for training.")
    if not Path(args.train_dataset_path).is_dir(): parser.error(f"Train dataset not found: {args.train_dataset_path}")

    # Validation for LOCAL evaluation
    if args.local_eval:
        if not Path(args.evaluate_script_path).is_file():
            parser.error(f"--local_eval specified, but evaluation script not found: {args.evaluate_script_path}")
        if args.trigger_standard_eval and not args.validation_dataset_path:
            parser.error("--local_eval and --trigger_standard_eval specified, but --validation_dataset_path missing.")
        if args.trigger_priming_eval and not args.priming_eval_dir_path:
            parser.error("--local_eval and --trigger_priming_eval specified, but --priming_eval_dir_path missing.")
        if args.validation_dataset_path and not Path(args.validation_dataset_path).is_dir():
            parser.error(f"Local eval: Validation dataset not found: {args.validation_dataset_path}")
        if args.priming_eval_dir_path and not Path(args.priming_eval_dir_path).is_dir():
            parser.error(f"Local eval: Priming dir not found: {args.priming_eval_dir_path}")

    if not Path(args.output_dir).exists():
        print(
            f"Warning: Output directory {args.output_dir} does not exist. It will be created by the training script or orchestrator.")

    return args


# --- Keep Helper Functions Needed for Training ---
# get_device, setup_logging, set_seed, setup_distributed, save_checkpoint, create_small_model_config
# (Implementations are the same as previous version)
def get_device():
    """Gets the appropriate device for PyTorch computations."""
    import torch;
    import os
    if torch.backends.mps.is_available():
        device = torch.device("mps");
        print("Using MPS")
    elif torch.cuda.is_available():
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if local_rank >= torch.cuda.device_count(): local_rank = 0
            device = torch.device(f"cuda:{local_rank}");
            print(f"Using CUDA GPU: {local_rank} - {torch.cuda.get_device_name(device)}")
        except Exception as e:
            print(f"Error setting CUDA: {e}. Using CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu");
        print("Using CPU")
    return device


def setup_logging(log_level=logging.INFO, rank=0):
    """Configures basic logging."""
    global logger
    fmt = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s"
    dfmt = "%Y-%m-%d %H:%M:%S"
    level = log_level if rank == 0 else logging.CRITICAL + 1  # Only rank 0 logs INFO+
    logging.basicConfig(level=level, format=fmt, datefmt=dfmt, force=True)
    logger = logging.getLogger(__name__)  # Get logger for this module
    if rank == 0: logger.info("Logging setup complete (Rank 0).")
    # For other ranks, logger will be created but effectively silenced by level


def set_seed(seed_value):
    """Sets random seeds."""
    import random;
    import numpy as np;
    import torch;
    global logger
    random.seed(seed_value);
    np.random.seed(seed_value);
    torch.manual_seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)
    # Check if logger is initialized and not disabled for rank 0
    if logger and (not hasattr(logger, 'disabled') or not logger.disabled):
        logger.info(f"Set random seed: {seed_value}")
    else:  # Fallback for early calls or non-rank 0
        if int(os.environ.get("RANK", "0")) == 0: print(
            f"Set random seed: {seed_value} (logger N/A or disabled at call time)")


def setup_distributed(args):
    """Sets up DDP environment."""
    import torch;
    import os;
    global logger
    is_dist, rank, world_size, local_rank = False, 0, 1, 0
    if 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1:
        is_dist = True
        try:
            rank = int(os.environ["RANK"]);
            world_size = int(os.environ["WORLD_SIZE"]);
            local_rank = int(os.environ["LOCAL_RANK"])
            if not torch.cuda.is_available(): raise RuntimeError("DDP needs CUDA.")
            torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
            torch.cuda.set_device(local_rank)
            msg = f"DDP Init (Rank {rank}/{world_size}, LocalRank {local_rank}, Device: cuda:{local_rank})"
            if logger and (not hasattr(logger, 'disabled') or not logger.disabled):
                logger.info(msg)
            else:
                print(f"Info: {msg}")  # Fallback if logger not ready
            torch.distributed.barrier()  # Sync after setup
        except Exception as e:
            # Use print as logger might not be fully setup or for critical DDP failure
            print(f"CRITICAL ERROR: DDP init failed: {e}")
            traceback.print_exc()
            raise
    else:
        msg = "DDP not enabled."
        if logger and (not hasattr(logger, 'disabled') or not logger.disabled):
            logger.info(msg)
        else:
            print(f"Info: {msg}")
    return is_dist, rank, world_size, local_rank


def load_training_data(args, is_distributed, rank, world_size, data_collator):
    """Loads standard Arrow datasets specifically for training."""
    global logger
    from datasets import load_from_disk
    from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
    train_dl, train_sampler = None, None

    if not args.train_dataset_path: logger.error("Missing training path."); raise ValueError("Missing path.")
    logger.info(f"Loading train data: {args.train_dataset_path}")
    try:
        ds = load_from_disk(args.train_dataset_path);
        logger.info(f"Train size: {len(ds):,}")
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True,
                                     seed=args.seed) if is_distributed else RandomSampler(ds)
        if rank == 0: logger.info(f"Using {'DistributedSampler' if is_distributed else 'RandomSampler'} for training.")
        train_dl = DataLoader(ds, sampler=sampler, batch_size=args.per_device_train_batch_size,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=data_collator,
                              persistent_workers=(True if args.num_workers > 0 else False))
        if rank == 0: logger.info("Train DataLoader created.")
        train_sampler = sampler
    except Exception as e:
        logger.error(f"Train data load fail: {e}", exc_info=True)
        raise
    return train_dl, train_sampler


def save_checkpoint(args, model, optimizer, lr_scheduler, scaler, epoch, global_step, rank, tokenizer=None):
    """Saves training state. Only rank 0 executes."""
    if rank != 0: return
    global logger
    import torch;
    import numpy as np;
    import random;
    from pathlib import Path
    ckpt_dir = Path(args.output_dir) / f"checkpoint-{global_step}";
    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving checkpoint {global_step} to {ckpt_dir}")
    except OSError as e:
        logger.error(f"Failed to create checkpoint directory {ckpt_dir}: {e}")
        return

    unwrapped_model = model.module if hasattr(model, 'module') else model
    state = {
        'model': unwrapped_model.state_dict(), 'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(), 'scaler': scaler.state_dict() if scaler.is_enabled() else None,
        'epoch': epoch, 'global_step': global_step, 'args': vars(args),
        'torch_rng_state': torch.get_rng_state(), 'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
        'torch_cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }
    state_file = ckpt_dir / "training_state.pt"
    try:
        torch.save(state, state_file)
        logger.info(f"Saved training state to: {state_file}")
    except Exception as e:
        logger.error(f"Failed to save training state: {e}", exc_info=True)

    try:
        unwrapped_model.save_pretrained(ckpt_dir)
        logger.info(f"Saved model weights and config to {ckpt_dir}")
        if tokenizer:
            tokenizer.save_pretrained(ckpt_dir)
            logger.info(f"Saved tokenizer to {ckpt_dir}")
    except Exception as e:
        logger.error(f"Failed to save model/tokenizer using save_pretrained: {e}", exc_info=True)


def create_small_model_config(base_model_name: str, corpus_size_tag: str, tokenizer: PreTrainedTokenizer,
                              logger_instance: logging.Logger):
    """Creates a small model configuration based on a corpus size tag."""
    # Use passed logger_instance instead of global logger for clarity if this function is moved
    try:
        config = AutoConfig.from_pretrained(base_model_name)
        if logger_instance: logger_instance.info(f"Loaded base config structure from: {base_model_name}")
    except Exception as e:
        if logger_instance: logger_instance.error(f"Failed to load base config '{base_model_name}': {e}", exc_info=True)
        raise
    tag = corpus_size_tag.lower()
    if tag == "10m":
        target_n_layer, target_n_head, target_n_embd = 4, 4, 256;
    elif tag == "100m":
        target_n_layer, target_n_head, target_n_embd = 6, 6, 384;
    else:
        raise ValueError(f"Unknown corpus_size_tag: '{corpus_size_tag}'. Expected '10m' or '100m'.")

    if logger_instance: logger_instance.info(f"Applying config for '{tag}'.")

    if hasattr(config, 'n_layer'): config.n_layer = target_n_layer
    if hasattr(config, 'n_head'): config.n_head = target_n_head
    if hasattr(config, 'n_embd'): config.n_embd = target_n_embd
    if hasattr(config, 'vocab_size'):
        config.vocab_size = len(tokenizer)
    else:
        logger_instance.warning(f"Base config type {type(config)} might not have 'vocab_size'.")
    if hasattr(config, 'use_cache'): config.use_cache = False  # Good practice for training
    if logger_instance:
        logger_instance.info(f"Final SMALL config params: n_layer={getattr(config, 'n_layer', 'N/A')}, "
                             f"n_head={getattr(config, 'n_head', 'N/A')}, n_embd={getattr(config, 'n_embd', 'N/A')}, "
                             f"vocab_size={getattr(config, 'vocab_size', 'N/A')}")
    return config


# --- Modified run_local_evaluation Function ---

def run_local_evaluation(args, checkpoint_dir: Path, global_step: int, rank: int):
    """
    Runs evaluate.py locally as a subprocess.
    This function is ONLY called if args.local_eval is True.
    """
    if rank != 0: return  # Only rank 0 runs evaluation
    global logger, run  # Access global Neptune run object

    eval_output_dir = checkpoint_dir / "eval_results"  # Define where eval results should go
    try:
        eval_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(
            f"Failed to create local evaluation output directory {eval_output_dir}: {e}. Skipping local evaluation.")
        return

    logger.info(f"--- Running Local Evaluation for Step {global_step} ---")
    python_executable = sys.executable  # Use the same python executable running this script
    eval_script_path = Path(args.evaluate_script_path).resolve()

    # Construct arguments for evaluate.py
    eval_args_list = [
        str(python_executable),
        str(eval_script_path),
        "--checkpoint_path", str(checkpoint_dir.resolve()),
        "--output_dir", str(eval_output_dir.resolve()),
        "--seed", str(args.seed),
        "--num_workers", str(args.num_workers),
    ]
    if args.use_amp: eval_args_list.append("--use_amp")

    if args.trigger_standard_eval:
        eval_args_list.append("--run_standard_eval")
        if args.validation_dataset_path:
            eval_args_list.extend(["--validation_dataset_path", str(Path(args.validation_dataset_path).resolve())])
        # No warning here, parse_args should have caught missing path if trigger_standard_eval is true

    if args.trigger_priming_eval:
        eval_args_list.append("--run_priming_eval")
        if args.priming_eval_dir_path:
            eval_args_list.extend(["--priming_eval_dir_path", str(Path(args.priming_eval_dir_path).resolve())])
        # No warning here, parse_args should have caught missing path

    # Add Neptune details if configured so local eval can log
    if args.neptune_project:
        eval_args_list.extend(["--neptune_project", args.neptune_project])
    # Pass the current training Neptune run ID if available
    current_neptune_run_id = None
    if NEPTUNE_AVAILABLE and run and hasattr(run, '_sys_id'):
        try:
            current_neptune_run_id = run['_sys_id'].fetch()
            eval_args_list.extend(
                ["--neptune_run_id", current_neptune_run_id])  # For evaluate.py to potentially log to the same run
            logger.info(f"Passing current Neptune run ID ({current_neptune_run_id}) to local evaluate.py.")
        except Exception as e:
            logger.warning(f"Could not fetch Neptune run ID for local eval: {e}")
    # NEPTUNE_API_TOKEN should be in the environment for evaluate.py to pick up

    logger.info(f"Executing local evaluation command:")
    logger.info(f"{' '.join(eval_args_list)}")

    eval_start_time = time.time()
    try:
        # Run evaluation synchronously, allow output to stream to main log
        # Using os.environ.copy() ensures the subprocess gets the current environment,
        # including CUDA_VISIBLE_DEVICES, NEPTUNE_API_TOKEN etc.
        process_env = os.environ.copy()
        result = subprocess.run(eval_args_list, check=True, env=process_env, capture_output=True, text=True)
        logger.info(f"Local evaluation process completed successfully in {time.time() - eval_start_time:.2f} seconds.")
        if result.stdout: logger.info(f"Local Eval STDOUT:\n{result.stdout}")
        if result.stderr: logger.warning(f"Local Eval STDERR:\n{result.stderr}")  # Log stderr as warning

    except FileNotFoundError:
        logger.error(
            f"Error: Python executable '{python_executable}' or script '{eval_script_path}' not found for local evaluation.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running local evaluation script for step {global_step}. Return code: {e.returncode}")
        logger.error(f"Local Eval STDOUT:\n{e.stdout}")
        logger.error(f"Local Eval STDERR:\n{e.stderr}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during local evaluation for step {global_step}: {e}",
                     exc_info=True)


def train_epoch(args, model, optimizer, lr_scheduler, scaler, train_dataloader,
                train_sampler, epoch, global_step, device, rank, world_size, run, tokenizer,
                max_train_steps, log_save_steps_set):  # MODIFIED: Added log_save_steps_set
    """
    Runs one training epoch.
    Saves checkpoints and triggers local evaluation periodically if enabled.
    Stops early if max_train_steps is reached.
    """
    global logger
    import torch;
    from torch.utils.data import DistributedSampler;
    from tqdm.auto import tqdm;
    import math;
    import sys;
    import gc
    from torch.nn.utils import clip_grad_norm_
    from torch.amp import autocast

    model.train()
    is_distributed = train_sampler is not None and isinstance(train_sampler, DistributedSampler)
    if is_distributed:
        try:
            train_sampler.set_epoch(epoch)
        except AttributeError:
            logger.warning("Train sampler does not have set_epoch method.")

    disable_tqdm = rank != 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_train_epochs}", leave=True,
                        disable=disable_tqdm, position=0)
    total_loss_since_logging, steps_since_logging = 0.0, 0
    last_logged_loss = float('inf')

    for step, batch in enumerate(progress_bar):
        try:
            batch_on_device = {k: v.to(device, non_blocking=True) for k, v in batch.items() if
                               isinstance(v, torch.Tensor)}
        except RuntimeError as e:
            logger.error(f"Error moving train batch to device: {e}")
            optimizer.zero_grad(set_to_none=True)
            continue
        try:
            amp_enabled = args.use_amp and device.type == 'cuda'
            with autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(**batch_on_device)
                loss = outputs.loss
            if loss is None:
                logger.error("Loss was None from model output.")
                optimizer.zero_grad(set_to_none=True)
                continue
            if not torch.isfinite(loss):
                logger.warning(f"Non-finite loss detected ({loss.item()}). Skipping update for this batch.")
                optimizer.zero_grad(set_to_none=True)
                continue
            scaled_loss = loss / args.gradient_accumulation_steps
            current_loss_value = loss.item()
        except Exception as e:
            if "BackendCompilerFailed" in str(e):
                logger.error(f"Forward pass failed due to BackendCompilerFailed: {e}")
                logger.error("Recommendation: Disable torch.compile() if enabled elsewhere.")
            else:
                logger.error(f"Forward pass error: {e}", exc_info=True)
            optimizer.zero_grad(set_to_none=True)
            continue

        if amp_enabled:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        total_loss_since_logging += current_loss_value
        steps_since_logging += 1
        last_logged_loss = current_loss_value  # Keep track of the most recent micro-batch loss for progress bar

        if (step + 1) % args.gradient_accumulation_steps == 0:
            try:
                if args.max_grad_norm and args.max_grad_norm > 0:
                    if scaler.is_enabled(): scaler.unscale_(optimizer)
                    params_to_clip = model.module.parameters() if hasattr(model, 'module') else model.parameters()
                    clip_grad_norm_(params_to_clip, args.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            except Exception as e:
                logger.error(f"Optimizer step error: {e}", exc_info=True)
                optimizer.zero_grad(set_to_none=True)  # Ensure gradients are cleared

            if rank == 0 and global_step % args.logging_steps == 0 and steps_since_logging > 0:
                avg_loss = total_loss_since_logging / steps_since_logging
                lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler,
                                                              'get_last_lr') and lr_scheduler.get_last_lr() else \
                    optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch + 1} | Step {global_step}: Avg Train Loss = {avg_loss:.4f}, LR = {lr:.6e}")
                if NEPTUNE_AVAILABLE and run:
                    try:
                        if math.isfinite(avg_loss): run["train/step_loss"].append(avg_loss, step=global_step)
                        if math.isfinite(lr): run["train/learning_rate"].append(lr, step=global_step)
                        if torch.cuda.is_available(): run["train/gpu_mem_alloc_gb"].append(
                            torch.cuda.memory_allocated(device) / 1e9, step=global_step)
                        if scaler.is_enabled(): run["train/grad_scale"].append(scaler.get_scale(), step=global_step)
                    except Exception as e:
                        logger.warning(f"Neptune train log failed at step {global_step}: {e}")
                total_loss_since_logging, steps_since_logging = 0.0, 0

            # ---- MODIFIED: Checkpoint saving logic ----
            time_for_log_save = global_step in log_save_steps_set
            # Regular save: ensure it's enabled, it's a multiple, and it's not step 0 (handled by init checkpoint)
            time_for_regular_save = args.save_steps > 0 and global_step > 0 and global_step % args.save_steps == 0

            should_save_this_step = time_for_log_save or time_for_regular_save

            if should_save_this_step:
                # ---- END MODIFIED ----
                if is_distributed: torch.distributed.barrier()
                save_checkpoint(args, model, optimizer, lr_scheduler, scaler, epoch, global_step, rank, tokenizer)
                if is_distributed: torch.distributed.barrier()

            # --- LOCAL Evaluation Trigger Step (Only if args.local_eval is True) ---
            if args.local_eval:  # Check this first
                time_for_std_eval = args.trigger_standard_eval and global_step > 0 and global_step % args.eval_steps == 0
                time_for_prime_eval = args.trigger_priming_eval and global_step > 0 and global_step % args.priming_eval_steps == 0
                trigger_eval_now = time_for_std_eval or time_for_prime_eval

                if trigger_eval_now:
                    checkpoint_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                    if rank == 0 and not checkpoint_dir.is_dir():
                        logger.warning(
                            f"Local eval: Checkpoint {checkpoint_dir} not found for eval trigger. "
                            f"This might happen if save_steps/log_steps do not align with eval_steps. Saving now as a fallback."
                        )
                        if is_distributed: torch.distributed.barrier()
                        save_checkpoint(args, model, optimizer, lr_scheduler, scaler, epoch, global_step, rank,
                                        tokenizer)
                        if is_distributed: torch.distributed.barrier()

                    if rank == 0 and checkpoint_dir.is_dir():
                        run_local_evaluation(args, checkpoint_dir, global_step, rank)  # Modified function call
                    elif rank == 0:
                        logger.error(
                            f"Local eval: Failed to find or save checkpoint {checkpoint_dir}. Cannot run local evaluation.")
            elif rank == 0 and global_step > 0 and (
                    global_step % args.eval_steps == 0 or global_step % args.priming_eval_steps == 0):
                # Log if it *would* have been an eval step but local_eval is off
                if args.trigger_standard_eval or args.trigger_priming_eval:
                    logger.info(
                        f"Step {global_step}: Local evaluation is disabled via --local_eval. Evaluation will be handled by external monitor if configured.")

            if max_train_steps > 0 and global_step >= max_train_steps:
                if rank == 0: logger.info(
                    f"Max steps ({max_train_steps}) reached at step {global_step} within epoch {epoch + 1}.")
                progress_bar.update(args.gradient_accumulation_steps - ((step + 1) % args.gradient_accumulation_steps))
                progress_bar.set_postfix(
                    {"loss": f"{last_logged_loss:.4f}", "step": global_step, "status": "Max steps reached"})
                return global_step

        if rank == 0: progress_bar.set_postfix({"loss": f"{last_logged_loss:.4f}", "step": global_step})

    if rank == 0: progress_bar.close(); logger.info(
        f"--- Epoch {epoch + 1} Finished (Reached Optimizer Step {global_step}) ---")
    return global_step


# --- Main Function (Modified Setup) ---
def main():
    """Main function to parse arguments, set up, and run training."""
    global run, logger  # Declare run and logger as global

    args = parse_args()  # Parse arguments first
    # Setup DDP and logging early, as they affect how messages are displayed
    is_distributed, rank, world_size, local_rank = setup_distributed(args)
    setup_logging(rank=rank)  # logger is initialized here
    # Now logger is safe to use by rank 0

    if rank == 0: logger.info(f"***** Starting Training Script (train.py) *****")
    if rank == 0: logger.info(f"Running with Arguments: {vars(args)}")
    if rank == 0 and not args.local_eval:
        logger.info("Local evaluation triggering by this script is DISABLED (--local_eval not set).")
        logger.info("Evaluation, if configured, will be handled by an external monitoring script.")
    elif rank == 0 and args.local_eval:
        logger.info("Local evaluation triggering by this script is ENABLED (--local_eval set).")

    # Setup Device and Seed
    device = get_device()  # Depends on LOCAL_RANK from DDP setup
    set_seed(args.seed + rank)  # Seed per rank for DDP

    # Neptune Setup (Rank 0 Only)
    run = None  # Initialize run to None
    if rank == 0 and NEPTUNE_AVAILABLE and args.neptune_project:
        try:
            run = neptune.init_run(project=args.neptune_project, name=args.neptune_run_name, tags=args.neptune_tags)
            args_log = {}
            logger.info("Processing arguments for Neptune logging...")
            for k, v_item in vars(args).items():  # Renamed v to v_item to avoid conflict
                if isinstance(v_item, Path):
                    args_log[k] = str(v_item)
                elif isinstance(v_item, list):
                    args_log[k] = ','.join(map(str, v_item)) if v_item else "None"
                elif v_item is None:
                    args_log[k] = "None"
                else:
                    args_log[k] = v_item
            try:
                run["parameters"] = args_log
                logger.info("Successfully logged processed parameters to Neptune.")
            except Exception as neptune_log_e:
                logger.error(f"Failed to log processed parameters to Neptune: {neptune_log_e}")

            logger.info(f"Neptune logging enabled. Run URL: {run.get_url()}")
            # run.sync() # sync can sometimes be problematic, often not strictly needed immediately
        except Exception as e:
            logger.error(f"Neptune init failed: {e}. Neptune logging disabled for this run.", exc_info=True)
            run = None  # Ensure run is None if init fails
    elif rank == 0:
        logger.info(
            "Neptune logging is disabled (either NEPTUNE_AVAILABLE is False or --neptune_project not specified).")

    # === Model & Tokenizer Loading ===
    model, tokenizer, config = None, None, None
    try:
        if rank == 0: logger.info(
            f"Initializing NEW small model ({args.model_size} config) from scratch. Tokenizer base: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                if rank == 0: logger.info(f"Set tokenizer pad_token to its eos_token: '{tokenizer.eos_token}'")
            else:  # Add a new pad token if no EOS token exists (unlikely for GPT2 but good practice)
                added_tokens = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                if rank == 0: logger.warning(
                    f"Tokenizer had no pad_token or eos_token. Added new [PAD] token (added {added_tokens} new tokens). Model embedding layer will need resizing if this happens after model init.")
        # Pass the initialized logger instance to create_small_model_config
        config = create_small_model_config(base_model_name=args.model, corpus_size_tag=args.model_size,
                                           tokenizer=tokenizer,
                                           logger_instance=logger if rank == 0 else logging.getLogger(
                                               "dummy_logger"))  # Give a dummy logger for non-rank0
        model = GPT2LMHeadModel(config=config)  # Initialize model with this config
        # Resize embeddings if new tokens were added to tokenizer *after* config was made (usually config reflects vocab size)
        # model.resize_token_embeddings(len(tokenizer)) # Good practice if vocab changed.

        if rank == 0: logger.info("Model architecture initialized with random weights.")
        model.to(device);
        if rank == 0: logger.info(f"Initialized model moved to {device}")

        if is_distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=False)  # Set find_unused_parameters if needed
            if rank == 0: logger.info("Model wrapped with DistributedDataParallel (DDP).")
    except Exception as e:
        logger.critical(f"Model/Tokenizer initialization failed (Rank {rank}): {e}", exc_info=True)
        if NEPTUNE_AVAILABLE and run: run["critical_errors/model_init"] = traceback.format_exc()
        sys.exit(1)

    # === Training Data Loading ===
    train_dataloader, train_sampler = None, None
    try:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        train_dataloader, train_sampler = load_training_data(args, is_distributed, rank, world_size, data_collator)
        if rank == 0: logger.info(f"Loaded train dataset ({len(train_dataloader.dataset):,} samples).")
    except Exception as e:
        logger.critical(f"Train data loading failed (Rank {rank}): {e}", exc_info=True)
        if NEPTUNE_AVAILABLE and run: run["critical_errors/data_load"] = traceback.format_exc()
        sys.exit(1)

    # === Optimizer, Scheduler, Scaler Setup ===
    if rank == 0: logger.info(f"Setting up Optimizer, LR Scheduler, Grad Scaler...")
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]  # Ensure consistent naming
    # Use model.module.named_parameters() if DDP is used, otherwise model.named_parameters()
    named_params_iterable = model.module.named_parameters() if is_distributed else model.named_parameters()
    optimizer_grouped_parameters = [
        {"params": [p for n, p in named_params_iterable if
                    not any(nd in n.lower() for nd in no_decay) and p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in named_params_iterable if any(nd in n.lower() for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0}, ]

    # Filter out empty param groups
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g["params"]]

    total_params_count = sum(p.numel() for p in (model.module.parameters() if is_distributed else model.parameters()))
    num_trainable_params = sum(p.numel() for g in optimizer_grouped_parameters for p in g['params'])
    if rank == 0: logger.info(f"Model Parameters: Total={total_params_count:,}, Trainable={num_trainable_params:,}")
    if num_trainable_params == 0:
        logger.critical("No trainable parameters found. Check model or optimizer group setup.")
        if NEPTUNE_AVAILABLE and run: run["critical_errors/no_trainable_params"] = "No trainable parameters found."
        sys.exit(1)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) if len(
        train_dataloader) > 0 else 1
    max_train_steps_from_epochs = args.num_train_epochs * num_update_steps_per_epoch

    if args.max_steps > 0:
        max_train_steps = args.max_steps
        if rank == 0: logger.info(
            f"Training will be limited to {max_train_steps:,} total optimizer steps (due to --max_steps).")
    else:
        max_train_steps = max_train_steps_from_epochs
        if rank == 0: logger.info(f"Calculated total training optimizer steps based on epochs: {max_train_steps:,}")

    if max_train_steps <= 0 and rank == 0:
        logger.warning(
            "Max training steps is zero or negative. Training might not proceed beyond setup if not resuming past this.")

    eff_warmup = min(args.num_warmup_steps, max_train_steps) if max_train_steps > 0 else args.num_warmup_steps
    if rank == 0: logger.info(f"Effective warmup steps for LR scheduler: {eff_warmup}")

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=eff_warmup,
        num_training_steps=max_train_steps if max_train_steps > 0 else 1  # Scheduler needs num_training_steps > 0
    )
    scaler_enabled = args.use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=scaler_enabled)
    if args.use_amp and not scaler_enabled and rank == 0: logger.warning(
        "AMP training was requested, but CUDA is not available. AMP disabled.")
    if rank == 0: logger.info(f"Automatic Mixed Precision (AMP) enabled: {scaler.is_enabled()}.")

    # === Resume from Checkpoint Logic ===
    start_epoch, global_step, resumed_from_checkpoint = 0, 0, False
    if args.checkpoint_path:
        state_file = Path(args.checkpoint_path) / "training_state.pt"
        if state_file.is_file():
            if rank == 0: logger.info(f"Attempting to load training state from: {state_file}")
            try:
                # Load checkpoint to CPU first to avoid GPU OOM if checkpoint is large, then move model to device
                ckpt = torch.load(state_file, map_location='cpu')
                model_to_load = model.module if hasattr(model, 'module') else model

                # Handle potential mismatch in vocab size if tokenizer changed
                ckpt_model_vocab_size = ckpt['model'].get('transformer.wte.weight', {}).shape[
                    0] if 'transformer.wte.weight' in ckpt['model'] else None
                current_model_vocab_size = model_to_load.config.vocab_size
                if ckpt_model_vocab_size and ckpt_model_vocab_size != current_model_vocab_size:
                    logger.warning(
                        f"Checkpoint vocab size ({ckpt_model_vocab_size}) differs from current model vocab size ({current_model_vocab_size}). Resizing model embeddings.")
                    model_to_load.resize_token_embeddings(len(tokenizer))  # Ensure tokenizer is the source of truth

                missing_keys, unexpected_keys = model_to_load.load_state_dict(ckpt['model'], strict=False)
                if rank == 0:
                    logger.info(
                        f"Model state loaded from checkpoint. Missing keys: {missing_keys or 'None'}. Unexpected keys: {unexpected_keys or 'None'}.")

                # Move model to device *after* loading state dict potentially from CPU
                model.to(device)
                if rank == 0: logger.info(f"Model moved to {device} after loading checkpoint state.")

                if 'optimizer' in ckpt: optimizer.load_state_dict(ckpt['optimizer'])
                if 'lr_scheduler' in ckpt: lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
                if 'scaler' in ckpt and ckpt['scaler'] is not None and scaler.is_enabled():
                    scaler.load_state_dict(ckpt['scaler'])
                elif scaler.is_enabled() and ('scaler' not in ckpt or ckpt['scaler'] is None):
                    logger.warning(
                        "Resuming with AMP enabled, but no scaler state found in checkpoint. Scaler starts fresh.")

                start_epoch = ckpt.get('epoch', 0)
                global_step = ckpt.get('global_step', 0)

                if max_train_steps > 0 and global_step >= max_train_steps:
                    # Effectively prevent training loop from running if already completed
                    start_epoch = args.num_train_epochs  # This will make the for loop range empty
                    if rank == 0: logger.info(
                        f"Resuming from step {global_step}, which already meets or exceeds max_steps ({max_train_steps}). Training will not continue further.")
                else:
                    # If not completed, prepare to start the *next* epoch after the one saved
                    start_epoch += 1
                    if rank == 0: logger.info(
                        f"Resuming training. Will start from Epoch {start_epoch}, Global Step {global_step}")

                resumed_from_checkpoint = True
                try:
                    if 'torch_rng_state' in ckpt: torch.set_rng_state(
                        ckpt['torch_rng_state'])  # Already on CPU from map_location='cpu'
                    if device.type == 'cuda' and 'torch_cuda_rng_state_all' in ckpt and ckpt[
                        'torch_cuda_rng_state_all']:
                        # Ensure CUDA RNG states are loaded correctly after model is on CUDA device
                        torch.cuda.set_rng_state_all([s.to(device) for s in ckpt['torch_cuda_rng_state_all']])
                    if 'numpy_rng_state' in ckpt: np.random.set_state(ckpt['numpy_rng_state'])
                    if 'python_rng_state' in ckpt: random.setstate(ckpt['python_rng_state'])
                    if rank == 0: logger.info("Restored RNG states from checkpoint.")
                except Exception as rng_e:
                    logger.warning(f"Could not fully restore RNG states from checkpoint: {rng_e}")

                del ckpt;
                gc.collect();
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                if is_distributed: torch.distributed.barrier()  # Sync all processes after loading
            except Exception as e:
                logger.error(f"Failed to load checkpoint from {state_file}: {e}", exc_info=True)
                start_epoch, global_step, resumed_from_checkpoint = 0, 0, False  # Reset to start from scratch
        else:
            if rank == 0: logger.warning(
                f"Checkpoint path {args.checkpoint_path} specified, but training_state.pt not found. Starting training from scratch.")
    else:
        if rank == 0: logger.info("No checkpoint path specified. Starting training from scratch.")

    # ---- ADDED: Save checkpoint at initialization (step 0) if not resuming ----
    if global_step == 0 and not resumed_from_checkpoint:
        if rank == 0:
            logger.info("Saving initial checkpoint at step 0 as training is starting from scratch.")
            save_checkpoint(args, model, optimizer, lr_scheduler, scaler, start_epoch, 0, rank, tokenizer)
        if is_distributed:
            torch.distributed.barrier()  # Ensure all processes wait if rank 0 saved, and sync before training loop
    # ---- END ADDED ----

    # ---- ADDED: Generate logarithmic save steps ----
    log_save_steps_set = set()
    if args.save_steps > 0:  # Only if regular saving is configured, to give context to "first save step"
        current_log_step = 1
        while current_log_step < args.save_steps:
            log_save_steps_set.add(current_log_step)
            next_log_step = current_log_step * 2
            if next_log_step <= current_log_step:  # Handles overflow or current_log_step becoming 0
                if rank == 0: logger.warning(
                    f"Logarithmic step generation issue: next step {next_log_step} not greater than current {current_log_step}. Stopping log step generation.")
                break
            current_log_step = next_log_step
    if rank == 0 and log_save_steps_set:  # Log only if set is not empty
        logger.info(
            f"Calculated logarithmic save steps (up to args.save_steps={args.save_steps}): {sorted(list(log_save_steps_set))}")
    elif rank == 0:
        logger.info(f"No logarithmic save steps generated (e.g. args.save_steps is too small or 0).")
    # ---- END ADDED ----

    if rank == 0:
        logger.info("***** Training Configuration Summary *****")
        logger.info(f"   Distributed Training: {is_distributed} (World Size: {world_size}, Rank: {rank})")
        logger.info(f"   Device: {device}")
        logger.info(f"   AMP Enabled: {scaler.is_enabled()}")
        logger.info(f"   Seed: {args.seed} (Rank 0 base, others incremented)")
        logger.info(f"   Output Directory: {args.output_dir}")
        logger.info(
            f"   Resumed from Checkpoint: {resumed_from_checkpoint} (Start Epoch: {start_epoch}, Global Step: {global_step})")
        logger.info(f"   Total Epochs: {args.num_train_epochs}")
        logger.info(f"   Max Optimizer Steps: {max_train_steps if max_train_steps > 0 else 'Determined by epochs'}")
        logger.info(f"   Effective Warmup Steps: {eff_warmup}")
        logger.info(f"   Logarithmic Save Steps: {sorted(list(log_save_steps_set)) if log_save_steps_set else 'None'}")
        logger.info(f"   Local Evaluation by this script: {'ENABLED' if args.local_eval else 'DISABLED'}")
        if args.local_eval:
            logger.info(f"      Eval Script: {args.evaluate_script_path}")
            logger.info(f"      Standard Eval Triggered: {args.trigger_standard_eval} (every {args.eval_steps} steps)")
            logger.info(
                f"      Priming Eval Triggered: {args.trigger_priming_eval} (every {args.priming_eval_steps} steps)")

    # === Training Loop ===
    if is_distributed: torch.distributed.barrier()  # Ensure all setup is done before training starts
    training_start_time = time.time()
    final_global_step = global_step  # Initialize with potentially resumed step count

    try:
        for epoch in range(start_epoch, args.num_train_epochs):
            if max_train_steps > 0 and final_global_step >= max_train_steps:
                if rank == 0: logger.info(
                    f"Max steps ({max_train_steps}) already reached or exceeded before starting epoch {epoch}. Current step: {final_global_step}. Stopping training.")
                break

            if rank == 0: logger.info(
                f"--- Starting Epoch {epoch + 1}/{args.num_train_epochs} (Current Global Step: {final_global_step}) ---")
            model.train()  # Ensure model is in training mode

            # Update final_global_step with the value returned by train_epoch
            final_global_step = train_epoch(
                args=args, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, scaler=scaler,
                train_dataloader=train_dataloader, train_sampler=train_sampler,
                epoch=epoch, global_step=final_global_step, device=device, rank=rank, world_size=world_size,
                run=run, tokenizer=tokenizer,
                max_train_steps=max_train_steps,
                log_save_steps_set=log_save_steps_set  # MODIFIED: Pass the set
            )
            # Check after epoch if max_steps was met, to break outer loop
            if max_train_steps > 0 and final_global_step >= max_train_steps:
                if rank == 0: logger.info(
                    f"Max steps ({max_train_steps}) reached during epoch {epoch + 1}. Final global step: {final_global_step}. Stopping training.")
                break

        training_duration = time.time() - training_start_time
        if rank == 0:
            logger.info(f"***** Training Finished *****")
            logger.info(f"Total Training Time: {training_duration:.2f} seconds")
            logger.info(f"Final Global Optimizer Step Reached: {final_global_step}")

        # Final Saving (Rank 0 Only)
        # Check if any training was done or if it's a fresh start (final_global_step > 0 or (final_global_step==0 and not resumed_from_checkpoint))
        # The initial checkpoint for step 0 is already saved if not resuming.
        # We only need to save a "final" checkpoint if we actually trained or if no initial checkpoint was made due to resuming.
        # The current logic: always save a final checkpoint based on final_global_step.
        # This is generally fine as it ensures the very last state is captured.
        if rank == 0:
            # Use the 'final_global_step' for naming the last checkpoint directory
            # Avoid re-saving step 0 if it was the only step and an initial checkpoint was already made
            should_save_final = True
            if final_global_step == 0 and not resumed_from_checkpoint:
                # Step 0 was already saved at initialization. No need to save again unless user wants explicit "final_model" dir.
                # The save_checkpoint for "checkpoint-0" is already done.
                # We will still proceed to save to "final_model" distinct path.
                logger.info(f"Final global step is 0 and training started fresh; initial checkpoint already saved.")
                # To prevent redundant full save_checkpoint call for step 0:
                # However, if we want the final_model artifacts, we proceed.
                # The current logic saves checkpoint-{final_global_step} and then final_model.
                # This is acceptable.
                pass

            final_checkpoint_dir = Path(args.output_dir) / f"checkpoint-{final_global_step}"
            # Only call save_checkpoint if it's not a redundant step 0 save.
            # Or, simplify: always call it, save_checkpoint is idempotent for directory creation.
            # The main concern is overwriting or redundant logging.
            # Let's save if it's not the *exact same* initial checkpoint event.
            # Condition: if we actually trained (final_global_step > 0) OR if we resumed (so init save didn't happen).
            if final_global_step > 0 or (final_global_step == 0 and resumed_from_checkpoint):
                logger.info(
                    f"Saving final training state at step {final_global_step} using save_checkpoint logic to: {final_checkpoint_dir}")
                save_checkpoint(args, model, optimizer, lr_scheduler, scaler,
                                epoch if 'epoch' in locals() else args.num_train_epochs - 1,  # Use last known epoch
                                final_global_step, rank, tokenizer)
                logger.info(f"Final training state saved successfully to {final_checkpoint_dir}.")
            elif final_global_step == 0 and not resumed_from_checkpoint:
                logger.info(
                    f"Final step is 0 (fresh start), initial checkpoint at 'checkpoint-0' is already the final state.")
                # final_checkpoint_dir would be 'checkpoint-0' which was just created.

            # Always save to a specific 'final_model' directory for clarity
            final_model_distinct_path = Path(args.output_dir) / "final_model"
            final_model_distinct_path.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Saving final model weights, tokenizer, and config to distinct directory: {final_model_distinct_path}")
            model_to_save_final = model.module if hasattr(model, 'module') else model
            model_to_save_final.save_pretrained(final_model_distinct_path)
            tokenizer.save_pretrained(final_model_distinct_path)
            # Save training args to this final_model directory
            args_dict_final_save = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            with open(final_model_distinct_path / "training_args.json", "w") as f:
                json.dump(args_dict_final_save, f, indent=4)
            logger.info(f"Final model, tokenizer, and args saved to {final_model_distinct_path}")

            if NEPTUNE_AVAILABLE and run:
                try:
                    if final_checkpoint_dir.is_dir():  # This dir should exist from save_checkpoint or init save
                        logger.info(
                            f"Attempting to upload final checkpoint directory '{final_checkpoint_dir.name}' to Neptune.")
                        run[f"checkpoints/final_step_{final_global_step}"].upload_files(str(final_checkpoint_dir))
                    if final_model_distinct_path.is_dir():
                        logger.info(
                            f"Attempting to upload final model directory '{final_model_distinct_path.name}' to Neptune.")
                        run["final_model_artifacts"].upload_files(str(final_model_distinct_path))

                    run["training_summary/duration_seconds"] = training_duration
                    run["training_summary/final_global_step"] = final_global_step
                    logger.info("Logged final training summary and artifacts to Neptune.")
                except Exception as e:
                    logger.warning(f"Neptune final upload/log failed: {e}", exc_info=True)

            # If local eval was enabled, consider one final evaluation run on the final model
            if args.local_eval and (args.trigger_standard_eval or args.trigger_priming_eval):
                logger.info(f"Performing final local evaluation on the model from step {final_global_step}.")
                # The checkpoint for final_global_step should exist.
                if final_checkpoint_dir.is_dir():
                    run_local_evaluation(args, final_checkpoint_dir, final_global_step, rank)
                else:
                    logger.warning(
                        f"Final checkpoint directory {final_checkpoint_dir} not found for final local evaluation. Skipping.")


    except KeyboardInterrupt:
        if rank == 0: logger.warning("Training interrupted by user (KeyboardInterrupt).")
        if NEPTUNE_AVAILABLE and run and rank == 0: run["status/training"] = "interrupted_keyboard"
    except Exception as e:
        logger.critical(f"An unhandled exception occurred during the training loop (Rank {rank}): {e}", exc_info=True)
        if NEPTUNE_AVAILABLE and run and rank == 0:
            run["critical_errors/training_loop"] = traceback.format_exc()
    finally:
        if is_distributed:
            torch.distributed.destroy_process_group()
        if rank == 0 and NEPTUNE_AVAILABLE and run:
            try:
                logger.info("Stopping Neptune run...")
                run.sync()  # Ensure all data is sent before stopping
                run.stop()
                logger.info("Neptune run stopped.")
            except Exception as ne:
                logger.error(f"Neptune stop operation failed: {ne}", exc_info=True)
        logger.info(f"Training script (train.py) finished on Rank {rank}.")


# Define fallback tqdm before __main__ block for safety, though import is tried in main
def _fallback_tqdm(iterable, *args, **kwargs): return iterable


if __name__ == "__main__":
    # Basic config for logging before DDP setup or if script is run directly without full setup.
    # This will be overridden by setup_logging once rank is known.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    # Attempt to import tqdm globally if needed, or handle its absence.
    try:
        from tqdm.auto import tqdm
    except ImportError:
        if int(os.environ.get("RANK", "0")) == 0: print(
            "Warning: tqdm.auto not installed. Progress bars will be disabled.")
        tqdm = _fallback_tqdm  # Assign fallback
    main()