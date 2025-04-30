# src/train.py (MODIFIED FOR DECOUPLED EVALUATION)

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
import subprocess # <<< ADDED for submitting sbatch jobs

# --- ML/data library imports ---
# Keep imports needed for training, model loading, data loading, saving
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import (
    GPT2LMHeadModel, # Assuming this is the primary model class
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizer, # Keep for type hints
    get_scheduler,
    DataCollatorForLanguageModeling
)
from torch.cuda.amp import GradScaler
import torch.amp # Use torch.amp namespace
from datasets import load_from_disk
from torch.nn.utils import clip_grad_norm_

# Optional Neptune import
try:
    import neptune
    NEPTUNE_AVAILABLE = True
except ImportError:
    print("Neptune.ai library not found, Neptune logging will be disabled.")
    sys.modules['neptune'] = None
    NEPTUNE_AVAILABLE = False
    neptune = None # Ensure neptune is None if import failed

# --- Logger Setup ---
logger = None # Define globally, assign after setup_logging

# --- Helper Function Definitions ---

def parse_args():
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train a GPT-2 like model using preprocessed Arrow datasets.")

    # --- REMOVED --evaluate_only flag, as this script is now only for training ---

    # === Essential Paths ===
    parser.add_argument("--train_dataset_path", type=str, required=True, help="Path to the training Arrow dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for checkpoints, logs, final model.")
    parser.add_argument("--model", type=str, default="gpt2", help="Base model identifier for tokenizer/config structure (e.g., 'gpt2').")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to RESUME training from.")

    # === Training Hyperparameters ===
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Train batch size per device.")
    # REMOVED --per_device_eval_batch_size (now handled by evaluate.py)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Steps for gradient accumulation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Peak learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay coefficient.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="LR scheduler type.", choices=["linear", "cosine", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="LR warmup steps.")

    # === Hardware & Precision ===
    parser.add_argument("--use_amp", action="store_true", help="Enable AMP training.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")

    # === Control & Reproducibility ===
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # === Logging & Saving ===
    parser.add_argument("--logging_steps", type=int, default=100, help="Log train metrics every X steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps.")

    # === Evaluation Triggering Control ===
    parser.add_argument("--eval_steps", type=int, default=500, help="Trigger evaluation job every X steps.")
    parser.add_argument("--submit_eval_script_path", type=str, required=True, help="Path to the Slurm script used to submit evaluation jobs (e.g., 'submit_eval_job.sh').")
    # Keep dataset paths needed to *pass* to the evaluation job
    parser.add_argument("--validation_dataset_path", type=str, default=None, help="Path to validation Arrow dataset (passed to eval job if standard eval enabled).")
    parser.add_argument("--priming_eval_dir_path", type=str, default=None, help="Directory containing priming CSVs (passed to eval job if priming eval enabled).")
    # Flags to control *whether* to trigger specific evaluations
    parser.add_argument("--trigger_standard_eval", action="store_true", default=False, help="Trigger standard evaluation jobs.")
    parser.add_argument("--trigger_priming_eval", action="store_true", default=False, help="Trigger priming evaluation jobs.")
    # Keep priming_eval_steps to control trigger frequency
    parser.add_argument("--priming_eval_steps", type=int, default=None, help="Trigger priming eval job every X steps. Defaults to --eval_steps.")
    # REMOVED arguments specific to how eval runs (batch size, sampling, etc.) - handled by evaluate.py args via submit_eval_job.sh

    # === Neptune Logging ===
    parser.add_argument("--neptune_project", type=str, default=None, help="Neptune project name. Disables if None.")
    parser.add_argument("--neptune_tags", type=str, nargs='+', default=None, help="Optional Neptune tags.")
    parser.add_argument("--neptune_run_name", type=str, default=None, help="Optional Neptune run name.")

    args = parser.parse_args()

    # Set defaults
    if args.priming_eval_steps is None: args.priming_eval_steps = args.eval_steps

    # Validation for Training
    if not args.train_dataset_path: parser.error("--train_dataset_path required for training.")
    if not args.output_dir: parser.error("--output_dir required for training.")
    if not Path(args.train_dataset_path).is_dir(): parser.error(f"Train dataset not found: {args.train_dataset_path}")
    if not Path(args.submit_eval_script_path).is_file(): parser.error(f"Evaluation submission script not found: {args.submit_eval_script_path}")
    if args.trigger_standard_eval and not args.validation_dataset_path: parser.error("--validation_dataset_path required if --trigger_standard_eval is set.")
    if args.trigger_priming_eval and not args.priming_eval_dir_path: parser.error("--priming_eval_dir_path required if --trigger_priming_eval is set.")
    if args.validation_dataset_path and not Path(args.validation_dataset_path).is_dir(): parser.error(f"Validation dataset not found: {args.validation_dataset_path}")
    if args.priming_eval_dir_path and not Path(args.priming_eval_dir_path).is_dir(): parser.error(f"Priming dir not found: {args.priming_eval_dir_path}")

    # Create output dir if not exists (only rank 0 should do this ideally, but safe enough here)
    if not Path(args.output_dir).exists():
         print(f"Warning: Output directory {args.output_dir} does not exist. It will be created.")
         # No error, will create later in setup

    return args

# --- Keep Helper Functions Needed for Training ---
# get_device, setup_logging, set_seed, setup_distributed, save_checkpoint, create_small_model_config
# REMOVE: evaluate_standard, run_priming_evaluation_on_directory, load_model_for_evaluation

def get_device():
    """Gets the appropriate device for PyTorch computations."""
    # (Keep implementation as in original train.py)
    import torch; import os
    if torch.backends.mps.is_available(): device = torch.device("mps"); print("Using MPS")
    elif torch.cuda.is_available():
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if local_rank >= torch.cuda.device_count(): local_rank = 0
            device = torch.device(f"cuda:{local_rank}"); print(f"Using CUDA GPU: {local_rank} - {torch.cuda.get_device_name(device)}")
        except Exception as e: print(f"Error setting CUDA: {e}. Using CPU."); device = torch.device("cpu")
    else: device = torch.device("cpu"); print("Using CPU")
    return device

def setup_logging(log_level=logging.INFO, rank=0):
    """Configures basic logging."""
    # (Keep implementation as in original train.py)
    global logger
    fmt = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s"
    dfmt = "%Y-%m-%d %H:%M:%S"
    level = log_level if rank == 0 else logging.CRITICAL + 1
    logging.basicConfig(level=level, format=fmt, datefmt=dfmt, force=True)
    logger = logging.getLogger(__name__)
    if rank == 0: logger.info("Logging setup complete (Rank 0).")
    else: logger.disabled = True

def set_seed(seed_value):
    """Sets random seeds."""
    # (Keep implementation as in original train.py)
    import random; import numpy as np; import torch; global logger
    random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)
    try:
        if not logger.disabled: logger.info(f"Set random seed: {seed_value}")
    except (NameError, AttributeError): print(f"Set random seed: {seed_value} (logger N/A or disabled)")

def setup_distributed(args):
    """Sets up DDP environment."""
    # (Keep implementation as in original train.py)
    import torch; import os; global logger
    is_dist, rank, world_size, local_rank = False, 0, 1, 0
    if 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1:
        is_dist = True
        try:
            rank = int(os.environ["RANK"]); world_size = int(os.environ["WORLD_SIZE"]); local_rank = int(os.environ["LOCAL_RANK"])
            if not torch.cuda.is_available(): raise RuntimeError("DDP needs CUDA.")
            torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
            torch.cuda.set_device(local_rank)
            msg = f"DDP Init (Rank {rank}/{world_size}, LocalRank {local_rank}, Device: cuda:{local_rank})"
            try:
                if not logger.disabled: logger.info(msg)
            except (NameError, AttributeError): print(f"Info: {msg}")
            torch.distributed.barrier() # Sync after setup
        except Exception as e: print(f"ERROR: DDP init failed: {e}"); raise
    else:
        msg = "DDP not enabled."
        try:
            if not logger.disabled: logger.info(msg)
        except (NameError, AttributeError): print(f"Info: {msg}")
    return is_dist, rank, world_size, local_rank


def load_training_data(args, is_distributed, rank, world_size, data_collator):
    """Loads standard Arrow datasets specifically for training."""
    # Renamed from load_standard_data to reflect its purpose now
    global logger
    from datasets import load_from_disk
    from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
    train_dl, train_sampler = None, None

    if not args.train_dataset_path: logger.error("Missing training path."); raise ValueError("Missing path.")
    logger.info(f"Loading train data: {args.train_dataset_path}")
    try:
        ds = load_from_disk(args.train_dataset_path); logger.info(f"Train size: {len(ds):,}")
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if is_distributed else RandomSampler(ds)
        if rank == 0: logger.info(f"Using {'DistributedSampler' if is_distributed else 'RandomSampler'} for training.")
        train_dl = DataLoader(ds, sampler=sampler, batch_size=args.per_device_train_batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=data_collator)
        if rank == 0: logger.info("Train DataLoader created.")
        train_sampler = sampler # Return the sampler for set_epoch
    except Exception as e: logger.error(f"Train data load fail: {e}", exc_info=True); raise

    # --- REMOVED validation data loading ---

    return train_dl, train_sampler


def save_checkpoint(args, model, optimizer, lr_scheduler, scaler, epoch, global_step, rank, tokenizer=None):
    """Saves training state. Only rank 0 executes."""
    # (Keep implementation as in original train.py)
    if rank != 0: return
    global logger
    import torch; import numpy as np; import random; from pathlib import Path
    ckpt_dir = Path(args.output_dir) / f"checkpoint-{global_step}";
    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving checkpoint {global_step} to {ckpt_dir}")
    except OSError as e:
        logger.error(f"Failed to create checkpoint directory {ckpt_dir}: {e}")
        return # Cannot save if dir creation failed

    unwrapped_model = model.module if hasattr(model, 'module') else model
    state = {
        'model': unwrapped_model.state_dict(), 'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(), 'scaler': scaler.state_dict() if scaler.is_enabled() else None,
        'epoch': epoch, 'global_step': global_step, 'args': vars(args),
        'torch_rng_state': torch.get_rng_state(), 'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(), 'torch_cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }
    state_file = ckpt_dir / "training_state.pt"
    try: torch.save(state, state_file); logger.info(f"Saved training state to: {state_file}")
    except Exception as e: logger.error(f"Failed to save training state: {e}", exc_info=True)
    try:
        unwrapped_model.save_pretrained(ckpt_dir); logger.info(f"Saved model weights and config to {ckpt_dir}")
        if tokenizer: tokenizer.save_pretrained(ckpt_dir); logger.info(f"Saved tokenizer to {ckpt_dir}")
    except Exception as e: logger.error(f"Failed to save model/tokenizer using save_pretrained: {e}", exc_info=True)


def create_small_model_config(base_model_name: str, corpus_size_tag: str, tokenizer: PreTrainedTokenizer, logger: logging.Logger):
    """Creates a small model configuration based on a corpus size tag."""
    # (Keep implementation as in original train.py)
    try:
        config = AutoConfig.from_pretrained(base_model_name)
        if logger: logger.info(f"Loaded base config structure from: {base_model_name}")
    except Exception as e:
        if logger: logger.error(f"Failed to load base config '{base_model_name}': {e}", exc_info=True); raise

    tag = corpus_size_tag.lower()
    if tag == "10m": target_n_layer, target_n_head, target_n_embd = 4, 4, 256; logger.info("Applying config for '10m'.")
    elif tag == "100m": target_n_layer, target_n_head, target_n_embd = 6, 6, 384; logger.info("Applying config for '100m'.")
    else: raise ValueError(f"Unknown corpus_size_tag: '{corpus_size_tag}'. Expected '10m' or '100m'.")

    if hasattr(config, 'n_layer'): config.n_layer = target_n_layer
    if hasattr(config, 'n_head'): config.n_head = target_n_head
    if hasattr(config, 'n_embd'): config.n_embd = target_n_embd
    if hasattr(config, 'vocab_size'): config.vocab_size = len(tokenizer)
    else: logger.warning(f"Base config type {type(config)} might not have 'vocab_size'.")
    if hasattr(config, 'use_cache'): config.use_cache = False

    logger.info(f"Final SMALL config params: n_layer={getattr(config, 'n_layer', 'N/A')}, "
                f"n_head={getattr(config, 'n_head', 'N/A')}, n_embd={getattr(config, 'n_embd', 'N/A')}, "
                f"vocab_size={getattr(config, 'vocab_size', 'N/A')}")
    return config

# --- MODIFIED Train Epoch Function ---

def trigger_evaluation_job(args, checkpoint_dir: Path, global_step: int, rank: int):
    """Constructs and submits an sbatch job for evaluation."""
    if rank != 0: return # Only rank 0 submits jobs
    global logger

    eval_output_dir = checkpoint_dir / "eval_results" # Define where eval results should go
    try:
        # Create the directory where the eval job will write outputs and logs
        eval_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create evaluation output directory {eval_output_dir}: {e}. Skipping eval job submission.")
        return

    # --- Construct environment variables to pass via --export ---
    # Use ALL to pass the current environment, then add/override specifics
    export_vars = ["ALL"]
    export_vars.append(f"CKPT_PATH={checkpoint_dir.resolve()}") # Use absolute path
    export_vars.append(f"EVAL_OUT_DIR={eval_output_dir.resolve()}")
    export_vars.append(f"RUN_STD_EVAL={1 if args.trigger_standard_eval else 0}")
    export_vars.append(f"RUN_PRIME_EVAL={1 if args.trigger_priming_eval else 0}")
    export_vars.append(f"SEED={args.seed}") # Pass seed for consistent sampling in eval
    # Only add dataset paths if they are actually specified
    if args.validation_dataset_path: export_vars.append(f"VALID_DATA_PATH={Path(args.validation_dataset_path).resolve()}")
    if args.priming_eval_dir_path: export_vars.append(f"PRIME_DATA_PATH={Path(args.priming_eval_dir_path).resolve()}")
    # Add Neptune details if configured, so eval job can potentially log to the same run
    if args.neptune_project: export_vars.append(f"NEPTUNE_PROJECT={args.neptune_project}")
    if os.getenv('NEPTUNE_API_TOKEN'): export_vars.append(f"NEPTUNE_API_TOKEN={os.getenv('NEPTUNE_API_TOKEN')}")
    # If train script logs to Neptune, pass the run ID so eval can log to the same run
    global run # Access the global Neptune run object
    if run and hasattr(run, '_sys_id'): # Check if run exists and has an ID
        export_vars.append(f"NEPTUNE_RUN_ID={run['_sys_id'].fetch()}") # Fetch the run ID

    export_string = ",".join(export_vars)

    # --- Construct sbatch command ---
    job_name = f"eval_{args.neptune_run_name or 'job'}_{global_step}"
    sbatch_command = [
        "sbatch",
        f"--job-name={job_name}",
        f"--output={eval_output_dir}/slurm-%j.out",
        f"--error={eval_output_dir}/slurm-%j.err",
        # Add other SBATCH directives if needed (partition, gres, mem, time) - ideally read from config or args
        # Example: f"--partition={args.eval_partition}", f"--gres=gpu:{args.eval_gpus}"
        f"--export={export_string}",
        str(Path(args.submit_eval_script_path).resolve()) # Absolute path to the submission script
    ]

    logger.info(f"Submitting evaluation job for step {global_step}...")
    logger.debug(f"sbatch command: {' '.join(sbatch_command)}") # Log the full command for debugging

    try:
        # Using subprocess.run is generally safer than os.system
        result = subprocess.run(sbatch_command, capture_output=True, text=True, check=True)
        logger.info(f"Evaluation job submission successful for step {global_step}. Output:\n{result.stdout}")
    except FileNotFoundError:
        logger.error(f"Error: 'sbatch' command not found. Ensure Slurm is installed and in PATH.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error submitting evaluation job for step {global_step}. Return code: {e.returncode}")
        logger.error(f"sbatch stdout:\n{e.stdout}")
        logger.error(f"sbatch stderr:\n{e.stderr}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during evaluation job submission for step {global_step}: {e}", exc_info=True)


def train_epoch(args, model, optimizer, lr_scheduler, scaler, train_dataloader,
                train_sampler, epoch, global_step, device, rank, world_size, run, tokenizer, max_train_steps):
    """
    Runs one training epoch.
    Saves checkpoints and triggers evaluation jobs periodically.
    """
    global logger
    import torch; from torch.utils.data import DistributedSampler; from tqdm.auto import tqdm; import math; import sys; import gc
    from torch.nn.utils import clip_grad_norm_
    from torch.cuda.amp import autocast

    model.train()
    is_distributed = train_sampler is not None and isinstance(train_sampler, DistributedSampler)
    if is_distributed:
        try: train_sampler.set_epoch(epoch)
        except AttributeError: logger.warning("Train sampler does not have set_epoch method.")

    disable_tqdm = rank != 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}", leave=True, disable=disable_tqdm, position=0)
    total_loss_since_logging, steps_since_logging = 0.0, 0
    last_logged_loss = float('inf')

    for step, batch in enumerate(progress_bar):
        # --- Training Step Logic (Mostly unchanged) ---
        try:
            batch_on_device = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        except RuntimeError as e:
            logger.error(f"Error moving training batch to device {device} at step {global_step}: {e}"); optimizer.zero_grad(set_to_none=True); continue
        try:
            amp_enabled = args.use_amp and device.type == 'cuda'
            with autocast(enabled=amp_enabled): outputs = model(**batch_on_device); loss = outputs.loss
            if loss is None: logger.error(f"Loss returned as None at step {global_step}. Skipping."); optimizer.zero_grad(set_to_none=True); continue
            if not torch.isfinite(loss): logger.warning(f"Non-finite loss ({loss.item()}) before scaling at step {global_step}. Skipping update."); optimizer.zero_grad(set_to_none=True); continue
            scaled_loss = loss / args.gradient_accumulation_steps; current_loss_value = loss.item()
        except Exception as e: logger.error(f"Error during training forward pass at step {global_step}: {e}", exc_info=True); optimizer.zero_grad(set_to_none=True); continue

        if amp_enabled: scaler.scale(scaled_loss).backward()
        else: scaled_loss.backward()
        total_loss_since_logging += current_loss_value; steps_since_logging += 1; last_logged_loss = current_loss_value

        # --- Optimizer Step, Scheduler Step, Grad Clipping (Unchanged) ---
        if (step + 1) % args.gradient_accumulation_steps == 0:
            try:
                if args.max_grad_norm and args.max_grad_norm > 0:
                    if scaler.is_enabled(): scaler.unscale_(optimizer)
                    params_to_clip = model.module.parameters() if hasattr(model, 'module') else model.parameters()
                    clip_grad_norm_(params_to_clip, args.max_grad_norm)
                if scaler.is_enabled(): scaler.step(optimizer); scaler.update()
                else: optimizer.step()
                lr_scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step += 1
            except Exception as e: logger.error(f"Error during optimizer step/clipping/lr_scheduler at step {global_step}: {e}", exc_info=True); optimizer.zero_grad(set_to_none=True)

            # --- Logging Step (Unchanged, only logs training metrics) ---
            if rank == 0 and global_step % args.logging_steps == 0 and steps_since_logging > 0:
                 avg_loss = total_loss_since_logging / steps_since_logging
                 lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, 'get_last_lr') and lr_scheduler.get_last_lr() else optimizer.param_groups[0]['lr']
                 logger.info(f"Epoch {epoch+1} | Step {global_step}: Avg Train Loss = {avg_loss:.4f}, LR = {lr:.6e}")
                 if run: # Log to Neptune
                     try:
                         if math.isfinite(avg_loss): run["train/step_loss"].append(avg_loss, step=global_step)
                         if math.isfinite(lr): run["train/learning_rate"].append(lr, step=global_step)
                         if torch.cuda.is_available(): run["train/gpu_mem_alloc_gb"].append(torch.cuda.memory_allocated(device)/1e9, step=global_step)
                         if scaler.is_enabled(): run["train/grad_scale"].append(scaler.get_scale(), step=global_step)
                     except Exception as e: logger.warning(f"Neptune train logging failed at step {global_step}: {e}")
                 total_loss_since_logging, steps_since_logging = 0.0, 0

            # --- Save Checkpoint Step ---
            # Save checkpoint BEFORE potentially triggering eval job that needs it
            if global_step > 0 and global_step % args.save_steps == 0:
                if is_distributed: torch.distributed.barrier()
                save_checkpoint(args, model, optimizer, lr_scheduler, scaler, epoch, global_step, rank, tokenizer)
                if is_distributed: torch.distributed.barrier()

            # --- Evaluation Trigger Step ---
            trigger_eval_now = (args.trigger_standard_eval or args.trigger_priming_eval) and global_step > 0 and \
                               (global_step % args.eval_steps == 0 or global_step % args.priming_eval_steps == 0)

            if trigger_eval_now:
                 # Ensure checkpoint is saved (redundant if save_steps aligns, but safe)
                 # Check if the checkpoint directory for the *current* step actually exists before triggering
                 checkpoint_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                 if rank == 0 and not checkpoint_dir.is_dir():
                     logger.warning(f"Checkpoint dir {checkpoint_dir} not found. Saving checkpoint now before triggering eval.")
                     if is_distributed: torch.distributed.barrier()
                     save_checkpoint(args, model, optimizer, lr_scheduler, scaler, epoch, global_step, rank, tokenizer)
                     if is_distributed: torch.distributed.barrier()

                 # Check again if saving worked (rank 0)
                 if rank == 0 and checkpoint_dir.is_dir():
                     logger.info(f"--- Triggering Evaluation Job Submission @ Step {global_step} ---")
                     trigger_evaluation_job(args, checkpoint_dir, global_step, rank)
                     # No waiting here, training continues immediately
                 elif rank == 0:
                     logger.error(f"Failed to find or save checkpoint {checkpoint_dir}. Cannot trigger evaluation job.")


        # Update progress bar on rank 0
        if rank == 0: progress_bar.set_postfix({"loss": f"{last_logged_loss:.4f}", "step": global_step})
        # Check max steps
        if max_train_steps > 0 and global_step >= max_train_steps:
             if rank == 0: logger.info(f"Maximum training steps ({max_train_steps}) reached. Finishing epoch {epoch+1}.")
             break # Exit inner loop

    # End of Epoch
    if rank == 0: progress_bar.close(); logger.info(f"--- Epoch {epoch+1} Finished (Reached Step {global_step}) ---")
    return global_step

# --- Main Function (Modified) ---

def main():
    """Main function to parse arguments, set up, and run training."""
    # --- Keep Imports ---
    # ... (standard, ML imports as before) ...
    global run # Declare run as global for access in trigger_evaluation_job

    args = parse_args()
    is_distributed, rank, world_size, local_rank = setup_distributed(args)
    setup_logging(rank=rank)
    global logger
    logger = logging.getLogger(__name__)

    if rank == 0: logger.info(f"***** Starting Training Script (Decoupled Eval) *****")
    if rank == 0: logger.info(f"Running with Arguments: {vars(args)}")

    # Setup Device and Seed
    device = get_device()
    set_seed(args.seed + rank)

    # Neptune Setup (Rank 0 Only, Unchanged)
    run = None
    if rank == 0 and NEPTUNE_AVAILABLE and args.neptune_project:
        try:
            run = neptune.init_run(project=args.neptune_project, name=args.neptune_run_name, tags=args.neptune_tags)
            args_log = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            run["parameters"] = args_log
            logger.info(f"Neptune logging enabled. Run URL: {run.get_url()}")
        except Exception as e:
            logger.error(f"Neptune initialization failed: {e}. Neptune logging disabled."); run = None
    elif rank == 0: logger.info("Neptune logging disabled.")

    # === Model & Tokenizer Loading (For Training) ===
    model, tokenizer, config = None, None, None
    try:
        if rank == 0:
            logger.info(f"TRAINING MODE: Initializing NEW small model (10m config) from scratch.")
            logger.info(f"Loading tokenizer specified by --model: {args.model}")
        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token: tokenizer.pad_token = tokenizer.eos_token; logger.info(f"Set pad_token=eos_token ('{tokenizer.eos_token}')")
            else: added = tokenizer.add_special_tokens({'pad_token': '[PAD]'}); logger.warning(f"Added pad_token '[PAD]' ({added} new token(s)).")
        # Create small model config
        config = create_small_model_config(base_model_name=args.model, corpus_size_tag="10m", tokenizer=tokenizer, logger=logger)
        # Initialize model
        logger.info("Initializing GPT2LMHeadModel with random weights using the '10m' config...")
        model = GPT2LMHeadModel(config=config)
        logger.info("Model initialized with random weights.")
        # Move model to device
        model.to(device)
        logger.info(f"Initialized NEW TRAIN model successfully on device {device} (Rank {rank})")
        # Wrap with DDP if needed
        if is_distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
            if rank == 0: logger.info(f"Model wrapped with DistributedDataParallel (Rank {rank}).")
    except Exception as e:
        logger.critical(f"Fatal Error: Model or Tokenizer loading/initialization failed: {e}", exc_info=True)
        if rank == 0 and run: try: run["error/message"] = f"Model/Tokenizer load failed: {e}"; run.stop() except Exception: pass
        sys.exit(1)

    # === Training Data Loading ===
    train_dataloader, train_sampler = None, None
    try:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        train_dataloader, train_sampler = load_training_data(args, is_distributed, rank, world_size, data_collator)
        if rank == 0: logger.info(f"Loaded training dataset with {len(train_dataloader.dataset):,} samples.")
    except Exception as e:
        logger.critical(f"Fatal Error: Failed to load training data: {e}", exc_info=True)
        if rank == 0 and run: try: run["error/message"] = f"Training data load failed: {e}"; run.stop() except Exception: pass
        sys.exit(1)

    # === Optimizer, Scheduler, Scaler Setup (Unchanged) ===
    if rank == 0: logger.info(f"Setting up Optimizer, LR Scheduler, Grad Scaler (Rank {rank})...")
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    named_params = model.module.named_parameters() if is_distributed else model.named_parameters()
    optimizer_grouped_parameters = [
        {"params": [p for n, p in named_params if not any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": args.weight_decay},
        {"params": [p for n, p in named_params if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
    ]
    total_params = sum(p.numel() for p in model.parameters()); num_trainable_params = sum(p.numel() for group in optimizer_grouped_parameters for p in group['params'])
    if rank == 0: logger.info(f"Initialized Small Model Parameters: Total={total_params:,}, Trainable={num_trainable_params:,}")
    if num_trainable_params == 0: logger.critical("No trainable parameters found."); sys.exit(1)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) if len(train_dataloader) > 0 else 0
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch if num_update_steps_per_epoch > 0 else 0
    if rank == 0: logger.info(f"Estimated total training steps: {max_train_steps:,}")
    eff_warmup_steps = min(args.num_warmup_steps, max_train_steps) if max_train_steps > 0 else 0; args.num_warmup_steps = eff_warmup_steps
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=max_train_steps)
    scaler_enabled = args.use_amp and device.type == 'cuda'; scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    if args.use_amp and not scaler_enabled and rank == 0: logger.warning("AMP requested but CUDA not available. Disabled.")
    if rank == 0: logger.info(f"AMP enabled: {scaler.is_enabled()}.")

    # === Resume from Checkpoint Logic (Unchanged) ===
    start_epoch, global_step, resumed_from_checkpoint = 0, 0, False
    if args.checkpoint_path:
        state_file = Path(args.checkpoint_path) / "training_state.pt"
        if state_file.is_file():
            if rank == 0: logger.info(f"Attempting to load checkpoint state from: {state_file}")
            try:
                checkpoint_state = torch.load(state_file, map_location=device, weights_only=False) # Set weights_only based on needs
                model_to_load = model.module if hasattr(model, 'module') else model
                missing, unexpected = model_to_load.load_state_dict(checkpoint_state['model'], strict=False)
                if rank == 0:
                    if missing: logger.warning(f"MISSING keys loading model state: {missing}")
                    if unexpected: logger.warning(f"UNEXPECTED keys loading model state: {unexpected}")
                if 'optimizer' in checkpoint_state: optimizer.load_state_dict(checkpoint_state['optimizer'])
                if 'lr_scheduler' in checkpoint_state: lr_scheduler.load_state_dict(checkpoint_state['lr_scheduler'])
                if 'scaler' in checkpoint_state and checkpoint_state['scaler'] and scaler.is_enabled(): scaler.load_state_dict(checkpoint_state['scaler'])
                start_epoch = checkpoint_state.get('epoch', 0) + 1; global_step = checkpoint_state.get('global_step', 0); resumed_from_checkpoint = True
                # Restore RNG states if present
                try:
                    if 'torch_rng_state' in checkpoint_state: torch.set_rng_state(checkpoint_state['torch_rng_state'].cpu())
                    if device.type == 'cuda' and 'torch_cuda_rng_state_all' in checkpoint_state and checkpoint_state['torch_cuda_rng_state_all']: torch.cuda.set_rng_state_all(checkpoint_state['torch_cuda_rng_state_all'])
                    if 'numpy_rng_state' in checkpoint_state: np.random.set_state(checkpoint_state['numpy_rng_state'])
                    if 'python_rng_state' in checkpoint_state: random.setstate(checkpoint_state['python_rng_state'])
                    if rank == 0: logger.info(f"Restored RNG states from checkpoint.")
                except Exception as rng_e: logger.warning(f"Could not restore RNG states: {rng_e}")

                del checkpoint_state; gc.collect(); torch.cuda.empty_cache()
                if rank == 0: logger.info(f"Resuming training from Epoch {start_epoch}, Global Step {global_step}")
                if is_distributed: torch.distributed.barrier()
            except Exception as e:
                logger.error(f"Failed to load checkpoint state: {e}", exc_info=True); start_epoch, global_step, resumed_from_checkpoint = 0, 0, False
        else: logger.warning(f"Checkpoint specified ({args.checkpoint_path}), but training_state.pt not found. Starting from scratch.")
    else: logger.info("No checkpoint specified, starting training from scratch.")

    # --- REMOVED Initial Evaluation block ---

    # Training Start Logging (Rank 0 Only, Unchanged)
    if rank == 0:
        logger.info("***** Training Configuration *****")
        logger.info(f"   Model Config: '10m' (Small, From Scratch)")
        # ... (Log other params like batch size, LR, etc.) ...
        eff_batch_size = args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps
        logger.info(f"   Effective Global Batch Size: {eff_batch_size}")
        logger.info(f"   Logging Steps: {args.logging_steps}")
        logger.info(f"   Save Checkpoint Steps: {args.save_steps}")
        logger.info(f"   Trigger Standard Eval: {args.trigger_standard_eval}, Freq: {args.eval_steps} steps")
        logger.info(f"   Trigger Priming Eval: {args.trigger_priming_eval}, Freq: {args.priming_eval_steps} steps")
        logger.info(f"   Eval Submission Script: {args.submit_eval_script_path}")
        logger.info(f"   Output Directory: {args.output_dir}")
        logger.info(f"   AMP Enabled: {scaler.is_enabled()}")

    # === Training Loop ===
    if is_distributed: torch.distributed.barrier()
    training_start_time = time.time()
    final_global_step = global_step

    try:
        for epoch in range(start_epoch, args.num_train_epochs):
            if rank == 0: logger.info(f"--- Starting Epoch {epoch + 1}/{args.num_train_epochs} ---")
            model.train()
            final_global_step = train_epoch(
                args=args, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, scaler=scaler,
                train_dataloader=train_dataloader, train_sampler=train_sampler,
                epoch=epoch, global_step=final_global_step, device=device, rank=rank, world_size=world_size,
                run=run, tokenizer=tokenizer, max_train_steps=max_train_steps,
            )
            if max_train_steps > 0 and final_global_step >= max_train_steps:
                 if rank == 0: logger.info(f"Max training steps ({max_train_steps}) reached. Stopping training.")
                 break # Exit epoch loop

        # Training finished
        training_duration = time.time() - training_start_time
        if rank == 0:
            logger.info(f"***** Training Finished *****")
            logger.info(f"   Total Training Time: {training_duration:.2f} seconds")
            logger.info(f"   Final Global Step: {final_global_step}")

        # Final Saving (Rank 0 Only, Unchanged)
        if rank == 0:
            final_model_path = Path(args.output_dir) / "final_model"
            logger.info(f"Saving final trained model, tokenizer, and config to {final_model_path}")
            try:
                final_model_path.mkdir(parents=True, exist_ok=True)
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(final_model_path)
                tokenizer.save_pretrained(final_model_path)
                args_dict_final = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
                with open(final_model_path / "training_args.json", "w", encoding='utf-8') as f: json.dump(args_dict_final, f, indent=4)
                logger.info(f"Final model assets saved successfully.")
                if run: # Upload to Neptune
                    try:
                        run["final_model"].upload(str(final_model_path))
                        run["training/duration_seconds"] = training_duration
                        run["training/final_global_step"] = final_global_step
                        logger.info("Uploaded final model directory and training summary to Neptune.")
                    except Exception as e: logger.warning(f"Failed to upload final model to Neptune: {e}")
            except Exception as e: logger.error(f"Failed to save final model assets: {e}", exc_info=True)

        # --- REMOVED Final Evaluation block ---
        # Final evaluation should be triggered like any other evaluation step if desired,
        # or run manually using evaluate.py on the final_model directory.

    except KeyboardInterrupt: logger.warning("Training interrupted by user.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during the training loop (Rank {rank}): {e}", exc_info=True)
        if rank == 0 and run: try: run["error/message"] = f"Training loop error: {e}"; run["error/traceback"] = traceback.format_exc() except Exception: pass
    finally:
        # Clean up DDP and Neptune
        if is_distributed: torch.distributed.destroy_process_group()
        if rank == 0 and run: try: run.stop() except Exception as ne_stop: logger.error(f"Neptune stop failed: {ne_stop}")
        logger.info(f"Training script finished (Rank {rank}).")


# Define a fallback tqdm function in case the import fails
def _fallback_tqdm(iterable, *args, **kwargs):
    """Placeholder for tqdm if it's not installed."""
    # You could add a simple print statement here if you want some indication
    # print("Warning: tqdm not installed, progress bars disabled.")
    return iterable

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    try:
        from tqdm.auto import tqdm # Try importing the real tqdm
    except ImportError:
        print("Warning: tqdm library not found. Progress bars will be disabled.")
        tqdm = _fallback_tqdm # Assign the fallback function to the name 'tqdm'

    # Now call main, which will use either the real tqdm or the fallback
    main()