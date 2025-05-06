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
import numpy as np # <<< Added numpy
import json
import time
import traceback
import gc
import subprocess # For submitting sbatch jobs OR running local eval

# --- ML/data library imports ---
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Dataset # Added Dataset
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
    sys.modules['neptune'] = None
    NEPTUNE_AVAILABLE = False
    neptune = None

# --- Logger Setup ---
logger = None

# --- Helper Function Definitions ---

def parse_args():
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train a GPT-2 like model using preprocessed Arrow datasets.")

    # === Essential Paths ===
    parser.add_argument("--train_dataset_path", type=str, required=True, help="Path to the training Arrow dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for checkpoints, logs, final model.")
    parser.add_argument("--model", type=str, default="gpt2", help="Base model identifier for tokenizer/config structure (e.g., 'gpt2').")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to RESUME training from.")

    # === Training Hyperparameters ===
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total training epochs (used if max_steps <= 0).")
    # --- MODIFIED: max_steps is optional again ---
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum number of optimizer steps to train for. Overrides num_train_epochs if > 0.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Train batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Steps for gradient accumulation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Peak learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay coefficient.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="LR scheduler type.", choices=["linear", "cosine", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="LR warmup steps.")
    parser.add_argument("--model_size", type=str, default="10m", help="Model size tag for small model config (e.g., '10m' or '100m').")

    # === Hardware & Precision ===
    parser.add_argument("--use_amp", action="store_true", help="Enable AMP training (recommended for Flash Attention).")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--use_flash_attention_2", action="store_true", help="Attempt to use Flash Attention 2 (requires installation and compatible hardware).")

    # === Control & Reproducibility ===
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # === Logging & Saving ===
    parser.add_argument("--logging_steps", type=int, default=100, help="Log train metrics every X steps.")
    # --- save_steps argument remains removed ---

    # === Evaluation Triggering Control ===
    # --- MODIFIED: eval_steps/priming_eval_steps are now optional if eval tied to checkpoints ---
    parser.add_argument("--eval_steps", type=int, default=0, help="Trigger evaluation every X steps (if > 0 and eval_after_save is False).")
    parser.add_argument("--priming_eval_steps", type=int, default=0, help="Trigger priming eval every X steps (if > 0 and eval_after_save is False). Defaults to eval_steps.")
    parser.add_argument("--eval_after_save", action="store_true", default=True, help="Trigger evaluation immediately after each checkpoint save (overrides eval_steps/priming_eval_steps).")

    parser.add_argument("--local_eval", action="store_true", help="Run evaluation script locally as subprocess instead of submitting Slurm job.")
    parser.add_argument("--submit_eval_script_path", type=str, default=None, help="Path to the Slurm script for submitting eval jobs (required if not --local_eval).")
    parser.add_argument("--evaluate_script_path", type=str, default="src/evaluate.py", help="Path to evaluate.py (required if --local_eval).")
    parser.add_argument("--validation_dataset_path", type=str, default=None, help="Path to validation Arrow dataset (passed to eval job/script if standard eval enabled).")
    parser.add_argument("--priming_eval_dir_path", type=str, default=None, help="Directory containing priming CSVs (passed to eval job/script if priming eval enabled).")
    parser.add_argument("--trigger_standard_eval", action="store_true", default=False, help="Trigger/run standard evaluation.")
    parser.add_argument("--trigger_priming_eval", action="store_true", default=False, help="Trigger/run priming evaluation.")

    # === Neptune Logging ===
    parser.add_argument("--neptune_project", type=str, default=None, help="Neptune project name.")
    parser.add_argument("--neptune_tags", type=str, nargs='+', default=None, help="Optional Neptune tags.")
    parser.add_argument("--neptune_run_name", type=str, default=None, help="Optional Neptune run name.")

    args = parser.parse_args()

    # Set defaults
    # --- MODIFIED: Default priming_eval_steps logic ---
    if args.priming_eval_steps is None or args.priming_eval_steps <= 0:
         args.priming_eval_steps = args.eval_steps

    # --- REMOVED: Validation check for max_steps > 0 ---

    # Validation for Training
    if not args.train_dataset_path: parser.error("--train_dataset_path required for training.")
    if not args.output_dir: parser.error("--output_dir required for training.")
    if not Path(args.train_dataset_path).is_dir(): parser.error(f"Train dataset not found: {args.train_dataset_path}")
    if not args.local_eval and not args.submit_eval_script_path: parser.error("--submit_eval_script_path required unless --local_eval is set.")
    if not args.local_eval and args.submit_eval_script_path and not Path(args.submit_eval_script_path).is_file(): parser.error(f"Evaluation submission script not found: {args.submit_eval_script_path}")
    if args.local_eval and not Path(args.evaluate_script_path).is_file(): parser.error(f"Evaluation script not found: {args.evaluate_script_path}")
    if args.trigger_standard_eval and not args.validation_dataset_path: parser.error("--validation_dataset_path required if --trigger_standard_eval is set.")
    if args.trigger_priming_eval and not args.priming_eval_dir_path: parser.error("--priming_eval_dir_path required if --trigger_priming_eval is set.")
    if args.validation_dataset_path and not Path(args.validation_dataset_path).is_dir(): parser.error(f"Validation dataset not found: {args.validation_dataset_path}")
    if args.priming_eval_dir_path and not Path(args.priming_eval_dir_path).is_dir(): parser.error(f"Priming dir not found: {args.priming_eval_dir_path}")
    # Check for logger existence before using it in validation
    if args.use_flash_attention_2 and not args.use_amp and logger:
         logger.warning("Flash Attention 2 is requested but AMP is not enabled (--use_amp). Performance may be suboptimal or compatibility issues may arise.")
    elif args.use_flash_attention_2 and not args.use_amp:
         print("Warning: Flash Attention 2 is requested but AMP is not enabled (--use_amp). Performance may be suboptimal or compatibility issues may arise.")

    # Validate eval trigger settings
    if not args.eval_after_save and (args.eval_steps <= 0 and args.priming_eval_steps <= 0):
         print("Warning: --eval_after_save is False, but --eval_steps and --priming_eval_steps are not positive. No evaluation will be triggered.")
    elif args.eval_after_save and (args.eval_steps > 0 or args.priming_eval_steps > 0):
         print("Info: --eval_after_save is True. --eval_steps and --priming_eval_steps will be ignored. Evaluation will trigger after each save.")


    if not Path(args.output_dir).exists(): print(f"Warning: Output directory {args.output_dir} does not exist. It will be created.")

    return args


# --- Helper Functions (get_device, setup_logging, set_seed, setup_distributed, load_training_data, save_checkpoint, create_small_model_config, run_or_trigger_evaluation) ---
# (Implementations remain largely the same)
def get_device():
    """Gets the appropriate device for PyTorch computations."""
    import torch; import os
    if torch.backends.mps.is_available(): device = torch.device("mps"); print("Using MPS")
    elif torch.cuda.is_available():
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if local_rank >= torch.cuda.device_count(): local_rank = 0
            device = torch.device(f"cuda:{local_rank}"); print(f"Using CUDA GPU: {local_rank} - {torch.cuda.get_device_name(device)}")
        except Exception as e:
            print(f"Error setting CUDA: {e}. Using CPU.")
            device = torch.device("cpu")
    else: device = torch.device("cpu"); print("Using CPU")
    return device

def setup_logging(log_level=logging.INFO, rank=0):
    """Configures basic logging."""
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
    import random; import numpy as np; import torch; global logger
    random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)
    try:
        # Check logger is initialized and not disabled before using
        if logger and not logger.disabled: logger.info(f"Set random seed: {seed_value}")
    except NameError: # Handle case where logger might not be globally defined yet
        print(f"Set random seed: {seed_value} (logger not available)")

def setup_distributed(args):
    """Sets up DDP environment."""
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
                if logger and not logger.disabled: logger.info(msg)
            except NameError:
                 print(f"Info: {msg} (logger not available)")
            torch.distributed.barrier() # Sync after setup
        except Exception as e:
            print(f"ERROR: DDP init failed: {e}")
            raise
    else:
        msg = "DDP not enabled."
        try:
            if logger and not logger.disabled: logger.info(msg)
        except NameError:
            print(f"Info: {msg} (logger not available)")
    return is_dist, rank, world_size, local_rank


def load_training_data(args, is_distributed, rank, world_size, data_collator):
    """Loads standard Arrow datasets specifically for training."""
    global logger
    from datasets import load_from_disk, Dataset # Ensure Dataset is imported
    from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
    train_ds: Optional[Dataset] = None # Type hint
    train_dl: Optional[DataLoader] = None
    train_sampler: Optional[Union[DistributedSampler, RandomSampler]] = None

    if not args.train_dataset_path:
        logger.error("Missing training dataset path (--train_dataset_path).")
        raise ValueError("Missing training dataset path.")
    logger.info(f"Loading train data from: {args.train_dataset_path}")
    try:
        train_ds = load_from_disk(args.train_dataset_path)
        logger.info(f"Train dataset size: {len(train_ds):,} samples")

        if len(train_ds) == 0:
             logger.warning("Loaded training dataset is empty!")
             # Return None or raise error depending on desired behavior
             return None, None, None # Return None dataloader, sampler, dataset

        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if is_distributed else RandomSampler(train_ds)
        if rank == 0: logger.info(f"Using {'DistributedSampler' if is_distributed else 'RandomSampler'} for training.")

        train_dl = DataLoader(train_ds, sampler=sampler, batch_size=args.per_device_train_batch_size,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=data_collator,
                              persistent_workers=(True if args.num_workers > 0 else False))

        if rank == 0: logger.info("Train DataLoader created.")
        train_sampler = sampler # Return the sampler for set_epoch

    except FileNotFoundError:
         logger.error(f"Training dataset not found at path: {args.train_dataset_path}")
         raise
    except Exception as e:
        logger.error(f"Failed to load or process training data: {e}", exc_info=True)
        raise
    return train_dl, train_sampler, train_ds # Return dataset size info too


def save_checkpoint(args, model, optimizer, lr_scheduler, scaler, epoch, global_step, rank, tokenizer=None):
    """Saves training state. Only rank 0 executes."""
    if rank != 0: return
    global logger
    import torch; import numpy as np; import random; from pathlib import Path
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
        'epoch': epoch, 'global_step': global_step, 'args': vars(args), # Save args used
        'torch_rng_state': torch.get_rng_state(), 'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(), 'torch_cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }
    state_file = ckpt_dir / "training_state.pt"
    try:
        torch.save(state, state_file)
    except Exception as e:
        logger.error(f"Failed to save training state: {e}", exc_info=True)
        return # Don't proceed if state saving fails

    try:
        unwrapped_model.save_pretrained(ckpt_dir)
        if tokenizer:
            tokenizer.save_pretrained(ckpt_dir)
        logger.info(f"Completed saving model/tokenizer/state for step {global_step} to {ckpt_dir}")
    except Exception as e:
        logger.error(f"Failed to save model/tokenizer using save_pretrained for step {global_step}: {e}", exc_info=True)


def create_small_model_config(base_model_name: str, corpus_size_tag: str, tokenizer: PreTrainedTokenizer, logger: logging.Logger):
    """Creates a small model configuration based on a corpus size tag."""
    try:
        config = AutoConfig.from_pretrained(base_model_name)
        if logger: logger.info(f"Loaded base config structure from: {base_model_name}")
    except Exception as e:
        if logger: logger.error(f"Failed to load base config '{base_model_name}': {e}", exc_info=True)
        raise
    tag = corpus_size_tag.lower()
    if tag == "10m": target_n_layer, target_n_head, target_n_embd = 4, 4, 256; logger.info("Applying config for '10m'.")
    elif tag == "100m": target_n_layer, target_n_head, target_n_embd = 6, 6, 384; logger.info("Applying config for '100m'.")
    else: raise ValueError(f"Unknown corpus_size_tag: '{corpus_size_tag}'. Expected '10m' or '100m'.")
    if hasattr(config, 'n_layer'): config.n_layer = target_n_layer
    if hasattr(config, 'n_head'): config.n_head = target_n_head
    if hasattr(config, 'n_embd'): config.n_embd = target_n_embd
    if hasattr(config, 'vocab_size'): config.vocab_size = len(tokenizer)
    else: logger.warning(f"Base config type {type(config)} might not have 'vocab_size'.")
    if hasattr(config, 'use_cache'): config.use_cache = False # Recommended for training
    logger.info(f"Final SMALL config params: n_layer={getattr(config, 'n_layer', 'N/A')}, n_head={getattr(config, 'n_head', 'N/A')}, n_embd={getattr(config, 'n_embd', 'N/A')}, vocab_size={getattr(config, 'vocab_size', 'N/A')}")
    return config

# --- Modified Trigger/Run Evaluation Function ---
def run_or_trigger_evaluation(args, checkpoint_dir: Path, global_step: int, rank: int):
    """
    Triggers evaluation either via Slurm sbatch or by running evaluate.py
    locally as a subprocess, based on the --local_eval flag.
    """
    global logger, run # Moved global declarations to the top
    if rank != 0: return # Only rank 0 triggers/runs evaluation

    eval_output_dir = checkpoint_dir / "eval_results" # Define where eval results should go
    try:
        eval_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create evaluation output directory {eval_output_dir}: {e}. Skipping evaluation.")
        return

    if args.local_eval:
        # --- Run evaluate.py Locally ---
        logger.info(f"--- Running Local Evaluation for Step {global_step} ---")
        python_executable = sys.executable
        eval_script_path = Path(args.evaluate_script_path).resolve()

        eval_args_list = [
            str(python_executable), str(eval_script_path),
            "--checkpoint_path", str(checkpoint_dir.resolve()),
            "--output_dir", str(eval_output_dir.resolve()), # Eval output goes into subdir
            "--seed", str(args.seed),
            "--num_workers", str(args.num_workers),
            "--analyze_training_state", # Always analyze state when triggering eval
        ]
        if args.use_amp: eval_args_list.append("--use_amp")
        if args.use_flash_attention_2: eval_args_list.append("--eval_use_flash_attention_2")

        if args.trigger_standard_eval:
            eval_args_list.append("--run_standard_eval")
            if args.validation_dataset_path:
                eval_args_list.extend(["--validation_dataset_path", str(Path(args.validation_dataset_path).resolve())])
            else: logger.warning("Local standard eval requested but --validation_dataset_path not provided.")

        if args.trigger_priming_eval:
            eval_args_list.append("--run_priming_eval")
            if args.priming_eval_dir_path:
                eval_args_list.extend(["--priming_eval_dir_path", str(Path(args.priming_eval_dir_path).resolve())])
            else: logger.warning("Local priming eval requested but --priming_eval_dir_path not provided.")

        if args.neptune_project:
             eval_args_list.extend(["--neptune_project", args.neptune_project])
        if run and hasattr(run, '_sys_id'):
             try:
                 run_id = run['_sys_id'].fetch()
                 eval_args_list.extend(["--neptune_run_id", run_id])
             except Exception as e: logger.warning(f"Could not fetch Neptune run ID for local eval: {e}")

        logger.info(f"Executing local evaluation command:\n{' '.join(eval_args_list)}")
        eval_start_time = time.time()
        try:
            result = subprocess.run(eval_args_list, check=True, env=os.environ.copy())
            logger.info(f"Local evaluation process completed successfully in {time.time() - eval_start_time:.2f} seconds.")
        except FileNotFoundError: logger.error(f"Error: Python executable '{python_executable}' or script '{eval_script_path}' not found.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running local evaluation script for step {global_step}. Return code: {e.returncode}")
            logger.error("Check console output above for potential error messages from evaluate.py.")
        except Exception as e: logger.error(f"Unexpected error during local evaluation for step {global_step}: {e}", exc_info=True)

    else:
        # --- Submit Slurm Job ---
        logger.info(f"--- Triggering Slurm Evaluation Job Submission @ Step {global_step} ---")
        if not args.submit_eval_script_path:
            logger.error("Slurm evaluation requested, but --submit_eval_script_path not provided.")
            return

        export_vars = ["ALL"]
        export_vars.append(f"CKPT_PATH={checkpoint_dir.resolve()}")
        export_vars.append(f"EVAL_OUT_DIR={eval_output_dir.resolve()}")
        export_vars.append(f"RUN_STD_EVAL={1 if args.trigger_standard_eval else 0}")
        export_vars.append(f"RUN_PRIME_EVAL={1 if args.trigger_priming_eval else 0}")
        export_vars.append(f"SEED={args.seed}")
        export_vars.append(f"USE_AMP={1 if args.use_amp else 0}")
        export_vars.append(f"EVAL_USE_FLASH_ATTN_2={1 if args.use_flash_attention_2 else 0}")
        export_vars.append(f"ANALYZE_STATE={1}")
        if args.validation_dataset_path: export_vars.append(f"VALID_DATA_PATH={Path(args.validation_dataset_path).resolve()}")
        if args.priming_eval_dir_path: export_vars.append(f"PRIME_DATA_PATH={Path(args.priming_eval_dir_path).resolve()}")
        if args.neptune_project: export_vars.append(f"NEPTUNE_PROJECT={args.neptune_project}")
        if os.getenv('NEPTUNE_API_TOKEN'): export_vars.append(f"NEPTUNE_API_TOKEN={os.getenv('NEPTUNE_API_TOKEN')}")
        if run and hasattr(run, '_sys_id'):
             try: export_vars.append(f"NEPTUNE_RUN_ID={run['_sys_id'].fetch()}")
             except Exception as e: logger.warning(f"Could not fetch Neptune run ID for Slurm job: {e}")

        export_string = ",".join(export_vars)
        job_name = f"eval_{args.neptune_run_name or 'job'}_{global_step}"
        sbatch_command = [
            "sbatch", f"--job-name={job_name}",
            f"--output={eval_output_dir}/slurm-%j.out", f"--error={eval_output_dir}/slurm-%j.err",
            f"--export={export_string}", str(Path(args.submit_eval_script_path).resolve())
        ]
        logger.debug(f"sbatch command: {' '.join(sbatch_command)}")
        try:
            result = subprocess.run(sbatch_command, capture_output=True, text=True, check=True)
            logger.info(f"Slurm evaluation job submission successful for step {global_step}. Output:\n{result.stdout}")
        except FileNotFoundError: logger.error("Error: 'sbatch' command not found.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error submitting Slurm evaluation job for step {global_step}. Return code: {e.returncode}\nStdout:\n{e.stdout}\nStderr:\n{e.stderr}")
        except Exception as e: logger.error(f"Unexpected error during Slurm job submission for step {global_step}: {e}", exc_info=True)


# --- Modified train_epoch ---
def train_epoch(args, model, optimizer, lr_scheduler, scaler, train_dataloader,
                train_sampler, epoch, global_step, device, rank, world_size, run, tokenizer,
                max_train_steps, target_save_steps):
    """
    Runs one training epoch. Saves checkpoints and triggers evaluations based on schedule.
    """
    global logger
    import torch; from torch.utils.data import DistributedSampler; from tqdm.auto import tqdm; import math; import sys; import gc
    from torch.nn.utils import clip_grad_norm_
    from torch.amp import autocast

    model.train()
    is_distributed = train_sampler is not None and isinstance(train_sampler, DistributedSampler)
    if is_distributed:
        try: train_sampler.set_epoch(epoch)
        except AttributeError: logger.warning("Train sampler does not have set_epoch method.")

    disable_tqdm = rank != 0
    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) if len(train_dataloader) > 0 else 0
    progress_bar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}", leave=True, disable=disable_tqdm, position=0, initial=global_step % steps_per_epoch if steps_per_epoch > 0 else 0)

    total_loss_since_logging, steps_since_logging = 0.0, 0
    last_logged_loss = float('inf')
    micro_step_counter = 0

    for batch in train_dataloader:
        try:
            batch_on_device = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        except RuntimeError as e:
            logger.error(f"Error moving train batch to {device}: {e}")
            optimizer.zero_grad(set_to_none=True)
            continue

        try:
            amp_enabled = args.use_amp and device.type == 'cuda'
            amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16 if amp_enabled else torch.float32
            with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                outputs = model(**batch_on_device)
                loss = outputs.loss
            if loss is None: logger.error("Loss was None. Skipping step."); optimizer.zero_grad(set_to_none=True); continue
            if not torch.isfinite(loss): logger.warning(f"Non-finite loss ({loss.item()}) at step {global_step}. Skipping."); optimizer.zero_grad(set_to_none=True); micro_step_counter = 0; continue
            scaled_loss = loss / args.gradient_accumulation_steps
            current_loss_value = loss.item()
        except Exception as e:
            # Handle specific errors like Flash Attention issues or general errors
            if "BackendCompilerFailed" in str(e): logger.error(f"Forward pass failed: BackendCompilerFailed: {e}")
            elif "FlashAttention is not installed" in str(e): logger.error(f"Forward pass failed: Flash Attention not installed. {e}. Exiting."); sys.exit(1)
            elif "does not support Flash Attention" in str(e): logger.error(f"Forward pass failed: Hardware/environment does not support Flash Attention. {e}. Exiting."); sys.exit(1)
            else: logger.error(f"Forward pass error at step {global_step}: {e}", exc_info=True)
            optimizer.zero_grad(set_to_none=True); micro_step_counter = 0; continue

        try:
            if amp_enabled: scaler.scale(scaled_loss).backward()
            else: scaled_loss.backward()
            total_loss_since_logging += current_loss_value
            steps_since_logging += 1
            last_logged_loss = current_loss_value
        except Exception as e:
             logger.error(f"Backward pass error at step {global_step}: {e}", exc_info=True)
             optimizer.zero_grad(set_to_none=True); micro_step_counter = 0; continue

        micro_step_counter += 1

        if micro_step_counter % args.gradient_accumulation_steps == 0:
            perform_opt_step = True
            try:
                if args.max_grad_norm > 0:
                    if scaler.is_enabled(): scaler.unscale_(optimizer)
                    params_to_clip = model.module.parameters() if hasattr(model, 'module') else model.parameters()
                    clip_grad_norm_(params_to_clip, args.max_grad_norm)

                if scaler.is_enabled():
                    optimizer_status = scaler.step(optimizer)
                    scaler.update()
                    if optimizer_status is None: perform_opt_step = False; logger.warning(f"Optimizer step skipped due to invalid gradients at step {global_step}.")
                else: optimizer.step()

                if perform_opt_step: lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if rank == 0: progress_bar.update(1)

            except Exception as e:
                logger.error(f"Optimizer/Scheduler step error at global step {global_step}: {e}", exc_info=True)
                optimizer.zero_grad(set_to_none=True); perform_opt_step = False

            max_steps_reached = max_train_steps > 0 and global_step >= max_train_steps

            if perform_opt_step and rank == 0 and global_step % args.logging_steps == 0 and steps_since_logging > 0:
                 avg_loss = total_loss_since_logging / steps_since_logging
                 lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, 'get_last_lr') and lr_scheduler.get_last_lr() else optimizer.param_groups[0]['lr']
                 logger.info(f"Epoch {epoch+1} | Step {global_step}/{max_train_steps if max_train_steps > 0 else 'inf'}: Avg Loss = {avg_loss:.4f}, LR = {lr:.6e}")
                 if run:
                     try:
                         if math.isfinite(avg_loss): run["train/step_loss"].append(avg_loss, step=global_step)
                         if math.isfinite(lr): run["train/learning_rate"].append(lr, step=global_step)
                         if torch.cuda.is_available(): run["train/gpu_mem_alloc_gb"].append(torch.cuda.memory_allocated(device)/1e9, step=global_step)
                         if scaler.is_enabled(): run["train/grad_scale"].append(scaler.get_scale(), step=global_step)
                     except Exception as e: logger.warning(f"Neptune train log failed at step {global_step}: {e}")
                 total_loss_since_logging, steps_since_logging = 0.0, 0

            saved_checkpoint_this_step = False
            if perform_opt_step and global_step in target_save_steps:
                if is_distributed: torch.distributed.barrier()
                save_checkpoint(args, model, optimizer, lr_scheduler, scaler, epoch, global_step, rank, tokenizer)
                saved_checkpoint_this_step = True
                if is_distributed: torch.distributed.barrier()

            # --- Evaluation Trigger Logic ---
            trigger_eval_now = False
            if args.eval_after_save:
                # Trigger if checkpoint saved AND eval enabled
                trigger_eval_now = saved_checkpoint_this_step and (args.trigger_standard_eval or args.trigger_priming_eval)
            else:
                # Trigger based on eval_steps if not saving-based
                time_for_std_eval = args.trigger_standard_eval and args.eval_steps > 0 and global_step > 0 and global_step % args.eval_steps == 0
                time_for_prime_eval = args.trigger_priming_eval and args.priming_eval_steps > 0 and global_step > 0 and global_step % args.priming_eval_steps == 0
                trigger_eval_now = perform_opt_step and (time_for_std_eval or time_for_prime_eval)

            if trigger_eval_now:
                checkpoint_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                # Ensure checkpoint exists if triggering based on interval (might need saving)
                if not args.eval_after_save and rank == 0 and not checkpoint_dir.is_dir():
                     logger.warning(f"Checkpoint {checkpoint_dir} needed for interval eval trigger not found. Saving now.")
                     if is_distributed: torch.distributed.barrier()
                     save_checkpoint(args, model, optimizer, lr_scheduler, scaler, epoch, global_step, rank, tokenizer)
                     if is_distributed: torch.distributed.barrier()

                if rank == 0 and checkpoint_dir.is_dir():
                    if args.eval_after_save: logger.info(f"Triggering evaluation following checkpoint save at step {global_step}.")
                    else: logger.info(f"Triggering interval-based evaluation at step {global_step}.")
                    run_or_trigger_evaluation(args, checkpoint_dir, global_step, rank)
                elif rank == 0:
                    logger.error(f"Checkpoint {checkpoint_dir} not found. Cannot run/trigger evaluation for step {global_step}.")
            # --- End Evaluation Trigger Logic ---

            if rank == 0: progress_bar.set_postfix({"loss": f"{last_logged_loss:.4f}"})

            if max_steps_reached:
                 if rank == 0:
                     logger.info(f"Max steps ({max_train_steps}) reached at step {global_step}. Finishing epoch {epoch+1}.")
                     progress_bar.set_postfix({"loss": f"{last_logged_loss:.4f}", "status": "Max steps reached"})
                     progress_bar.close()
                 return global_step

    if rank == 0:
         progress_bar.close()
         logger.info(f"--- Epoch {epoch+1} Finished (Reached Step {global_step}) ---")
    return global_step


# --- Main Function (Modified Setup) ---
def main():
    """Main function to parse arguments, set up, and run training."""
    global run # Declare run as global for access

    args = parse_args()
    is_distributed, rank, world_size, local_rank = setup_distributed(args)
    setup_logging(rank=rank)
    global logger
    logger = logging.getLogger(__name__)

    if rank == 0: logger.info(f"***** Starting Training Script (Auto Checkpointing) *****")
    if rank == 0: logger.info(f"Running with Arguments: {vars(args)}")

    # Setup Device and Seed
    device = get_device()
    set_seed(args.seed + rank)

    # Neptune Setup (Rank 0 Only)
    run = None
    if rank == 0 and NEPTUNE_AVAILABLE and args.neptune_project:
        try:
            run = neptune.init_run(project=args.neptune_project, name=args.neptune_run_name, tags=args.neptune_tags)
            args_log = {}
            logger.info("Processing arguments for Neptune logging...")
            for k, v in vars(args).items():
                if isinstance(v, Path): args_log[k] = str(v)
                elif isinstance(v, list): args_log[k] = ','.join(map(str, v)) if v else "None"
                elif v is None: args_log[k] = "None"
                else: args_log[k] = v
            try:
                run["parameters"] = args_log
                logger.info("Successfully logged processed parameters to Neptune.")
            except Exception as neptune_log_e:
                logger.error(f"Failed to log processed parameters to Neptune: {neptune_log_e}")

            logger.info(f"Neptune logging enabled. Run URL: {run.get_url()}")
            run.sync()
            logger.info("Neptune run object synchronized.")
        except Exception as e:
            logger.error(f"Neptune init failed: {e}. Disabled.")
            run = None
    elif rank == 0:
        logger.info("Neptune logging disabled.")

    # === Tokenizer Loading (early) ===
    tokenizer = None
    try:
        if rank == 0: logger.info(f"Loading Tokenizer: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Set pad_token=eos_token")
            else:
                added = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.warning(f"Added pad_token ('[PAD]') to tokenizer ({added} new special token).")
    except Exception as e:
         logger.critical(f"Failed to load tokenizer '{args.model}': {e}", exc_info=True)
         sys.exit(1)

    # === Training Data Loading (Moved earlier) ===
    train_dataloader, train_sampler, train_dataset = None, None, None
    try:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        train_dataloader, train_sampler, train_dataset = load_training_data(args, is_distributed, rank, world_size, data_collator)
        if train_dataloader is None:
             # Handle case where dataset was empty
             logger.critical("Training dataloader could not be created (likely empty dataset). Exiting.")
             sys.exit(1)
        if rank == 0: logger.info(f"Loaded train dataset ({len(train_dataset):,} samples). Dataloader length: {len(train_dataloader)}")
    except Exception as e:
        logger.critical(f"Train data load failed: {e}", exc_info=True)
        sys.exit(1)

    # === Calculate Max Steps (if not provided) ===
    if args.max_steps <= 0:
        if args.num_train_epochs <= 0:
             logger.error("Must provide either --max_steps or a positive --num_train_epochs.")
             sys.exit(1)
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if num_update_steps_per_epoch == 0:
             logger.error("Calculated zero update steps per epoch (dataloader length or grad accum issue?). Cannot train.")
             sys.exit(1)
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if rank == 0: logger.info(f"Calculated max_steps based on epochs: {args.num_train_epochs} epochs * {num_update_steps_per_epoch} steps/epoch = {max_train_steps} steps.")
    else:
        max_train_steps = args.max_steps
        if rank == 0: logger.info(f"Using provided max_steps: {max_train_steps}")
    # --- END MAX STEPS CALCULATION ---

    # === Model Loading (after tokenizer) ===
    model, config = None, None
    try:
        if rank == 0: logger.info(f"Creating NEW small model config ({args.model_size})")
        config = create_small_model_config(base_model_name=args.model, corpus_size_tag=args.model_size, tokenizer=tokenizer, logger=logger)

        # --- FLASH ATTENTION INTEGRATION ---
        model_kwargs = {"config": config}
        attn_implementation = None
        if args.use_flash_attention_2:
            if device.type == 'cuda':
                try:
                    import flash_attn
                    attn_implementation = "flash_attention_2"
                    logger.info("Attempting to use Flash Attention 2 implementation.")
                except ImportError:
                    logger.warning("Flash Attention 2 requested but flash-attn library is not installed. Falling back.")
                    attn_implementation = None
                except Exception as e:
                    logger.warning(f"Flash Attention check failed: {e}. Falling back.")
                    attn_implementation = None
            else:
                logger.warning("Flash Attention 2 requested but no CUDA device found. Falling back.")
                attn_implementation = None

        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                 model_kwargs["torch_dtype"] = torch.bfloat16
                 if rank == 0: logger.info("Using bfloat16 dtype with Flash Attention 2.")
            else:
                 model_kwargs["torch_dtype"] = torch.float16
                 if rank == 0: logger.info("Using float16 dtype with Flash Attention 2.")

        model = GPT2LMHeadModel(**model_kwargs)
        # Resize embeddings again in case tokenizer changed vocab size (e.g. pad token)
        model.resize_token_embeddings(len(tokenizer))
        # --- END FLASH ATTENTION INTEGRATION ---

        if rank == 0:
            try:
                impl = getattr(model.config, '_attn_implementation', 'default')
                logger.info(f"Model configured with attention implementation: {impl}")
            except AttributeError: pass

        logger.info("Model initialized.")
        model.to(device); logger.info(f"Moved model to {device} (Rank {rank})")

    except Exception as e:
        logger.critical(f"Model init failed: {e}", exc_info=True)
        sys.exit(1)


    # === Optimizer, Scheduler, Scaler Setup (Uses max_train_steps) ===
    if rank == 0: logger.info(f"Setting up Optimizer, LR Scheduler, Grad Scaler...")
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    named_params_func = model.named_parameters # Before DDP wrapping
    optimizer_grouped_parameters = [
        {"params": [p for n, p in named_params_func() if not any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": args.weight_decay},
        {"params": [p for n, p in named_params_func() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0}, ]
    total_params=sum(p.numel() for p in model.parameters()); num_trainable=sum(p.numel() for g in optimizer_grouped_parameters for p in g['params'])
    if rank == 0: logger.info(f"Initial Model Params: Total={total_params:,}, Trainable={num_trainable:,}")
    if num_trainable == 0: logger.critical("No trainable params."); sys.exit(1)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # --- LR Scheduler setup (uses final max_train_steps) ---
    eff_warmup = min(args.num_warmup_steps, max_train_steps) if max_train_steps > 0 else 0
    if rank == 0: logger.info(f"Effective warmup steps: {eff_warmup}")
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type, optimizer=optimizer,
        num_warmup_steps=eff_warmup, num_training_steps=max_train_steps
    )

    # GradScaler setup
    scaler_enabled = args.use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=scaler_enabled)
    if args.use_amp and not scaler_enabled and rank == 0: logger.warning("AMP requested but CUDA unavailable.")
    if rank == 0: logger.info(f"AMP enabled: {scaler.is_enabled()}.")

    # === Calculate Checkpoint Schedule (Uses max_train_steps) ===
    target_save_steps = set()
    num_linear_checkpoints = 144
    log_steps = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}
    target_save_steps.update(s for s in log_steps if s <= max_train_steps)
    if max_train_steps > 0 and num_linear_checkpoints > 0:
        linear_steps = np.linspace(0, max_train_steps, num=num_linear_checkpoints, dtype=int)
        target_save_steps.update(linear_steps)
        target_save_steps.add(max_train_steps) # Ensure final step is included

    if rank == 0:
         sorted_steps = sorted(list(target_save_steps))
         logger.info(f"Calculated {len(target_save_steps)} target checkpoint steps.")
         log_preview_count = 15
         if len(sorted_steps) <= log_preview_count * 2: logger.info(f"  Steps: {sorted_steps}")
         else: logger.info(f"  Steps (Preview): {sorted_steps[:log_preview_count]} ... {sorted_steps[-log_preview_count:]}")
    # --- End Checkpoint Schedule Calculation ---

    # === Resume from Checkpoint Logic ===
    start_epoch, global_step, resumed_from_checkpoint = 0, 0, False
    if args.checkpoint_path:
        state_file = Path(args.checkpoint_path) / "training_state.pt"
        if state_file.is_file():
            if rank == 0: logger.info(f"Attempting to load checkpoint state: {state_file}")
            try:
                ckpt = torch.load(state_file, map_location=device, weights_only=False)
                model_to_load = model # Before DDP wrapping
                ckpt_args = ckpt.get('args', {})
                ckpt_used_flash = ckpt_args.get('use_flash_attention_2', False) if isinstance(ckpt_args, dict) else False
                if args.use_flash_attention_2 != ckpt_used_flash:
                     logger.warning(f"Checkpoint flash attention setting mismatch. Loading with strict=False.")
                miss, unex = model_to_load.load_state_dict(ckpt['model'], strict=False)
                if rank == 0: logger.info(f"Model state loaded. Missing:{miss or 'None'}. Unexpected:{unex or 'None'}.")

                if 'optimizer' in ckpt: optimizer.load_state_dict(ckpt['optimizer'])
                if 'lr_scheduler' in ckpt: lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
                if 'scaler' in ckpt and ckpt['scaler'] and scaler.is_enabled():
                    try: scaler.load_state_dict(ckpt['scaler'])
                    except Exception as e: logger.warning(f"Failed to load scaler state: {e}.")
                elif scaler.is_enabled(): logger.warning("Scaler enabled but no state found in checkpoint.")

                global_step = ckpt.get('global_step', 0)
                # Recalculate start_epoch based on resumed step
                num_steps_per_epoch_resume = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) if len(train_dataloader) > 0 else 0
                if num_steps_per_epoch_resume > 0:
                     start_epoch = global_step // num_steps_per_epoch_resume
                else: start_epoch = ckpt.get('epoch', 0) # Fallback

                if max_train_steps > 0 and global_step >= max_train_steps:
                    start_epoch = math.ceil(max_train_steps / num_steps_per_epoch_resume) if num_steps_per_epoch_resume > 0 else 0 # Prevent loop
                    if rank == 0: logger.info(f"Resuming from step {global_step}, which meets/exceeds max_steps ({max_train_steps}). Training will not continue.")
                else:
                    if rank == 0: logger.info(f"Resuming training from Global Step {global_step} (Epoch approx {start_epoch+1})")

                resumed_from_checkpoint = True
                try: # Restore RNG
                    if 'torch_rng_state' in ckpt: torch.set_rng_state(ckpt['torch_rng_state'].cpu())
                    if device.type == 'cuda' and 'torch_cuda_rng_state_all' in ckpt and ckpt['torch_cuda_rng_state_all']: torch.cuda.set_rng_state_all(ckpt['torch_cuda_rng_state_all'])
                    if 'numpy_rng_state' in ckpt: np.random.set_state(ckpt['numpy_rng_state'])
                    if 'python_rng_state' in ckpt: random.setstate(ckpt['python_rng_state'])
                    if rank == 0: logger.info("Restored RNG states from checkpoint.")
                except Exception as rng_e: logger.warning(f"Could not restore RNG states: {rng_e}")

                del ckpt; gc.collect();
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                if is_distributed: torch.distributed.barrier()

            except Exception as e:
                logger.error(f"Failed to load checkpoint state: {e}", exc_info=True)
                start_epoch, global_step, resumed_from_checkpoint = 0, 0, False
        else: logger.warning(f"Checkpoint path specified, but state file missing. Starting fresh.")
    else: logger.info("No checkpoint specified, starting training from scratch.")


    # --- DDP Wrapping (after loading state) ---
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        logger.info("Model wrapped with DDP.")


    # --- Initial Save (Step 0) ---
    if rank == 0 and global_step == 0 and not resumed_from_checkpoint:
         logger.info("Performing initial save at step 0...")
         save_checkpoint(args, model, optimizer, lr_scheduler, scaler, 0, 0, rank, tokenizer)

    # Training Start Logging (Rank 0 Only)
    if rank == 0:
        logger.info("***** Training Configuration *****")
        logger.info(f"   Model Class: {type(model)}")
        unwrapped_model_for_log = model.module if hasattr(model, 'module') else model
        if hasattr(unwrapped_model_for_log.config, '_attn_implementation'): logger.info(f"   Attention Implementation Used: {unwrapped_model_for_log.config._attn_implementation}")
        else: logger.info(f"   Attention Implementation Requested: {'flash_attention_2' if args.use_flash_attention_2 else 'Standard'}")
        logger.info(f"   AMP Enabled: {scaler.is_enabled()}")
        logger.info(f"   Evaluation Trigger: {'After Each Save' if args.eval_after_save else f'Interval (Std: {args.eval_steps}, Prime: {args.priming_eval_steps})'}")
        logger.info(f"   Max Steps Target: {max_train_steps}")
        logger.info(f"   Num Epochs Target: {args.num_train_epochs}")
        effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
        logger.info(f"   Effective Batch Size: {effective_batch}")
        logger.info(f"   Device: {device}, Distributed: {is_distributed} (World Size: {world_size})")
        logger.info(f"   Checkpoint Steps: {len(target_save_steps)} total calculated steps.")


    # === Training Loop ===
    if is_distributed: torch.distributed.barrier()
    training_start_time = time.time()
    final_global_step = global_step
    # Calculate max epochs needed based on final max_train_steps
    num_steps_per_epoch_final = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) if len(train_dataloader) > 0 else 1
    max_epochs_needed = math.ceil(max_train_steps / num_steps_per_epoch_final) if num_steps_per_epoch_final > 0 else 1

    try:
        for epoch in range(start_epoch, max_epochs_needed):
            if final_global_step >= max_train_steps:
                if rank == 0: logger.info(f"Max steps ({max_train_steps}) reached before starting epoch {epoch + 1}. Stopping.")
                break

            if rank == 0: logger.info(f"--- Starting Epoch {epoch + 1}/{max_epochs_needed} (Step {final_global_step}/{max_train_steps}) ---")
            model.train()
            final_global_step = train_epoch(
                args=args, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, scaler=scaler,
                train_dataloader=train_dataloader, train_sampler=train_sampler,
                epoch=epoch, global_step=final_global_step, device=device, rank=rank, world_size=world_size,
                run=run, tokenizer=tokenizer,
                max_train_steps=max_train_steps,
                target_save_steps=target_save_steps,
            )
            if final_global_step >= max_train_steps: break

        training_duration = time.time() - training_start_time
        if rank == 0:
             logger.info(f"***** Training Finished *****")
             logger.info(f"Total Training Time: {training_duration:.2f}s")
             logger.info(f"Final Global Step: {final_global_step}")
             if final_global_step < max_train_steps: logger.warning(f"Training finished early at step {final_global_step} (target: {max_train_steps}).")

        # Final Saving (Rank 0 Only)
        if rank == 0:
            if final_global_step not in target_save_steps:
                 logger.info(f"Performing final save at step {final_global_step} (not part of scheduled saves)...")
                 save_checkpoint(args, model, optimizer, lr_scheduler, scaler, epoch, final_global_step, rank, tokenizer)
            else: logger.info(f"Final step {final_global_step} was already saved according to schedule.")

            final_model_path = Path(args.output_dir) / "final_model"
            logger.info(f"Saving final model weights/tokenizer/config to: {final_model_path}")
            try:
                final_model_path.mkdir(parents=True, exist_ok=True)
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(final_model_path)
                tokenizer.save_pretrained(final_model_path)
                args_dict_final = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
                # Add calculated max_steps if it wasn't provided initially
                args_dict_final['calculated_max_steps'] = max_train_steps
                with open(final_model_path / "training_args.json", "w") as f: json.dump(args_dict_final, f, indent=4)
                logger.info(f"Final model components saved to {final_model_path}")

                if run:
                    try:
                        final_ckpt_dir = Path(args.output_dir) / f"checkpoint-{final_global_step}"
                        if final_ckpt_dir.is_dir(): run[f"artifacts/final_checkpoint_step_{final_global_step}"].upload_files(str(final_ckpt_dir))
                        run["artifacts/final_model_dir"].upload_files(str(final_model_path))
                        run["training/duration_seconds"] = training_duration
                        run["training/final_global_step"] = final_global_step
                        logger.info("Logged final training stats and artifacts to Neptune.")
                    except Exception as e: logger.warning(f"Neptune final upload/log failed: {e}")
            except Exception as e: logger.error(f"Failed to save final model components: {e}", exc_info=True)

    except KeyboardInterrupt: logger.warning("Training interrupted by user (KeyboardInterrupt).")
    except Exception as e: logger.critical(f"Unhandled error during training loop (Rank {rank}): {e}", exc_info=True)
    finally: # Cleanup
        if is_distributed: logger.info(f"Destroying process group (Rank {rank})..."); torch.distributed.destroy_process_group()
        if rank == 0 and run:
            try: logger.info("Stopping Neptune run..."); run.stop(); logger.info("Neptune run stopped.")
            except Exception as ne: logger.error(f"Neptune stop failed: {ne}")
        logger.info(f"Training script finished (Rank {rank}).")


# Define fallback tqdm before __main__ block
def _fallback_tqdm(iterable, *args, **kwargs): return iterable

if __name__ == "__main__":
    # Basic logging config before full setup in main
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    try: from tqdm.auto import tqdm
    except ImportError: print("Warning: tqdm not installed."); tqdm = _fallback_tqdm
    main()