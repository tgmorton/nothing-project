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
import datetime  # For sentinel file timestamp

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
run = None  # For Neptune, global to be accessible


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
    parser.add_argument("--checkpoint_ready_sentinel", type=str, default="EVAL_READY.txt",  # New Argument
                        help="Filename created inside a checkpoint dir when it's fully written and ready for evaluation.")

    # === Logging & Saving ===
    parser.add_argument("--logging_steps", type=int, default=100, help="Log train metrics every X steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps.")

    # === LOCAL Evaluation Triggering Control (these are used if --local_eval is set) ===
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Trigger local evaluation every X steps (if --local_eval).")
    parser.add_argument("--local_eval", action="store_true",
                        help="Run evaluation script locally as subprocess. If False, no evaluation is triggered by this script.")
    parser.add_argument("--evaluate_script_path", type=str, default="src/evaluate.py",  # Relative to project root
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

    if args.priming_eval_steps is None: args.priming_eval_steps = args.eval_steps

    if not args.train_dataset_path: parser.error("--train_dataset_path required for training.")
    if not args.output_dir: parser.error("--output_dir required for training.")
    if not Path(args.train_dataset_path).is_dir(): parser.error(f"Train dataset not found: {args.train_dataset_path}")

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
        print(f"Warning: Output directory {args.output_dir} does not exist. It will be created.")
    return args


def get_device():
    """Gets the appropriate device for PyTorch computations."""
    import torch
    import os
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS")
    elif torch.cuda.is_available():
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if local_rank >= torch.cuda.device_count(): local_rank = 0  # Safety
            device = torch.device(f"cuda:{local_rank}")
            print(f"Using CUDA GPU: {local_rank} - {torch.cuda.get_device_name(device)}")
        except Exception as e:
            print(f"Error setting CUDA device based on LOCAL_RANK: {e}. Defaulting to cuda:0 or CPU.")
            if torch.cuda.device_count() > 0:
                device = torch.device(f"cuda:0")
                print(f"Defaulted to CUDA GPU: 0 - {torch.cuda.get_device_name(device)}")
            else:
                device = torch.device("cpu")
                print("CUDA devices found but error occurred. Using CPU.")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def setup_logging(log_level=logging.INFO, rank=0):
    """Configures basic logging."""
    global logger
    fmt = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s"
    dfmt = "%Y-%m-%d %H:%M:%S"
    level_to_set = log_level if rank == 0 else logging.CRITICAL + 1  # Only rank 0 logs INFO+

    # Clear existing handlers if any (useful for re-runs in notebooks or testing)
    # logging.getLogger().handlers = [] # Be cautious with this in complex setups

    logging.basicConfig(level=level_to_set, format=fmt, datefmt=dfmt, force=True, stream=sys.stdout)
    logger = logging.getLogger(__name__)
    if rank == 0: logger.info(f"Logging setup complete (Rank {rank}). Logging to stdout.")


def set_seed(seed_value, rank=0):  # Added rank for conditional logging
    """Sets random seeds."""
    import random
    import numpy as np
    import torch
    global logger  # logger might not be initialized if this is called before setup_logging
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    log_msg = f"Set random seed: {seed_value} for rank {rank}"
    if logger and logger.isEnabledFor(logging.INFO):  # Check if logger is ready and level allows
        logger.info(log_msg)
    elif rank == 0:  # Fallback for rank 0 if logger not fully ready
        print(f"INFO (Rank {rank} pre-log): {log_msg}")


def setup_distributed(args):
    """Sets up DDP environment."""
    import torch
    import os
    global logger  # logger may not be fully initialized here
    is_dist, rank, world_size, local_rank = False, 0, 1, 0

    # Check for Slurm environment variables first for DDP
    if 'SLURM_PROCID' in os.environ:
        is_dist = True
        try:
            rank = int(os.environ['SLURM_PROCID'])
            world_size = int(os.environ['SLURM_NPROCS'])
            local_rank = int(
                os.environ.get('SLURM_LOCALID', rank % torch.cuda.device_count()))  # SLURM_LOCALID might not be set

            os.environ['RANK'] = str(rank)
            os.environ['WORLD_SIZE'] = str(world_size)
            # MASTER_ADDR and MASTER_PORT need to be set for torch.distributed.init_process_group
            # Slurm usually handles this, or a helper script does. If not, this needs robust handling.
            # Assuming srun or a launch utility sets these:
            if 'MASTER_ADDR' not in os.environ:
                os.environ['MASTER_ADDR'] = 'localhost'  # Fallback, not robust for multi-node
                if rank == 0: print("Warning: MASTER_ADDR not set, defaulting to localhost.")
            if 'MASTER_PORT' not in os.environ:
                os.environ['MASTER_PORT'] = '29500'  # Fallback, ensure unique
                if rank == 0: print("Warning: MASTER_PORT not set, defaulting to 29500.")

            if not torch.cuda.is_available(): raise RuntimeError("DDP (Slurm) needs CUDA.")

            torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
            torch.cuda.set_device(local_rank)

            msg = f"DDP (Slurm) Init: Rank {rank}/{world_size}, LocalRank {local_rank} on cuda:{local_rank}"
            if logger and logger.isEnabledFor(logging.INFO):
                logger.info(msg)
            else:
                print(f"INFO (Rank {rank} pre-log): {msg}")
            torch.distributed.barrier()
        except Exception as e:
            print(f"CRITICAL ERROR (Rank {rank} pre-log): DDP (Slurm) init failed: {e}")
            traceback.print_exc()
            sys.exit(1)  # Critical failure

    elif 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1:  # Torchrun/Elastic
        is_dist = True
        try:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
            if not torch.cuda.is_available(): raise RuntimeError("DDP (Torchrun) needs CUDA.")

            torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
            torch.cuda.set_device(local_rank)

            msg = f"DDP (Torchrun) Init: Rank {rank}/{world_size}, LocalRank {local_rank} on cuda:{local_rank}"
            if logger and logger.isEnabledFor(logging.INFO):
                logger.info(msg)
            else:
                print(f"INFO (Rank {rank} pre-log): {msg}")
            torch.distributed.barrier()
        except Exception as e:
            print(f"CRITICAL ERROR (Rank {rank} pre-log): DDP (Torchrun) init failed: {e}")
            traceback.print_exc()
            sys.exit(1)  # Critical failure
    else:
        msg = "DDP not enabled (single process)."
        if logger and logger.isEnabledFor(logging.INFO):
            logger.info(msg)
        else:
            print(f"INFO (Rank {rank} pre-log): {msg}")

    return is_dist, rank, world_size, local_rank


def load_training_data(args, is_distributed, rank, world_size, data_collator, seed):
    """Loads standard Arrow datasets specifically for training."""
    global logger
    from datasets import load_from_disk
    from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
    train_dl, train_sampler = None, None

    if not args.train_dataset_path:
        if rank == 0: logger.error("Missing training dataset path.")
        raise ValueError("Missing training dataset path.")

    if rank == 0: logger.info(f"Loading train data from: {args.train_dataset_path}")
    try:
        ds = load_from_disk(args.train_dataset_path)
        if rank == 0: logger.info(f"Full train dataset size: {len(ds):,}")

        sampler_seed = seed  # Use the base seed for sampler consistency across restarts if set_epoch is used
        if is_distributed:
            train_sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=sampler_seed)
            if rank == 0: logger.info(f"Using DistributedSampler for training (seed: {sampler_seed}).")
        else:
            g = torch.Generator()
            g.manual_seed(sampler_seed)
            train_sampler = RandomSampler(ds, generator=g)
            if rank == 0: logger.info(f"Using RandomSampler for training (seed: {sampler_seed}).")

        train_dl = DataLoader(ds, sampler=train_sampler, batch_size=args.per_device_train_batch_size,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=data_collator,
                              persistent_workers=(True if args.num_workers > 0 else False))
        if rank == 0: logger.info(
            f"Train DataLoader created (num_workers: {args.num_workers}, persistent: {args.num_workers > 0}).")
    except Exception as e:
        if rank == 0: logger.error(f"Training data loading failed: {e}", exc_info=True)
        raise
    return train_dl, train_sampler


def save_checkpoint(args, model, optimizer, lr_scheduler, scaler, epoch, global_step, rank, tokenizer=None):
    """Saves training state. Only rank 0 executes the save, but all ranks might need paths."""
    global logger  # Assumes logger is initialized

    # Define checkpoint directory path (all ranks might need this for consistency if loading)
    # However, only rank 0 performs the actual write operations.
    checkpoint_dir_name = f"checkpoint-{global_step}"
    ckpt_dir = Path(args.output_dir) / checkpoint_dir_name

    if rank != 0:
        return  # Only rank 0 saves checkpoints and creates sentinel

    logger.info(f"Attempting to save checkpoint {global_step} to {ckpt_dir}...")
    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create checkpoint directory {ckpt_dir}: {e}")
        return  # Cannot proceed with saving

    # Unwrap model if DDP
    unwrapped_model = model.module if hasattr(model, 'module') else model

    # Save training state dictionary
    state = {
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'scaler': scaler.state_dict() if scaler.is_enabled() else None,
        'epoch': epoch,
        'global_step': global_step,
        'args': vars(args),  # Save args for reproducibility
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
        'torch_cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }
    # Note: model state_dict is saved via save_pretrained below for HF compatibility.
    # If you need to save it in training_state.pt as well, add: 'model': unwrapped_model.state_dict()

    state_file = ckpt_dir / "training_state.pt"
    try:
        torch.save(state, state_file)
        logger.info(f"Saved training state dictionary to: {state_file}")
    except Exception as e:
        logger.error(f"Failed to save training_state.pt: {e}", exc_info=True)
        # Decide if to proceed with model/tokenizer save if state dict fails

    # Save model and tokenizer using Hugging Face's method
    try:
        unwrapped_model.save_pretrained(str(ckpt_dir))  # Ensure path is string for older HF versions
        logger.info(f"Saved model weights and config to {ckpt_dir} using save_pretrained.")
        if tokenizer:
            tokenizer.save_pretrained(str(ckpt_dir))
            logger.info(f"Saved tokenizer to {ckpt_dir} using save_pretrained.")

        # --- Create the ready sentinel file ---
        if args.checkpoint_ready_sentinel:
            sentinel_file_path = ckpt_dir / args.checkpoint_ready_sentinel
            try:
                with open(sentinel_file_path, 'w') as f:
                    f.write(f"Checkpoint {global_step} fully written at {datetime.datetime.now().isoformat()}.\n")
                logger.info(f"Created ready sentinel file: {sentinel_file_path}")
            except IOError as e_sentinel:
                logger.error(f"Failed to create ready sentinel file {sentinel_file_path}: {e_sentinel}")
        # --- End sentinel file creation ---

    except Exception as e_hf_save:
        logger.error(f"Failed during Hugging Face save_pretrained or sentinel creation for {ckpt_dir}: {e_hf_save}",
                     exc_info=True)

    logger.info(f"Checkpoint {global_step} saving process completed.")


def create_small_model_config(base_model_name: str, corpus_size_tag: str, tokenizer: PreTrainedTokenizer,
                              logger_instance: Optional[logging.Logger]):  # logger_instance can be None
    """Creates a small model configuration based on a corpus size tag."""
    try:
        config = AutoConfig.from_pretrained(base_model_name)
        if logger_instance: logger_instance.info(f"Loaded base config structure from: {base_model_name}")
    except Exception as e:
        if logger_instance: logger_instance.error(f"Failed to load base config '{base_model_name}': {e}", exc_info=True)
        raise

    tag = corpus_size_tag.lower()
    if tag == "10m":
        target_n_layer, target_n_head, target_n_embd = 4, 4, 256
    elif tag == "100m":
        target_n_layer, target_n_head, target_n_embd = 6, 6, 384
    else:
        err_msg = f"Unknown corpus_size_tag: '{corpus_size_tag}'. Expected '10m' or '100m'."
        if logger_instance: logger_instance.error(err_msg)
        raise ValueError(err_msg)

    if logger_instance: logger_instance.info(f"Applying config modifications for model size tag: '{tag}'.")

    # Standard GPT-2 config attributes
    config.n_layer = target_n_layer
    config.n_head = target_n_head
    config.n_embd = target_n_embd

    # Ensure vocab_size matches tokenizer, and other good practices for training
    config.vocab_size = len(tokenizer)
    config.use_cache = False  # Disable for training

    if logger_instance:
        logger_instance.info(f"Final SMALL config params: n_layer={config.n_layer}, "
                             f"n_head={config.n_head}, n_embd={config.n_embd}, "
                             f"vocab_size={config.vocab_size}, use_cache={config.use_cache}")
    return config


def run_local_evaluation(args, checkpoint_dir: Path, global_step: int, rank: int):
    """
    Runs evaluate.py locally as a subprocess. Only called by rank 0 if args.local_eval is True.
    """
    if rank != 0 or not args.local_eval:
        return

    global logger, run  # Access global Neptune run object if used by train.py

    eval_output_base_dir = Path(args.output_dir) / checkpoint_dir.name / "local_eval_results"
    try:
        eval_output_base_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(
            f"Failed to create local evaluation output directory {eval_output_base_dir}: {e}. Skipping local eval.")
        return

    logger.info(f"--- Triggering Local Evaluation for Step {global_step} (Checkpoint: {checkpoint_dir.name}) ---")
    python_executable = sys.executable

    # Resolve evaluate_script_path: if it's relative, assume it's relative to the project root (where train.py is likely run from)
    # For robustness, it's better if evaluate_script_path is absolute or clearly defined.
    # Assuming it's like 'src/evaluate.py' and PWD is the project root.
    eval_script_path_abs = Path(args.evaluate_script_path).resolve()
    if not eval_script_path_abs.is_file():
        logger.error(
            f"Local evaluation script not found at resolved path: {eval_script_path_abs}. Check --evaluate_script_path.")
        return

    # Construct arguments for evaluate.py
    eval_cmd_list = [
        str(python_executable), str(eval_script_path_abs),
        "--checkpoint_path", str(checkpoint_dir.resolve()),  # Pass the specific checkpoint being evaluated
        "--output_dir", str(eval_output_base_dir.resolve()),  # eval.py will manage its outputs here
        "--seed", str(args.seed),  # Pass training seed for consistency in sampling
        "--num_workers", str(args.num_workers),
        "--base_model_name_or_path", args.model,  # So evaluate.py knows the base arch if needed
        # No need to pass --checkpoint_ready_sentinel to evaluate.py; it consumes ready checkpoints
    ]

    if args.use_amp: eval_cmd_list.append("--use_amp")

    # Standard Evaluation Args
    if args.trigger_standard_eval:
        eval_cmd_list.append("--run_standard_eval")
        if args.validation_dataset_path:  # This was validated in parse_args
            eval_cmd_list.extend(["--validation_dataset_path", str(Path(args.validation_dataset_path).resolve())])
        # eval_max_samples could be passed too if desired

    # Priming Evaluation Args
    if args.trigger_priming_eval:
        eval_cmd_list.append("--run_priming_eval")
        if args.priming_eval_dir_path:  # Validated in parse_args
            eval_cmd_list.extend(["--priming_eval_dir_path", str(Path(args.priming_eval_dir_path).resolve())])
        # priming_eval_max_samples_per_file, priming_delimiter could be passed

    # Neptune details for evaluate.py to log to the *same training run*
    if args.neptune_project and NEPTUNE_AVAILABLE and run and hasattr(run, '_sys_id'):
        eval_cmd_list.extend(["--neptune_project", args.neptune_project])
        try:
            # Fetch the system ID of the current Neptune training run
            training_run_id = run['_sys_id'].fetch()
            eval_cmd_list.extend(["--neptune_run_id", training_run_id])
            logger.info(f"Passing current Neptune training run ID ({training_run_id}) to local evaluate.py.")
            # evaluate.py can then append its metrics to this existing run.
            # It will need to handle how it namespaces its metrics (e.g., eval_metrics/step_X/...)
        except Exception as e_neptune_id:
            logger.warning(f"Could not fetch Neptune run ID for local eval: {e_neptune_id}")
    # NEPTUNE_API_TOKEN should be inherited via environment.

    logger.info(f"Executing local evaluation command: {' '.join(eval_cmd_list)}")
    eval_start_time = time.time()
    try:
        process_env = os.environ.copy()
        # Capture output to avoid cluttering main log, but log it on completion/error
        result = subprocess.run(eval_cmd_list, check=True, env=process_env, capture_output=True, text=True)
        logger.info(f"Local evaluation for step {global_step} completed in {time.time() - eval_start_time:.2f}s.")
        if result.stdout: logger.info(f"Local Eval STDOUT for step {global_step}:\n{result.stdout.strip()}")
        if result.stderr: logger.warning(
            f"Local Eval STDERR for step {global_step}:\n{result.stderr.strip()}")  # Log stderr as warning
    except FileNotFoundError:
        logger.error(f"Error: Python executable '{python_executable}' or script '{eval_script_path_abs}' not found.")
    except subprocess.CalledProcessError as e_subproc:
        logger.error(
            f"Error running local evaluation script for step {global_step}. Return code: {e_subproc.returncode}")
        if e_subproc.stdout: logger.error(f"Local Eval STDOUT (Error):\n{e_subproc.stdout.strip()}")
        if e_subproc.stderr: logger.error(f"Local Eval STDERR (Error):\n{e_subproc.stderr.strip()}")
    except Exception as e_generic:
        logger.error(f"Unexpected error during local evaluation for step {global_step}: {e_generic}", exc_info=True)


def train_epoch(args, model, optimizer, lr_scheduler, scaler, train_dataloader,
                train_sampler, epoch, global_step_counter, device, rank, world_size,
                # Renamed global_step to global_step_counter
                neptune_run_obj, tokenizer,  # Renamed run to neptune_run_obj for clarity
                max_train_steps, log_save_steps_set):
    """
    Runs one training epoch. Returns the updated global_step_counter.
    """
    global logger  # Assumes logger is initialized
    import torch
    from torch.utils.data import DistributedSampler
    from tqdm.auto import tqdm  # Import here if not global
    import math
    import sys
    import gc
    from torch.nn.utils import clip_grad_norm_
    from torch.amp import autocast

    model.train()
    if train_sampler is not None and isinstance(train_sampler, DistributedSampler):
        try:
            train_sampler.set_epoch(epoch)  # Important for shuffling in DDP
        except AttributeError:  # Should not happen with standard DistributedSampler
            if rank == 0: logger.warning(
                "Train sampler does not have set_epoch method. Shuffling might be affected in DDP.")

    # Progress bar only on rank 0
    prog_bar_desc = f"Epoch {epoch + 1}/{args.num_train_epochs}"
    progress_bar = tqdm(train_dataloader, desc=prog_bar_desc, leave=True, disable=(rank != 0), position=0)

    total_loss_for_logging = 0.0
    num_steps_for_logging = 0

    # Loop over batches
    for step_in_epoch, batch in enumerate(progress_bar):
        # Move batch to device
        try:
            batch_on_device = {k: v.to(device, non_blocking=True) for k, v in batch.items() if
                               isinstance(v, torch.Tensor)}
        except RuntimeError as e:
            if rank == 0: logger.error(f"Error moving training batch to device {device}: {e}. Skipping batch.")
            optimizer.zero_grad(set_to_none=True)  # Clear gradients for this failed batch
            continue  # Skip to next batch

        # Forward pass
        try:
            amp_enabled_for_pass = args.use_amp and device.type == 'cuda'
            with autocast(device_type=device.type, enabled=amp_enabled_for_pass):
                outputs = model(**batch_on_device)
                loss = outputs.loss  # This loss is per-device, per-micro-batch

            if loss is None:  # Should not happen with HF models usually
                if rank == 0: logger.error("Loss was None from model output. Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            if not torch.isfinite(loss):
                if rank == 0: logger.warning(
                    f"Non-finite loss detected ({loss.item()}). Skipping update for this micro-batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            # Normalize loss for gradient accumulation
            # The loss from the model is already an average over the micro-batch.
            # We sum these normalized losses before averaging for logging.
            # For backward pass, the average loss (loss / grad_accum_steps) is used.
            current_micro_batch_loss_val = loss.item()
            loss_for_backward = loss / args.gradient_accumulation_steps

        except Exception as e_fwd:
            if rank == 0: logger.error(f"Forward pass error: {e_fwd}", exc_info=True)
            optimizer.zero_grad(set_to_none=True)
            continue  # Skip to next batch

        # Backward pass
        try:
            if scaler.is_enabled():  # AMP enabled
                scaler.scale(loss_for_backward).backward()
            else:  # AMP disabled
                loss_for_backward.backward()
        except Exception as e_bwd:
            if rank == 0: logger.error(f"Backward pass error: {e_bwd}", exc_info=True)
            optimizer.zero_grad(set_to_none=True)
            continue  # Skip to next batch

        total_loss_for_logging += current_micro_batch_loss_val  # Accumulate micro-batch loss for logging
        num_steps_for_logging += 1

        # Optimizer step (gradient accumulation)
        if (step_in_epoch + 1) % args.gradient_accumulation_steps == 0:
            try:
                if args.max_grad_norm and args.max_grad_norm > 0:
                    if scaler.is_enabled(): scaler.unscale_(optimizer)  # Unscale before clipping
                    # Ensure clipping is done on model.module.parameters() if DDP, else model.parameters()
                    params_to_clip = model.module.parameters() if hasattr(model, 'module') else model.parameters()
                    clip_grad_norm_(params_to_clip, args.max_grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()  # Update scale for next iteration
                else:
                    optimizer.step()  # Standard optimizer step

                lr_scheduler.step()  # Step LR scheduler
                optimizer.zero_grad(set_to_none=True)  # Clear gradients for next accumulation

                global_step_counter += 1  # Increment global optimizer step counter

            except Exception as e_opt:
                if rank == 0: logger.error(
                    f"Optimizer/Scheduler step error at global_step ~{global_step_counter}: {e_opt}", exc_info=True)
                optimizer.zero_grad(set_to_none=True)  # Ensure clear grads
                # Continue training if possible, but this step was problematic

            # Logging (Rank 0 Only)
            if rank == 0 and global_step_counter % args.logging_steps == 0 and num_steps_for_logging > 0:
                avg_loss_since_last_log = total_loss_for_logging / num_steps_for_logging
                current_lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler,
                                                                      'get_last_lr') and lr_scheduler.get_last_lr() else \
                optimizer.param_groups[0]['lr']

                log_msg_train = f"Epoch {epoch + 1} | Step {global_step_counter}: Avg Train Loss (last {num_steps_for_logging} micro-batches) = {avg_loss_since_last_log:.4f}, LR = {current_lr:.3e}"
                logger.info(log_msg_train)

                if NEPTUNE_AVAILABLE and neptune_run_obj:
                    try:
                        if math.isfinite(avg_loss_since_last_log): neptune_run_obj["train/step_loss"].append(
                            avg_loss_since_last_log, step=global_step_counter)
                        if math.isfinite(current_lr): neptune_run_obj["train/learning_rate"].append(current_lr,
                                                                                                    step=global_step_counter)
                        if torch.cuda.is_available():
                            mem_alloc_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
                            neptune_run_obj["train/gpu_mem_alloc_gb"].append(mem_alloc_gb, step=global_step_counter)
                        if scaler.is_enabled(): neptune_run_obj["train/grad_scale"].append(scaler.get_scale(),
                                                                                           step=global_step_counter)
                    except Exception as e_neptune_log:
                        logger.warning(f"Neptune train logging failed at step {global_step_counter}: {e_neptune_log}")

                total_loss_for_logging = 0.0  # Reset for next logging period
                num_steps_for_logging = 0

            # Checkpoint Saving (Rank 0 Only, after optimizer step)
            # Check against `log_save_steps_set` for logarithmic early saves
            # and `args.save_steps` for regular interval saves.
            is_log_save_step = global_step_counter in log_save_steps_set
            is_regular_save_step = args.save_steps > 0 and global_step_counter > 0 and (
                        global_step_counter % args.save_steps == 0)

            if rank == 0 and (is_log_save_step or is_regular_save_step):
                if is_distributed: torch.distributed.barrier()  # Sync before save
                save_checkpoint(args, model, optimizer, lr_scheduler, scaler, epoch, global_step_counter, rank,
                                tokenizer)
                if is_distributed: torch.distributed.barrier()  # Sync after save

            # Local Evaluation Trigger (Rank 0 Only, after optimizer step & potential save)
            if args.local_eval and rank == 0:
                # Check if current global_step_counter aligns with standard eval or priming eval steps
                time_for_std_eval = args.trigger_standard_eval and global_step_counter > 0 and (
                            global_step_counter % args.eval_steps == 0)
                time_for_prime_eval = args.trigger_priming_eval and global_step_counter > 0 and (
                            global_step_counter % args.priming_eval_steps == 0)

                if time_for_std_eval or time_for_prime_eval:
                    checkpoint_dir_for_eval = Path(args.output_dir) / f"checkpoint-{global_step_counter}"
                    if not checkpoint_dir_for_eval.is_dir():
                        logger.warning(f"Local eval trigger: Checkpoint dir {checkpoint_dir_for_eval} not found. "
                                       f"This may happen if save_steps don't align. Attempting to save now for eval.")
                        if is_distributed: torch.distributed.barrier()
                        save_checkpoint(args, model, optimizer, lr_scheduler, scaler, epoch, global_step_counter, rank,
                                        tokenizer)
                        if is_distributed: torch.distributed.barrier()

                    if checkpoint_dir_for_eval.is_dir():  # Re-check after potential save
                        run_local_evaluation(args, checkpoint_dir_for_eval, global_step_counter, rank)
                    else:
                        logger.error(
                            f"Local eval trigger: Failed to find/save checkpoint {checkpoint_dir_for_eval}. Cannot run local eval.")

            # Check for Max Steps Completion
            if max_train_steps > 0 and global_step_counter >= max_train_steps:
                if rank == 0:
                    logger.info(
                        f"Max training steps ({max_train_steps}) reached at global step {global_step_counter} within epoch {epoch + 1}.")
                # Update progress bar one last time before returning from epoch
                if rank == 0:
                    progress_bar.set_postfix(
                        {"loss": f"{current_micro_batch_loss_val:.4f}", "step": global_step_counter,
                         "status": "Max steps reached"})
                return global_step_counter  # Exit epoch early

        # Update progress bar at end of micro-batch loop iteration (rank 0 only)
        if rank == 0:
            progress_bar.set_postfix({"loss": f"{current_micro_batch_loss_val:.4f}", "step": global_step_counter})

    # End of epoch
    if rank == 0:
        progress_bar.close()
        logger.info(f"--- Epoch {epoch + 1} Finished (Optimizer Step reached: {global_step_counter}) ---")

    return global_step_counter


# --- Main Function ---
def main():
    global run, logger  # Make run (Neptune) and logger global

    # 1. Parse Arguments (Happens on all ranks, but only rank 0 uses some args for decisions)
    args = parse_args()

    # 2. Setup DDP Environment (Crucial to do this early)
    is_distributed, rank, world_size, local_rank = setup_distributed(args)

    # 3. Setup Logging (Rank-aware)
    setup_logging(rank=rank)  # logger is initialized here globally
    if rank == 0: logger.info("***** Starting Training Script (train.py) *****")
    if rank == 0: logger.info(f"Running with Arguments: {vars(args)}")

    # 4. Setup Device and Seed (Rank-aware)
    device = get_device()  # get_device uses LOCAL_RANK if DDP is active
    set_seed(args.seed + rank, rank=rank)  # Different seed per rank for DDP data loading, etc.

    # 5. Neptune Setup (Rank 0 Only)
    # `run` is already declared global
    if rank == 0 and NEPTUNE_AVAILABLE and args.neptune_project:
        try:
            run = neptune.init_run(project=args.neptune_project, name=args.neptune_run_name, tags=args.neptune_tags,
                                   mode="async")
            # Log processed args (handling Path objects and lists for Neptune)
            processed_args_for_neptune = {}
            for k, v_item in vars(args).items():
                if isinstance(v_item, Path):
                    processed_args_for_neptune[k] = str(v_item)
                elif isinstance(v_item, list):
                    processed_args_for_neptune[k] = ','.join(map(str, v_item)) if v_item else "None"
                elif v_item is None:
                    processed_args_for_neptune[k] = "None"
                else:
                    processed_args_for_neptune[k] = v_item
            run["parameters"] = processed_args_for_neptune
            logger.info(f"Neptune logging enabled. Run URL: {run.get_url()}")
        except Exception as e_neptune_init:
            logger.error(f"Neptune init failed: {e_neptune_init}. Neptune logging disabled.", exc_info=True)
            run = None  # Ensure run is None if init fails
    elif rank == 0:
        logger.info("Neptune logging disabled (client not available or project not specified).")

    # 6. Model & Tokenizer Initialization/Loading
    model, tokenizer, config = None, None, None
    try:
        if rank == 0: logger.info(f"Initializing Tokenizer from base: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                if rank == 0: logger.info(f"Set tokenizer pad_token to its eos_token: '{tokenizer.eos_token}'")
            else:  # This case is rare for models like GPT-2
                num_added = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                if rank == 0: logger.warning(
                    f"Tokenizer had no pad_token or eos_token. Added new [PAD] token ({num_added} new tokens).")

        if rank == 0: logger.info(f"Creating model config for size '{args.model_size}' based on '{args.model}'.")
        # Pass logger instance only for rank 0, or None for other ranks
        logger_for_config = logger if rank == 0 else None
        config = create_small_model_config(base_model_name=args.model, corpus_size_tag=args.model_size,
                                           tokenizer=tokenizer, logger_instance=logger_for_config)

        if rank == 0: logger.info("Initializing NEW model from generated config (random weights).")
        model = GPT2LMHeadModel(config=config)
        # If tokenizer was modified (e.g. new pad_token added), resize model embeddings
        # This should ideally be done *before* wrapping with DDP or loading checkpoint state.
        if tokenizer.vocab_size != model.config.vocab_size:  # Check if vocab size differs
            model.resize_token_embeddings(len(tokenizer))
            if rank == 0: logger.info(f"Resized model token embeddings to: {len(tokenizer)}")

        model.to(device)
        if rank == 0: logger.info(
            f"Initialized model '{model.name_or_path if hasattr(model, 'name_or_path') else type(model).__name__}' moved to {device}.")

        if is_distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=False)  # Adjust find_unused_parameters if needed
            if rank == 0: logger.info("Model wrapped with DistributedDataParallel (DDP).")

    except Exception as e_model_init:
        if rank == 0: logger.critical(f"Model/Tokenizer initialization failed: {e_model_init}", exc_info=True)
        if NEPTUNE_AVAILABLE and run and rank == 0: run["critical_errors/model_init"] = traceback.format_exc()
        if is_distributed: torch.distributed.destroy_process_group()
        sys.exit(1)

    # 7. Training Data Loading
    train_dataloader, train_sampler = None, None
    try:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        train_dataloader, train_sampler = load_training_data(args, is_distributed, rank, world_size, data_collator,
                                                             args.seed)
        if rank == 0: logger.info(f"Loaded training dataset. DataLoader has ~{len(train_dataloader)} batches.")
    except Exception as e_data_load:
        if rank == 0: logger.critical(f"Training data loading failed: {e_data_load}", exc_info=True)
        if NEPTUNE_AVAILABLE and run and rank == 0: run["critical_errors/data_load"] = traceback.format_exc()
        if is_distributed: torch.distributed.destroy_process_group()
        sys.exit(1)

    # 8. Optimizer, Scheduler, GradScaler Setup
    if rank == 0: logger.info("Setting up Optimizer, LR Scheduler, and Gradient Scaler...")
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]  # Common no_decay parameters

    # Correctly get parameters from DDP model if applicable
    model_params_for_optimizer = model.module.parameters() if is_distributed else model.parameters()
    model_named_params_for_optimizer = model.module.named_parameters() if is_distributed else model.named_parameters()

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model_named_params_for_optimizer if
                    not any(nd in n.lower() for nd in no_decay) and p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model_named_params_for_optimizer if
                    any(nd in n.lower() for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0},
    ]
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g["params"]]  # Filter empty groups

    if not optimizer_grouped_parameters:
        if rank == 0: logger.critical(
            "No parameters found for optimizer. Check model structure or requires_grad settings.")
        if NEPTUNE_AVAILABLE and run and rank == 0: run["critical_errors/optimizer_setup"] = "No params for optimizer."
        if is_distributed: torch.distributed.destroy_process_group()
        sys.exit(1)

    total_params = sum(p.numel() for p in model_params_for_optimizer)
    trainable_params = sum(p.numel() for g in optimizer_grouped_parameters for p in g['params'])
    if rank == 0: logger.info(f"Model Parameters: Total={total_params:,}, Trainable for Optimizer={trainable_params:,}")

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Calculate total training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) if len(
        train_dataloader) > 0 else 1
    if args.max_steps > 0:
        max_train_steps = args.max_steps
        if rank == 0: logger.info(f"Training limited to {max_train_steps:,} total optimizer steps (from --max_steps).")
    else:
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if rank == 0: logger.info(
            f"Calculated total training optimizer steps: {max_train_steps:,} ({args.num_train_epochs} epochs * {num_update_steps_per_epoch} steps/epoch).")

    if max_train_steps <= 0 and rank == 0:
        logger.warning("Max training steps is <= 0. Training might not proceed if not resuming past this point.")

    effective_warmup_steps = min(args.num_warmup_steps,
                                 max_train_steps) if max_train_steps > 0 else args.num_warmup_steps
    if rank == 0: logger.info(f"Effective LR warmup steps: {effective_warmup_steps}")

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=effective_warmup_steps,
        num_training_steps=max_train_steps if max_train_steps > 0 else 1  # Scheduler needs >0 steps
    )

    scaler_is_enabled = args.use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=scaler_is_enabled)
    if rank == 0: logger.info(f"Automatic Mixed Precision (AMP) enabled: {scaler.is_enabled()}.")

    # 9. Resume from Checkpoint Logic
    start_epoch = 0
    global_step = 0  # This will track the number of optimizer steps performed
    resumed_from_checkpoint_flag = False

    if args.checkpoint_path:
        # Checkpoint path is for the directory (e.g., .../checkpoint-500)
        # training_state.pt is inside this directory.
        training_state_file = Path(args.checkpoint_path) / "training_state.pt"
        if training_state_file.is_file():
            if rank == 0: logger.info(f"Attempting to load training state from: {training_state_file}")
            # Load to CPU first to prevent GPU OOM, especially in DDP. Model will be moved later.
            # All ranks load the checkpoint state to ensure optimizer/scheduler/RNG are consistent.
            map_location = 'cpu'  # Load to CPU
            try:
                checkpoint_data = torch.load(training_state_file, map_location=map_location)

                # Load model weights (HF model from the same checkpoint dir)
                # Model structure should already be initialized. We load weights into it.
                model_to_load_into = model.module if is_distributed else model
                # Assume checkpoint_path contains the HF saved model files (pytorch_model.bin, config.json)
                # This requires that save_checkpoint saved the model using save_pretrained.
                try:
                    # For loading weights into an existing model structure, it's often safer to load the state_dict
                    # from a temporary model loaded from the checkpoint, or directly if only state_dict was saved.
                    # If HF save_pretrained was used for the checkpoint:
                    temp_model_for_state_dict = GPT2LMHeadModel.from_pretrained(args.checkpoint_path,
                                                                                config=config)  # Load with current config
                    missing_keys, unexpected_keys = model_to_load_into.load_state_dict(
                        temp_model_for_state_dict.state_dict(), strict=False)
                    del temp_model_for_state_dict;
                    gc.collect()
                    if rank == 0:
                        logger.info(f"Model weights loaded from HF checkpoint files in {args.checkpoint_path}.")
                        if missing_keys: logger.warning(f"Missing keys in model state_dict: {missing_keys}")
                        if unexpected_keys: logger.warning(f"Unexpected keys in model state_dict: {unexpected_keys}")
                except Exception as e_load_hf_model:
                    if rank == 0: logger.error(
                        f"Failed to load model weights from HF files in {args.checkpoint_path}: {e_load_hf_model}. Attempting from training_state.pt if 'model' key exists.")
                    # Fallback: if 'model' state_dict was in training_state.pt (not recommended with save_pretrained)
                    if 'model' in checkpoint_data:
                        missing_keys, unexpected_keys = model_to_load_into.load_state_dict(checkpoint_data['model'],
                                                                                           strict=False)
                        if rank == 0: logger.info(
                            f"Loaded model state_dict from 'model' key in training_state.pt.")  # less common

                # Model should already be on device from earlier setup. If not, move it.
                # model.to(device) is done before DDP wrapping.

                if 'optimizer' in checkpoint_data: optimizer.load_state_dict(checkpoint_data['optimizer'])
                if 'lr_scheduler' in checkpoint_data: lr_scheduler.load_state_dict(checkpoint_data['lr_scheduler'])
                if 'scaler' in checkpoint_data and checkpoint_data['scaler'] is not None and scaler.is_enabled():
                    scaler.load_state_dict(checkpoint_data['scaler'])

                start_epoch = checkpoint_data.get('epoch', 0)
                global_step = checkpoint_data.get('global_step', 0)

                # Adjust start_epoch: if global_step suggests we completed this epoch, start next one.
                # If global_step is 0, start_epoch is 0.
                # If global_step = num_steps_in_epoch_0, then epoch 0 is done, start_epoch should be 1 for the next loop.
                # The 'epoch' saved in checkpoint is the epoch that was *just completed* or was in progress.
                # So, training loop should start from `start_epoch` if it was in progress, or `start_epoch + 1` if completed.
                # Simpler: train_epoch increments global_step. If resuming, global_step is the count of *completed* steps.
                # The epoch loop will run from `start_epoch` up to `args.num_train_epochs`.
                # If an epoch was partially completed, train_epoch will pick up.
                # Let's assume `start_epoch` saved is the one *about to start* or *in progress*.
                # For simplicity, the loop `for epoch in range(start_epoch, args.num_train_epochs):` will handle it.
                # If `global_step` indicates we are past `max_train_steps`, then training won't run.

                if rank == 0: logger.info(f"Resuming from Epoch {start_epoch}, Global Optimizer Step {global_step}.")
                resumed_from_checkpoint_flag = True

                # Restore RNG states (all ranks do this for consistency)
                try:
                    if 'torch_rng_state' in checkpoint_data: torch.set_rng_state(checkpoint_data['torch_rng_state'])
                    # CUDA RNG state needs to be moved to the correct device for each rank
                    if torch.cuda.is_available() and 'torch_cuda_rng_state_all' in checkpoint_data and checkpoint_data[
                        'torch_cuda_rng_state_all']:
                        # This loads all CUDA states; individual ranks will use their specific device's state.
                        # For DDP, ensure each rank gets its correct RNG state if saved per device.
                        # torch.cuda.set_rng_state(checkpoint_data['torch_cuda_rng_state_all'][local_rank], device=local_rank) might be safer
                        # For now, assume set_rng_state_all handles DDP context correctly if states were saved by it.
                        # A common pattern is to save RNG state only on rank 0 and broadcast or re-seed.
                        # Here, we load what was saved.
                        torch.cuda.set_rng_state_all(
                            checkpoint_data['torch_cuda_rng_state_all'])  # This loads list of tensors
                    if 'numpy_rng_state' in checkpoint_data: np.random.set_state(checkpoint_data['numpy_rng_state'])
                    if 'python_rng_state' in checkpoint_data: random.setstate(checkpoint_data['python_rng_state'])
                    if rank == 0: logger.info("Restored RNG states from checkpoint.")
                except Exception as e_rng_restore:
                    if rank == 0: logger.warning(f"Could not fully restore RNG states: {e_rng_restore}")

                del checkpoint_data;
                gc.collect()
                if is_distributed: torch.distributed.barrier()  # Sync all ranks after loading

            except Exception as e_ckpt_load:
                if rank == 0: logger.error(f"Failed to load training state from {training_state_file}: {e_ckpt_load}",
                                           exc_info=True)
                start_epoch, global_step, resumed_from_checkpoint_flag = 0, 0, False  # Reset to start fresh
        else:
            if rank == 0: logger.warning(
                f"Checkpoint dir {args.checkpoint_path} specified, but training_state.pt not found. Starting fresh.")
    else:
        if rank == 0: logger.info("No checkpoint path specified. Starting training from scratch.")

    # Save initial checkpoint at step 0 if not resuming (Rank 0 Only)
    if global_step == 0 and not resumed_from_checkpoint_flag and rank == 0:
        logger.info("Saving initial checkpoint at step 0 (training from scratch).")
        if is_distributed: torch.distributed.barrier()  # Ensure all model/optimizer setups are synced
        save_checkpoint(args, model, optimizer, lr_scheduler, scaler, 0, 0, rank, tokenizer)
        if is_distributed: torch.distributed.barrier()

    # Generate logarithmic save steps (Rank 0 for logging, all ranks might need the set if logic changes)
    log_save_steps_set = set()
    if args.save_steps > 0:  # Only relevant if regular saving is on, to give context to "first save step"
        current_log_step_val = 1
        # Generate steps like 1, 2, 4, 8, ..., up to save_steps / 2 (approx)
        while current_log_step_val < args.save_steps:
            log_save_steps_set.add(current_log_step_val)
            next_log_step_val = current_log_step_val * 2
            if next_log_step_val <= current_log_step_val: break  # Safety for overflow or step becoming 0
            current_log_step_val = next_log_step_val
    if rank == 0 and log_save_steps_set:
        logger.info(
            f"Calculated logarithmic save steps (earlier than regular save_steps={args.save_steps}): {sorted(list(log_save_steps_set))}")

    # 10. Training Loop
    if rank == 0:
        logger.info("***** Training Configuration Summary *****")
        logger.info(
            f"   Total Epochs: {args.num_train_epochs}, Max Optimizer Steps: {max_train_steps if max_train_steps > 0 else 'Epoch-defined'}")
        logger.info(
            f"   Resumed: {resumed_from_checkpoint_flag} (Start Epoch: {start_epoch}, Global Step: {global_step})")
        logger.info(f"   Local Evaluation by this script: {'ENABLED' if args.local_eval else 'DISABLED'}")
        if args.local_eval:
            logger.info(
                f"      Eval Script: {args.evaluate_script_path}, Std Eval: {args.trigger_standard_eval} (every {args.eval_steps}), Prime Eval: {args.trigger_priming_eval} (every {args.priming_eval_steps})")

    if is_distributed: torch.distributed.barrier()  # Final sync before training loop

    training_start_time = time.time()
    final_global_step_reached = global_step  # Initialize with current global_step

    try:
        for epoch in range(start_epoch, args.num_train_epochs):
            if max_train_steps > 0 and final_global_step_reached >= max_train_steps:
                if rank == 0: logger.info(
                    f"Max steps ({max_train_steps}) already met before starting epoch {epoch + 1}. Stopping.")
                break

            if rank == 0: logger.info(
                f"--- Starting Epoch {epoch + 1}/{args.num_train_epochs} (Current Global Step: {final_global_step_reached}) ---")

            final_global_step_reached = train_epoch(
                args, model, optimizer, lr_scheduler, scaler, train_dataloader,
                train_sampler, epoch, final_global_step_reached, device, rank, world_size,
                run, tokenizer,  # Pass Neptune run object
                max_train_steps, log_save_steps_set
            )

            if max_train_steps > 0 and final_global_step_reached >= max_train_steps:
                if rank == 0: logger.info(
                    f"Max steps ({max_train_steps}) met during epoch {epoch + 1}. Final step: {final_global_step_reached}. Stopping.")
                break

        training_duration_secs = time.time() - training_start_time
        if rank == 0:
            logger.info("***** Training Finished *****")
            logger.info(f"Total Training Time: {training_duration_secs:.2f} seconds")
            logger.info(f"Final Global Optimizer Step Reached: {final_global_step_reached}")

        # Final save operations (Rank 0 Only)
        if rank == 0:
            # Save the state at the very end of training, identified by final_global_step_reached
            # This might overwrite an existing checkpoint if max_steps ended on a save_step interval.
            # This is usually desired, to have the LATEST state.
            final_checkpoint_dir_name = f"checkpoint-{final_global_step_reached}"
            logger.info(
                f"Saving final training state at step {final_global_step_reached} to directory: {final_checkpoint_dir_name}")
            save_checkpoint(args, model, optimizer, lr_scheduler, scaler,
                            epoch if 'epoch' in locals() else args.num_train_epochs - 1,
                            # Last completed/attempted epoch
                            final_global_step_reached, rank, tokenizer)

            # Also save to a distinct 'final_model' directory for easy access to the *absolute final* model
            final_model_output_path = Path(args.output_dir) / "final_model"
            final_model_output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving final model artifacts to distinct directory: {final_model_output_path}")

            model_to_save_final = model.module if hasattr(model, 'module') else model
            model_to_save_final.save_pretrained(str(final_model_output_path))
            tokenizer.save_pretrained(str(final_model_output_path))

            # Save training args to final_model directory
            final_args_to_save = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            with open(final_model_output_path / "training_args.json", "w") as f_args_final:
                json.dump(final_args_to_save, f_args_final, indent=4)

            # Create sentinel for final_model
            if args.checkpoint_ready_sentinel:
                final_model_sentinel_path = final_model_output_path / args.checkpoint_ready_sentinel
                try:
                    with open(final_model_sentinel_path, 'w') as f_fm_sentinel:
                        f_fm_sentinel.write(
                            f"Final model (step {final_global_step_reached}) written at {datetime.datetime.now().isoformat()}.\n")
                    logger.info(f"Created ready sentinel for final_model: {final_model_sentinel_path}")
                except IOError as e_fm_sentinel:
                    logger.error(f"Failed to create ready sentinel for final_model: {e_fm_sentinel}")

            if NEPTUNE_AVAILABLE and run:
                try:
                    # Upload the checkpoint-final_global_step directory
                    last_ckpt_path_on_disk = Path(args.output_dir) / f"checkpoint-{final_global_step_reached}"
                    if last_ckpt_path_on_disk.is_dir():
                        run[f"checkpoints/step_{final_global_step_reached}"].upload_files(str(last_ckpt_path_on_disk))

                    # Upload final_model directory
                    if final_model_output_path.is_dir():
                        run["final_model_artifacts"].upload_files(str(final_model_output_path))

                    run["training_summary/duration_seconds"] = training_duration_secs
                    run["training_summary/final_global_step"] = final_global_step_reached
                    logger.info("Logged final training summary and artifacts to Neptune.")
                except Exception as e_neptune_final:
                    logger.warning(f"Neptune final artifact upload/log failed: {e_neptune_final}", exc_info=True)

            # Final local evaluation run on the absolute final model state (if enabled)
            if args.local_eval and (args.trigger_standard_eval or args.trigger_priming_eval):
                logger.info(
                    f"Performing one last local evaluation on the final model state (from step {final_global_step_reached}).")
                # Use the final_model_output_path for this evaluation, as it's the canonical "final"
                if final_model_output_path.is_dir():  # Ensure it exists
                    # Check for its sentinel before eval
                    final_model_sentinel_for_eval = final_model_output_path / args.checkpoint_ready_sentinel
                    if final_model_sentinel_for_eval.is_file():
                        run_local_evaluation(args, final_model_output_path, final_global_step_reached,
                                             rank)  # Pass final_global_step for context
                    else:
                        logger.warning(
                            f"Sentinel for final_model at {final_model_sentinel_for_eval} not found. Skipping final local eval.")
                else:
                    logger.warning(
                        f"Final model directory {final_model_output_path} not found for final local evaluation. This shouldn't happen.")

    except KeyboardInterrupt:
        if rank == 0: logger.warning("Training interrupted by user (KeyboardInterrupt).")
        if NEPTUNE_AVAILABLE and run and rank == 0: run["status/training"] = "interrupted_keyboard"
    except Exception as e_train_loop:
        if rank == 0: logger.critical(f"Unhandled exception during training loop: {e_train_loop}", exc_info=True)
        if NEPTUNE_AVAILABLE and run and rank == 0: run["critical_errors/training_loop"] = traceback.format_exc()
    finally:
        if is_distributed:
            torch.distributed.destroy_process_group()
        if rank == 0 and NEPTUNE_AVAILABLE and run:
            try:
                logger.info("Stopping Neptune run...")
                run.sync()  # Ensure all data is sent
                run.stop()
                logger.info("Neptune run stopped.")
            except Exception as e_neptune_stop:
                logger.error(f"Neptune stop operation failed: {e_neptune_stop}", exc_info=True)

        if rank == 0: logger.info(f"Training script (train.py) finished on Rank {rank}.")


if __name__ == "__main__":
    # Basic config for logging before DDP setup or if script is run directly without full setup.
    # This will be overridden by setup_logging once rank is known.
    # Log to stdout by default for better capture in Slurm logs.
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        stream=sys.stdout,  # Explicitly log to stdout
                        force=True)  # Override any root handlers

    # Attempt to import tqdm globally if needed, or handle its absence.
    try:
        from tqdm.auto import tqdm
    except ImportError:
        _rank_for_tqdm_fallback = int(os.environ.get("RANK", "0"))  # Check rank for fallback message
        if _rank_for_tqdm_fallback == 0: print(
            "Warning: tqdm.auto not installed. Progress bars will be basic or disabled.")


        def _fallback_tqdm(iterable, *args, **kwargs):
            return iterable


        tqdm = _fallback_tqdm

    main()