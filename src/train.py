# src/train.py (Modified for Frequent Directory Eval & CSV Logging)

# === Imports ===
import logging
import argparse
from pathlib import Path
import csv  # For CSV writing
import os   # For directory creation

# Import ML/data libraries later inside main for structure

# Define logger globally, but assign it *after* setup_logging in main
logger = None

# --- Function Definitions ---

def parse_args():
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train a GPT-2 like model using preprocessed Arrow datasets.")

    # Add evaluation mode flag
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        help="If set, run only evaluation on a specified checkpoint and exit."
    )
    parser.add_argument(
        "--skip_standard_eval",
        action="store_true",
        default=False,
        help="Skip standard evaluation (perplexity) during training or evaluation."
    )
    parser.add_argument(
        "--eval_max_samples",
        type=int,
        default=50000,
        help="Maximum number of samples for standard evaluation. Default: 50,000. <= 0 uses full dataset"
    )

    # === Essential Paths ===
    parser.add_argument("--train_dataset_path", type=str, default=None, help="Path to the training Arrow dataset. Required for training.")
    parser.add_argument("--validation_dataset_path", type=str, default=None, help="Path to the validation Arrow dataset. Required for standard eval.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for checkpoints, logs, final model, and CSV results. Required for training or CSV logging.")
    parser.add_argument("--model", type=str, default="gpt2", help="Model identifier from Hub or local path.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to resume/evaluate.")

    # === Training Hyperparameters ===
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Train batch size per device.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Standard eval batch size per device.")
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
    parser.add_argument("--eval_steps", type=int, default=500, help="Run standard eval every X steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps.")

    # === Neptune Logging ===
    parser.add_argument("--neptune_project", type=str, default=None, help="Neptune project name. Disables if None.")
    parser.add_argument("--neptune_tags", type=str, nargs='+', default=None, help="Optional Neptune tags.")
    parser.add_argument("--neptune_run_name", type=str, default=None, help="Optional Neptune run name.")

    # === Priming Evaluation Arguments ===
    parser.add_argument("--run_priming_eval", action="store_true", default=False, help="Enable priming evaluation from directory.")
    parser.add_argument("--priming_eval_dir_path", type=str, default=None, help="DIRECTORY containing priming CSVs (required if --run_priming_eval).")
    parser.add_argument("--priming_eval_steps", type=int, default=None, help="Run priming eval every X steps. Defaults to --eval_steps.")
    parser.add_argument("--priming_per_device_eval_batch_size", type=int, default=None, help="Priming eval batch size. Defaults to --per_device_eval_batch_size.")
    parser.add_argument("--priming_delimiter", type=str, default=".", help="Delimiter in priming CSVs.")
    parser.add_argument("--priming_eval_max_samples_per_file", type=int, default=1000, help="Maximum number of samples to use from each priming CSV file. Set <= 0 to use all samples. Reduces eval time.")

    args = parser.parse_args()

    # Set defaults
    if args.priming_eval_steps is None: args.priming_eval_steps = args.eval_steps
    if args.priming_per_device_eval_batch_size is None: args.priming_per_device_eval_batch_size = args.per_device_eval_batch_size

    # Validation
    if args.evaluate_only:
        if not args.checkpoint_path: parser.error("--checkpoint_path required for --evaluate_only.")
        if not Path(args.checkpoint_path).is_dir(): parser.error(f"Checkpoint not found: {args.checkpoint_path}")
        if not args.skip_standard_eval and not args.validation_dataset_path: parser.error("--validation_dataset_path required for standard eval.")
        if args.run_priming_eval:
            if not args.priming_eval_dir_path: parser.error("--priming_eval_dir_path required for priming eval.")
            if not args.output_dir: parser.error("--output_dir required for saving priming CSV results.") # Need output dir for CSV
    else: # Training
        if not args.train_dataset_path: parser.error("--train_dataset_path required for training.")
        if not args.output_dir: parser.error("--output_dir required for training.") # Output dir needed for ckpts, logs, CSV
        if not Path(args.train_dataset_path).is_dir(): parser.error(f"Train dataset not found: {args.train_dataset_path}")
        if not args.skip_standard_eval and not args.validation_dataset_path: parser.error("--validation_dataset_path required for standard eval.")
        if args.run_priming_eval:
            if not args.priming_eval_dir_path: parser.error("--priming_eval_dir_path required for priming eval.")
            # Output dir check already done above

    if args.validation_dataset_path and not Path(args.validation_dataset_path).is_dir(): parser.error(f"Validation dataset not found: {args.validation_dataset_path}")
    if args.priming_eval_dir_path and not Path(args.priming_eval_dir_path).is_dir(): parser.error(f"Priming dir not found: {args.priming_eval_dir_path}")
    if args.output_dir and args.run_priming_eval and not Path(args.output_dir).exists():
         print(f"Warning: Output directory {args.output_dir} does not exist. It will be created.")
         # No error here, will create later

    return args


def get_device():
    """Gets the appropriate device for PyTorch computations."""
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
    global logger
    fmt = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s"
    dfmt = "%Y-%m-%d %H:%M:%S"
    level = log_level if rank == 0 else logging.CRITICAL + 1
    logging.basicConfig(level=level, format=fmt, datefmt=dfmt, force=True)
    logger = logging.getLogger(__name__)
    if rank == 0: logger.info("Logging setup complete (Rank 0).")
    else: logger.disabled = True # Keep logger object but disable output

def set_seed(seed_value):
    """Sets random seeds."""
    import random; import numpy as np; import torch; global logger
    random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)
    try:
        if not logger.disabled: logger.info(f"Set random seed: {seed_value}")
    except (NameError, AttributeError): print(f"Set random seed: {seed_value} (logger N/A or disabled)")

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

def load_standard_data(args, is_distributed, rank, world_size, data_collator, mode='train'):
    """Loads standard Arrow datasets."""
    global logger
    import numpy as np
    from datasets import load_from_disk
    from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
    train_dl, eval_dl, train_sampler = None, None, None

    if mode == 'train':
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

    if not args.skip_standard_eval:
        if not args.validation_dataset_path:
            if rank == 0: logger.error("Missing validation path but standard eval not skipped.");
            raise ValueError("Missing path.")
        if rank == 0: logger.info(f"Loading validation data: {args.validation_dataset_path}")
        try:
            ds = load_from_disk(args.validation_dataset_path)
            original_size = len(ds)
            if rank == 0: logger.info(f"Full Eval dataset size: {original_size:,} sequences")

            # --- Sampling Logic ---
            if args.eval_max_samples is not None and args.eval_max_samples > 0 and args.eval_max_samples < original_size:
                if rank == 0: logger.info(
                    f"Sampling {args.eval_max_samples:,} sequences from validation set for standard evaluation (seed: {args.seed}).")
                # Use a seeded random number generator for reproducibility
                rng = np.random.RandomState(args.seed)
                sampled_indices = rng.choice(original_size, size=args.eval_max_samples, replace=False)
                ds = ds.select(sampled_indices)
                if rank == 0: logger.info(f"Using subset for Eval: {len(ds):,} sequences")
            elif args.eval_max_samples is not None and args.eval_max_samples > 0:
                 if rank == 0: logger.info(
                    f"Eval_max_samples ({args.eval_max_samples:,}) >= dataset size ({original_size:,}). Using full validation set.")
            else: # None or <= 0
                 if rank == 0: logger.info("Eval_max_samples <= 0 or not set. Using full validation set.")
            # --- End Sampling Logic ---

            # Always use SequentialSampler for evaluation, even on the subset
            # DDP doesn't need special sampler for eval if we aggregate results later (which evaluate_standard does)
            sampler = SequentialSampler(ds)
            if rank == 0: logger.info("Using SequentialSampler for standard eval.")
            eval_dl = DataLoader(ds, sampler=sampler, batch_size=args.per_device_eval_batch_size,
                                 num_workers=args.num_workers, pin_memory=True, collate_fn=data_collator)
            if rank == 0: logger.info("Standard Eval DataLoader created.")
        except Exception as e:
            logger.error(f"Validation data load/sampling fail: {e}", exc_info=True); raise
    else:
        if rank == 0: logger.info("Skipping standard eval data loading.")

    return train_dl, eval_dl, train_sampler


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
        return # Cannot save if dir creation failed

    # Unwrap model for saving state_dict and using save_pretrained
    unwrapped_model = model.module if hasattr(model, 'module') else model

    # Prepare state dict
    state = {
        'model': unwrapped_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'scaler': scaler.state_dict() if scaler.is_enabled() else None,
        'epoch': epoch,
        'global_step': global_step,
        'args': vars(args), # Save args used for this training run
        # Save RNG states for reproducibility
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
        'torch_cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }
    state_file = ckpt_dir / "training_state.pt"

    # Save state dict
    try:
        torch.save(state, state_file)
        logger.info(f"Saved training state to: {state_file}")
    except Exception as e:
        logger.error(f"Failed to save training state: {e}", exc_info=True)

    # Save model weights, config, tokenizer using HF methods
    try:
        unwrapped_model.save_pretrained(ckpt_dir)
        logger.info(f"Saved model weights and config to {ckpt_dir}")
        if tokenizer:
            tokenizer.save_pretrained(ckpt_dir)
            logger.info(f"Saved tokenizer to {ckpt_dir}")
    except Exception as e:
        logger.error(f"Failed to save model/tokenizer using save_pretrained: {e}", exc_info=True)


def load_model_for_evaluation(model_class, checkpoint_path, base_model_name="gpt2"):
    """Loads model, tokenizer, and config from checkpoint for evaluation."""
    global logger
    from transformers import AutoTokenizer, AutoConfig; from pathlib import Path
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
            added_tokens = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.warning(f"Added pad_token '[PAD]' to tokenizer ({added_tokens} new token(s)).")
            # If we added tokens, we might need to resize embeddings, although from_pretrained usually handles this
            if added_tokens > 0 and len(tokenizer) > model.config.vocab_size:
                 logger.info(f"Resizing model token embeddings from {model.config.vocab_size} to {len(tokenizer)}.")
                 model.resize_token_embeddings(len(tokenizer))
                 # Update config vocab size to match potentially resized model
                 config.vocab_size = model.config.vocab_size

    return model, tokenizer, config


def evaluate_standard(args, model, eval_dataloader, device, rank, world_size):
    """Runs standard evaluation (perplexity). Handles DDP aggregation."""

    global logger

    if eval_dataloader is None:
        if rank == 0: logger.warning("Standard evaluation dataloader is None. Skipping standard eval.")
        return {}

    import torch; from tqdm.auto import tqdm; import numpy as np; import math
    from torch.cuda.amp import autocast # Use specific import

    original_mode = model.training
    model.eval()

    total_loss = torch.tensor(0.0, device=device)
    total_items = torch.tensor(0, dtype=torch.long, device=device) # Use long for item count

    is_dist = torch.distributed.is_initialized()
    is_rank0 = rank == 0
    disable_tqdm = not is_rank0

    if is_rank0: logger.info("Starting standard evaluation...")
    progress_bar = tqdm(eval_dataloader, desc="Eval (Std)", leave=False, disable=disable_tqdm)

    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            try:
                batch_on_device = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            except RuntimeError as e:
                logger.error(f"Error moving standard eval batch to device {device}: {e}")
                continue # Skip batch

            # Forward pass
            try:
                 # Enable AMP context based on args and device type
                amp_enabled = args.use_amp and device.type == 'cuda'
                with autocast(enabled=amp_enabled):
                    outputs = model(**batch_on_device)
                    loss = outputs.loss # The model calculates the avg loss for the batch
            except Exception as e:
                logger.error(f"Error during standard eval forward pass: {e}", exc_info=True)
                continue # Skip batch

            # Accumulate loss and count, handle non-finite values
            if loss is not None and torch.isfinite(loss):
                # outputs.loss is already averaged over batch items by HF LMHeadModel
                # We need the *sum* of losses for the batch to average correctly later
                # Number of items in batch (ignoring padding if possible, but difficult here, rely on HF loss)
                # Let's assume the loss is average per-token loss for the batch items.
                # To get total loss for averaging across all batches: multiply by batch size?
                # HF CausalLMOutput returns loss averaged over batch and sequence length (where labels != -100)
                # So, just summing this loss is not correct. We need to average it weighted by items.
                num_items_in_batch = batch_on_device['input_ids'].size(0)
                total_loss += loss.detach() * num_items_in_batch # Accumulate sum of losses
                total_items += num_items_in_batch                # Accumulate number of items
                if is_rank0:
                    progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
            elif is_rank0:
                logger.warning(f"Non-finite loss detected during standard evaluation: {loss.item() if loss is not None else 'None'}. Skipping batch contribution.")

    # Aggregate results across ranks if distributed
    if is_dist:
        torch.distributed.barrier() # Ensure all ranks finish processing batches
        logger.debug(f"Rank {rank} pre-reduce: total_loss={total_loss.item()}, total_items={total_items.item()}")
        torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_items, op=torch.distributed.ReduceOp.SUM)
        logger.debug(f"Rank {rank} post-reduce: total_loss={total_loss.item()}, total_items={total_items.item()}")

    # Calculate final metrics on Rank 0
    metrics = {}
    if is_rank0:
        if total_items.item() > 0:
            final_avg_loss = total_loss.item() / total_items.item()
            try:
                # Calculate perplexity, handle potential overflow
                perplexity = math.exp(final_avg_loss) if final_avg_loss < 700 else float('inf')
            except (OverflowError, ValueError):
                perplexity = float('inf') if final_avg_loss > 0 else float('nan') # Handle large positive or NaN loss

            logger.info(f"Standard Evaluation Results: Average Loss = {final_avg_loss:.4f}, Perplexity = {perplexity:.4f}, Total Items = {total_items.item()}")
            metrics = {"loss": final_avg_loss, "perplexity": perplexity, "total_items": total_items.item()}
        else:
            logger.warning("Standard evaluation completed, but total_items processed is zero. No metrics calculated.")
            metrics = {"loss": float('nan'), "perplexity": float('nan'), "total_items": 0}

    # Restore original model mode
    if original_mode:
        model.train()
    else:
        model.eval() # Ensure it stays in eval mode if it started that way

    if not disable_tqdm: progress_bar.close()
    if is_dist: torch.distributed.barrier() # Sync after calculation before proceeding

    return metrics


# --- Helper Function for Multi-CSV Priming Eval (MODIFIED FOR CSV LOGGING) ---
def run_priming_evaluation_on_directory(args, model, tokenizer, device, rank, run, global_step):
    """
    Finds CSVs, creates dataloaders, runs eval, aggregates summary results,
    and appends raw per-item results to a persistent CSV file (on Rank 0).
    Returns a dictionary of summary metrics per file.
    """

    global logger

    from pathlib import Path
    import numpy as np
    import math
    import torch  # For distributed check

    if not args.run_priming_eval or not args.priming_eval_dir_path:
        # If rank 0, log that it's skipped due to config
        if rank == 0: logger.info("Skipping priming evaluation (not enabled or path not provided).")
        return {}
    if not args.output_dir: # Need output dir for saving CSV
        if rank == 0: logger.error("Output directory (--output_dir) required for saving priming results CSV but not provided. Skipping priming CSV log.")
        # Can still proceed with calculation and Neptune logging, just skip CSV
        # Fall through, but csv_output_path will be None below
    else:
        # Ensure output dir exists or can be created (only on rank 0)
        if rank == 0:
            try:
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create output directory {args.output_dir}: {e}. Skipping priming CSV log.")
                # Disable CSV logging by setting output_dir to None effectively for CSV part
                args.output_dir = None # Modify local copy of args? Risky. Better to handle via csv_output_path.



    # Import priming libs here to keep them contained and fail gracefully if missing
    try:
        from priming_evaluation.data_loader import create_priming_dataloader
        # Assumes evaluator.py has the modified run_native_priming_eval returning (summary, raw)
        from priming_evaluation.evaluator import run_native_priming_eval
    except ImportError as e:
        if rank == 0: logger.error(f"Failed to import priming evaluation library: {e}. Cannot run priming eval.")
        return {"error": "Priming library import failed."}


    priming_dir = Path(args.priming_eval_dir_path)
    if not priming_dir.is_dir():
        if rank == 0: logger.error(f"Priming evaluation directory not found: {priming_dir}")
        return {"error": f"Priming directory not found: {priming_dir}"}

    csv_files = sorted(list(priming_dir.glob('*.csv')))
    if not csv_files:
        if rank == 0: logger.warning(f"No *.csv files found in priming directory: {priming_dir}")
        return {} # Return empty dict, not an error


    # --- CSV Setup (Rank 0 Only) ---
    csv_output_dir = None
    csv_output_path = None
    if rank == 0 and args.output_dir: # Check if output_dir is usable
        try:
            csv_output_dir = Path(args.output_dir) / "prime_measures"
            os.makedirs(csv_output_dir, exist_ok=True) # Create directory if needed
            csv_output_path = csv_output_dir / "results.csv"
            logger.info(f"Raw priming results CSV will be appended to: {csv_output_path}")
        except OSError as e:
            logger.error(f"Could not create directory for priming CSV log: {csv_output_dir}. Error: {e}")
            csv_output_path = None # Disable CSV logging if dir creation fails


    all_priming_summary_results = {} # Store the aggregated metrics per file
    if rank == 0: logger.info(f"Found {len(csv_files)} CSVs for priming eval in {priming_dir}.")
    is_distributed = torch.distributed.is_initialized()
    original_mode = model.training # Store original mode

    # --- CSV File Handling (Open once, write iteratively, close at end) ---
    csv_writer = None
    csv_file_handle = None
    if rank == 0 and csv_output_path: # Only attempt if path setup was successful
        try:
            # Open file in append mode ('a'), create if doesn't exist
            # Use newline='' to prevent extra blank rows in CSV
            csv_file_handle = open(csv_output_path, 'a', newline='', encoding='utf-8')
            csv_writer = csv.writer(csv_file_handle)
            # Write header only if the file is newly created (or empty)
            # Check position: 0 means empty file or just opened
            if csv_file_handle.tell() == 0:
                header = [
                    "eval_num", "corpus_file", "target_structure",
                    "item_index", "pe", "logp_con", "logp_incon"
                ]
                csv_writer.writerow(header)
                logger.info("Wrote header to new/empty priming results CSV.")
        except IOError as e:
            logger.error(f"Failed to open or write header to CSV {csv_output_path}: {e}")
            csv_writer = None # Disable writing if open failed
            if csv_file_handle:
                try: csv_file_handle.close() # Ensure file is closed if opened
                except Exception: pass # Ignore errors during cleanup close
            csv_file_handle = None


    # --- Process each CSV file ---
    model.eval() # Ensure model is in eval mode for all files
    for csv_path in csv_files:
        csv_filename = csv_path.name
        if rank == 0: logger.info(f"--- Running Priming Eval for: {csv_filename} (Step {global_step}) ---")

        priming_dataloader_single = None
        # Sync before processing each file to ensure consistent state if needed (e.g., RNG)
        if is_distributed: torch.distributed.barrier()

        try:
            # Create dataloader for the single CSV file
            priming_dataloader_single = create_priming_dataloader(
                csv_path=str(csv_path),
                tokenizer=tokenizer,
                batch_size=args.priming_per_device_eval_batch_size,
                delimiter=args.priming_delimiter,
                num_workers=args.num_workers,
                pin_memory=True,
                max_samples=args.priming_eval_max_samples_per_file, # Pass sampling arg
                seed=args.seed # Pass seed for reproducible sampling
            )
        except Exception as e:
            if rank == 0: logger.error(f"Dataloader creation failed for {csv_filename}: {e}", exc_info=True)
            all_priming_summary_results[csv_filename] = {"error": f"Dataloader creation failed: {e}"}
            continue # Skip to next file

        if priming_dataloader_single is None or len(priming_dataloader_single.dataset) == 0:
             if rank == 0: logger.warning(f"Dataloader for {csv_filename} is None or empty. Skipping.")
             all_priming_summary_results[csv_filename] = {"error": "Dataloader None or empty."}
             continue # Skip to next file


        # --- Run the evaluation ---
        try:
            # run_native_priming_eval returns (summary_dict, raw_results_dict)
            # It handles internal looping, aggregation, and AMP usage
            # Pass use_amp flag down
            priming_summary_metrics, priming_raw_results = run_native_priming_eval(
                model=model,
                priming_dataloader=priming_dataloader_single,
                device=device,
                tokenizer=tokenizer, # Pass tokenizer for potential debug inside
                use_amp=args.use_amp # Pass AMP setting from args
            )

            # Store summary results (for JSON output / function return)
            all_priming_summary_results[csv_filename] = priming_summary_metrics
            if rank == 0: logger.info(f"Priming Summary Metrics for {csv_filename}: {priming_summary_metrics}")

            # --- Log Summary to Neptune (Rank 0 Only) ---
            if rank == 0 and run: # Check if Neptune run object exists
                 log_prefix = f"eval/priming/{csv_filename.replace('.', '_').replace('/','_')}" # Sanitize name
                 try:
                     metrics_to_log = {k: v for k, v in priming_summary_metrics.items() if isinstance(v, (int, float)) and math.isfinite(v)}
                     if metrics_to_log:
                         # Use append for step-based logging
                         for k, v in metrics_to_log.items():
                             run[f"{log_prefix}/{k}"].append(v, step=global_step)
                         logger.info(f"Logged {len(metrics_to_log)} summary priming metrics for {csv_filename} to Neptune @ step {global_step}.")
                     else:
                        logger.info(f"No finite summary metrics to log to Neptune for {csv_filename}.")

                 except Exception as e:
                     logger.warning(f"Neptune logging failed for {csv_filename} summary metrics: {e}")

            # --- Write Raw Results to CSV (Rank 0 Only) ---
            if rank == 0 and csv_writer and priming_raw_results: # Check writer exists and results are not empty
                logger.info(f"Appending {sum(len(v) for v in priming_raw_results.values()):,} raw priming results to CSV for {csv_filename}...")
                items_written_count = 0
                try:
                    for target_structure, results_list in priming_raw_results.items():
                        for idx, item_data in enumerate(results_list):
                            # Check if item_data is the expected dictionary format
                            if isinstance(item_data, dict):
                                # Extract values safely using .get() with NaN default
                                pe_val = item_data.get('pe', float('nan'))
                                logp_con_val = item_data.get('logp_con', float('nan'))
                                logp_incon_val = item_data.get('logp_incon', float('nan'))

                                row = [
                                    global_step,              # eval_num (current global step)
                                    csv_filename,             # corpus_file name
                                    target_structure,         # target_structure string
                                    idx,                      # item_index (0-based index within this structure/file list)
                                    f"{pe_val:.6f}" if not math.isnan(pe_val) else 'NaN',             # pe formatted
                                    f"{logp_con_val:.6f}" if not math.isnan(logp_con_val) else 'NaN', # logp_con formatted
                                    f"{logp_incon_val:.6f}" if not math.isnan(logp_incon_val) else 'NaN'# logp_incon formatted
                                ]
                                csv_writer.writerow(row)
                                items_written_count += 1
                            else:
                                logger.warning(f"Skipping invalid item data format in raw results for {target_structure} index {idx} in {csv_filename}: Type={type(item_data)}")
                    # Flush after writing all rows for a file to ensure data is written
                    if csv_file_handle: csv_file_handle.flush()
                    logger.info(f"Finished appending {items_written_count} rows to CSV for {csv_filename}.")

                except Exception as e:
                     logger.error(f"Error occurred while writing raw results to CSV for {csv_filename}: {e}", exc_info=True)
                     # Attempt to continue to next file, but CSV might be corrupted

        except Exception as e:
            # Catch errors during the run_native_priming_eval call itself
            if rank == 0: logger.error(f"Priming evaluation run failed for {csv_filename}: {e}", exc_info=True)
            all_priming_summary_results[csv_filename] = {"error": f"Evaluation run failed: {e}"}
        finally:
            # Explicitly delete dataloader to free memory, especially important in loops
            del priming_dataloader_single
            import gc; gc.collect() # Optional: Force garbage collection

    # --- Close CSV File (Rank 0 Only) ---
    if rank == 0 and csv_file_handle:
        try:
            csv_file_handle.close()
            logger.info("Closed priming results CSV file.")
        except Exception as e:
            logger.error(f"Error closing CSV file {csv_output_path}: {e}")


    if is_distributed: torch.distributed.barrier() # Sync after all files are done
    # Restore original model mode (evaluator might have left it in eval)
    if original_mode:
        model.train()
    else:
        model.eval()

    if rank == 0: logger.info("--- Finished All Priming Evaluations for this step ---")

    # Return only the summary results dictionary (per file)
    # Raw results were saved to CSV by rank 0 if enabled/possible
    return all_priming_summary_results


# --- Modified train_epoch (No structural changes needed here, just calls the modified helper) ---
def train_epoch(args, model, optimizer, lr_scheduler, scaler, train_dataloader, eval_dataloader,
                train_sampler, epoch, global_step, device, rank, world_size, run, tokenizer):
    """Runs one training epoch with periodic standard and multi-CSV priming eval."""
    global logger
    import torch; from torch.utils.data import DistributedSampler; from tqdm.auto import tqdm; import math; import sys; import gc
    from torch.nn.utils import clip_grad_norm_
    from torch.cuda.amp import autocast # Use specific import

    model.train() # Ensure model is in training mode at the start of epoch
    is_distributed = train_sampler is not None and isinstance(train_sampler, DistributedSampler)
    # Set epoch for distributed sampler shuffling
    if is_distributed:
        try:
            train_sampler.set_epoch(epoch)
            if rank == 0: logger.debug(f"Set DistributedSampler epoch to {epoch}")
        except AttributeError:
             if rank == 0: logger.warning("Train sampler does not have set_epoch method.")


    disable_tqdm = rank != 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}", leave=True, disable=disable_tqdm, position=0)
    total_loss_since_logging, steps_since_logging = 0.0, 0
    last_logged_loss = float('inf') # For progress bar display

    for step, batch in enumerate(progress_bar):
        # --- Training Step Logic ---
        try:
            batch_on_device = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        except RuntimeError as e:
            logger.error(f"Error moving training batch to device {device} at step {global_step}: {e}")
            optimizer.zero_grad(set_to_none=True) # Prevent using stale grads
            continue # Skip batch

        try:
            # Use AMP context manager
            amp_enabled = args.use_amp and device.type == 'cuda'
            with autocast(enabled=amp_enabled):
                outputs = model(**batch_on_device)
                loss = outputs.loss # Model calculates average loss for the batch

            if loss is None:
                logger.error(f"Loss returned as None at step {global_step}. Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            # Check for non-finite loss *before* scaling
            if not torch.isfinite(loss):
                logger.warning(f"Non-finite loss detected ({loss.item()}) before scaling at step {global_step}. Skipping gradient update.")
                optimizer.zero_grad(set_to_none=True) # Zero grads even if update is skipped
                # Optionally, could try reducing scaler state here if using AMP and loss explodes
                # if scaler.is_enabled() and scaler.get_scale() > 1.0:
                #     scaler.update(scaler.get_scale() / 2.0) # Example: Halve scale factor
                continue

            # Scale loss for accumulation and backward pass
            scaled_loss = loss / args.gradient_accumulation_steps
            current_loss_value = loss.item() # Get scalar value for logging

        except Exception as e:
            logger.error(f"Error during training forward pass at step {global_step}: {e}", exc_info=True)
            optimizer.zero_grad(set_to_none=True) # Zero grads on error
            continue # Skip batch

        # Accumulate gradients
        # Use scaler.scale() only if AMP is enabled
        if amp_enabled:
             scaler.scale(scaled_loss).backward()
        else:
             scaled_loss.backward()

        total_loss_since_logging += current_loss_value
        steps_since_logging += 1
        last_logged_loss = current_loss_value # Update loss for progress bar display

        # --- Optimizer Step, Scheduler Step, Grad Clipping ---
        if (step + 1) % args.gradient_accumulation_steps == 0:
            try:
                # Gradient Clipping (before optimizer step)
                if args.max_grad_norm and args.max_grad_norm > 0:
                    # Unscale gradients before clipping if using AMP scaler
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)

                    # Determine parameters to clip (handle DDP wrapper)
                    params_to_clip = model.module.parameters() if hasattr(model, 'module') else model.parameters()
                    clip_grad_norm_(params_to_clip, args.max_grad_norm)

                # Optimizer step (using scaler if enabled)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update() # Update scaler for next iteration
                else:
                    optimizer.step() # Normal optimizer step

                lr_scheduler.step() # Update learning rate
                optimizer.zero_grad(set_to_none=True) # Clear gradients for next accumulation
                global_step += 1 # Increment global step only after optimizer step

            except Exception as e:
                logger.error(f"Error during optimizer step/clipping/lr_scheduler at step {global_step}: {e}", exc_info=True)
                optimizer.zero_grad(set_to_none=True) # Ensure grads are zeroed even if step failed
                # Continue training? Or break? Depends on severity. Continue for now.

            # --- Logging Step ---
            if rank == 0 and global_step % args.logging_steps == 0 and steps_since_logging > 0:
                 avg_loss = total_loss_since_logging / steps_since_logging
                 lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, 'get_last_lr') and lr_scheduler.get_last_lr() else optimizer.param_groups[0]['lr']
                 logger.info(f"Epoch {epoch+1} | Step {global_step}: Avg Train Loss = {avg_loss:.4f}, LR = {lr:.6e}")

                 # Neptune logging (if enabled)
                 if run:
                     try:
                         if math.isfinite(avg_loss): run["train/step_loss"].append(avg_loss, step=global_step)
                         if math.isfinite(lr): run["train/learning_rate"].append(lr, step=global_step)
                         if torch.cuda.is_available(): run["train/gpu_mem_alloc_gb"].append(torch.cuda.memory_allocated(device)/1e9, step=global_step)
                         if scaler.is_enabled(): run["train/grad_scale"].append(scaler.get_scale(), step=global_step)
                     except Exception as e:
                         logger.warning(f"Neptune train logging failed at step {global_step}: {e}")

                 # Reset counters for next logging interval
                 total_loss_since_logging, steps_since_logging = 0.0, 0

            # --- Evaluation Step ---
            # Check if it's time to run standard or priming evaluation
            run_std_eval_now = not args.skip_standard_eval and eval_dataloader and global_step > 0 and global_step % args.eval_steps == 0
            run_prime_eval_now = args.run_priming_eval and global_step > 0 and global_step % args.priming_eval_steps == 0

            if run_std_eval_now or run_prime_eval_now:
                if rank == 0: logger.info(f"--- Starting Periodic Evaluation @ Step {global_step} ---")
                # Sync processes before evaluation starts
                if is_distributed: torch.distributed.barrier()

                original_train_mode = model.training # Remember current mode (should be train)
                model.eval() # Set model to evaluation mode

                # --- Standard Evaluation ---
                if run_std_eval_now:
                    if rank == 0: logger.info("--- Running Periodic Standard Evaluation ---")
                    std_metrics = evaluate_standard(args, model, eval_dataloader, device, rank, world_size)
                    # Standard metrics logging (e.g., Neptune) is handled inside evaluate_standard on rank 0
                    if rank == 0 and run and std_metrics: # Check if Neptune run exists and metrics were returned
                        try:
                            loss_val = std_metrics.get("loss", float('nan')); ppl_val = std_metrics.get("perplexity", float('nan'))
                            if math.isfinite(loss_val): run["eval/loss"].append(loss_val, step=global_step)
                            if math.isfinite(ppl_val): run["eval/perplexity"].append(ppl_val, step=global_step)
                            logger.info(f"Logged periodic standard eval metrics to Neptune @ step {global_step}.")
                        except Exception as e:
                            logger.warning(f"Neptune periodic standard eval log failed at step {global_step}: {e}")
                elif global_step % args.eval_steps == 0: # Log skip only if it *would* have run based on steps
                    if rank == 0: logger.info("--- Skipping Periodic Standard Evaluation (per config or dataloader missing) ---")

                # --- Priming Evaluation ---
                if run_prime_eval_now:
                    if rank == 0: logger.info("--- Running Periodic Priming Evaluation on Directory ---")
                    # This helper function now handles:
                    # - Finding CSVs
                    # - Creating dataloaders
                    # - Calling the core evaluator (which returns summary + raw)
                    # - Logging summary metrics to Neptune (rank 0)
                    # - Saving raw metrics to CSV (rank 0)
                    # - Returns only the summary metrics dict
                    priming_metrics_all_summary = run_priming_evaluation_on_directory(
                        args, model, tokenizer, device, rank, run, global_step
                    )
                    # Console logging of summary happens inside the helper too
                elif global_step % args.priming_eval_steps == 0: # Log skip only if it *would* have run
                     if rank == 0: logger.info("--- Skipping Periodic Priming Evaluation (per config) ---")

                # --- Post-Evaluation ---
                # Sync processes after evaluation finishes
                if is_distributed: torch.distributed.barrier()
                # Restore original model mode (usually training)
                if original_train_mode:
                    model.train()
                else:
                     model.eval() # Should not happen if called from train_epoch, but safe
                if rank == 0: logger.info(f"--- Finished Periodic Evaluation, Resuming Train ---")
                gc.collect() # Clean up memory after eval
                if torch.cuda.is_available(): torch.cuda.empty_cache()


            # --- Save Checkpoint ---
            if global_step > 0 and global_step % args.save_steps == 0:
                # Sync before saving to ensure all processes are ready or done with eval
                if is_distributed: torch.distributed.barrier()
                # save_checkpoint function already includes rank 0 check
                save_checkpoint(args, model, optimizer, lr_scheduler, scaler, epoch, global_step, rank, tokenizer)
                # Sync after saving before resuming training
                if is_distributed: torch.distributed.barrier()

        # Update progress bar on rank 0
        if rank == 0:
            progress_bar.set_postfix({"loss": f"{last_logged_loss:.4f}", "step": global_step})

        # Check if max steps reached after incrementing global_step
        if max_train_steps > 0 and global_step >= max_train_steps:
             if rank == 0: logger.info(f"Maximum training steps ({max_train_steps}) reached or exceeded. Finishing epoch {epoch+1}.")
             break # Exit the inner loop (batches)


    # End of Epoch
    if rank == 0:
        progress_bar.close()
        logger.info(f"--- Epoch {epoch+1} Finished (Reached Step {global_step}) ---")

    return global_step # Return the final global step count after the epoch


# --- Main Execution ---
def main():
    """Main function to parse arguments, set up, and run training or evaluation."""
    print("Importing standard libraries...")
    # Standard imports
    import math; import torch; import random; import numpy as np; import logging
    import json; import sys; import time; import traceback; import gc; import csv; import os
    from tqdm.auto import tqdm; from pathlib import Path
    print("Finished standard imports.")

    print("Importing ML/data libraries...")
    try:
        # ML imports
        from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
        from torch.nn.utils import clip_grad_norm_; from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.optim import AdamW; from transformers import GPT2LMHeadModel, AutoConfig, get_scheduler, AutoTokenizer, DataCollatorForLanguageModeling
        from torch.cuda.amp import GradScaler; import torch.amp # Use torch.amp namespace
        try:
             import neptune
             NEPTUNE_AVAILABLE = True
        except ImportError:
             print("Neptune.ai library not found, Neptune logging will be disabled.")
             # Create a placeholder if needed or just use 'run = None'
             sys.modules['neptune'] = None # Ensure it's None if import failed
             NEPTUNE_AVAILABLE = False
        from datasets import load_from_disk
        # Priming imports are handled inside run_priming_evaluation_on_directory to keep dependencies optional
        print("Finished ML/data imports.")
    except ImportError as e:
        print(f"ERROR: A required ML/data library failed to import: {e}", file=sys.stderr)
        print("Please ensure all dependencies (torch, transformers, datasets, numpy, tqdm, etc.) are installed.", file=sys.stderr)
        sys.exit(1)

    args = parse_args()
    is_distributed, rank, world_size, local_rank = setup_distributed(args)
    setup_logging(rank=rank) # Setup logging AFTER getting rank
    global logger # Assign the configured logger
    logger = logging.getLogger(__name__) # Get logger configured by setup_logging

    if rank == 0: logger.info(f"Script Mode: {'EVALUATION ONLY' if args.evaluate_only else 'TRAINING'}")
    if rank == 0: logger.info(f"Running with Arguments: {vars(args)}")

    # Setup Device and Seed
    device = get_device()
    # Set seed differently for each rank for potentially different initializations if needed, but synced later
    set_seed(args.seed + rank)

    # Neptune Setup (Rank 0 Only)
    run = None
    if rank == 0 and NEPTUNE_AVAILABLE and args.neptune_project:
        try:
            run = neptune.init_run(project=args.neptune_project, name=args.neptune_run_name, tags=args.neptune_tags)
            # Log hyperparameters (convert Path objects to strings)
            args_log = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            run["parameters"] = args_log
            logger.info(f"Neptune logging enabled. Run URL: {run.get_url()}")
        except Exception as e:
            logger.error(f"Neptune initialization failed: {e}. Neptune logging disabled.")
            run = None # Ensure run is None if init fails
    elif rank == 0:
        logger.info("Neptune logging disabled (Neptune not installed, or no project specified).")

    # === Model & Tokenizer Loading ===
    model, tokenizer, config = None, None, None # Initialize
    try:
        if args.evaluate_only:
            # load_model_for_evaluation handles loading from checkpoint
            model, tokenizer, config = load_model_for_evaluation(GPT2LMHeadModel, args.checkpoint_path, args.model)
        else: # Training mode (load base model or resume from checkpoint later)
            if rank == 0: logger.info(f"Loading base model and tokenizer: {args.model}")
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
            # Ensure pad token exists
            if tokenizer.pad_token is None:
                if tokenizer.eos_token:
                    tokenizer.pad_token = tokenizer.eos_token
                    if rank == 0: logger.info(f"Set tokenizer pad_token to eos_token ('{tokenizer.eos_token}')")
                else:
                    added_tokens = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    if rank == 0: logger.warning(f"Added pad_token '[PAD]' to tokenizer ({added_tokens} new token(s)).")

            config = AutoConfig.from_pretrained(args.model)
            config.use_cache = False # Disable cache for training
            model = GPT2LMHeadModel.from_pretrained(args.model, config=config)

            # Resize embeddings if tokenizer vocab size changed (e.g., added pad token)
            if len(tokenizer) > model.config.vocab_size:
                 if rank == 0: logger.info(f"Resizing model token embeddings from {model.config.vocab_size} to {len(tokenizer)}.")
                 model.resize_token_embeddings(len(tokenizer))
                 # Ensure model config also reflects the new size
                 config.vocab_size = model.config.vocab_size
                 if rank == 0: logger.info(f"Model vocab size updated to: {model.config.vocab_size}")

        # Move model to device
        model.to(device)
        if rank == 0: logger.info(f"Model '{args.model}' loaded successfully to device {device} (Rank {rank})")

        # Wrap model with DDP if distributed
        if is_distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
            if rank == 0: logger.info(f"Model wrapped with DistributedDataParallel (Rank {rank}).")

    except Exception as e:
        logger.critical(f"Fatal Error: Model or Tokenizer loading failed: {e}", exc_info=True)
        if run: run["error/message"] = f"Model/Tokenizer load failed: {e}"; run.stop()
        sys.exit(1)

    # === Standard Eval Data Loading ===
    eval_dataloader = None
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    if not args.skip_standard_eval:
        try:
            # load_standard_data returns train_dl, eval_dl, train_sampler
            # We only need eval_dl here. Train data is loaded later in training block.
            _, eval_dataloader, _ = load_standard_data(args, is_distributed, rank, world_size, data_collator, mode='eval')
            if rank == 0 and eval_dataloader: logger.info(f"Loaded standard evaluation dataset with {len(eval_dataloader.dataset):,} samples.")
        except Exception as e:
            if rank == 0: logger.error(f"Failed to load standard evaluation data: {e}. Skipping standard evaluation.", exc_info=True)
            args.skip_standard_eval = True # Force skip if loading failed
    else:
        if rank == 0: logger.info("Skipping standard validation dataset loading as per arguments.")


    # === Evaluation Only Mode ===
    if args.evaluate_only:
        if rank == 0:
             logger.info(f"***** RUNNING EVALUATION ONLY *****")
             logger.info(f"Using Checkpoint: {args.checkpoint_path}")
             logger.info(f"Output Directory: {args.output_dir}") # Log output dir where CSV will go

        std_metrics = {}
        prime_metrics_all_summary = {} # Store summary priming results

        # --- Run Standard Evaluation ---
        if not args.skip_standard_eval:
            if eval_dataloader:
                 if rank == 0: logger.info("--- Running Standard Evaluation ---")
                 std_metrics = evaluate_standard(args, model, eval_dataloader, device, rank, world_size)
                 # Logging to console/Neptune handled inside evaluate_standard
            else:
                 if rank == 0: logger.warning("Standard evaluation skipped (dataloader not available).")
        else:
            if rank == 0: logger.info("--- Skipping Standard Evaluation (per arguments) ---")

        # --- Run Priming Evaluation ---
        if args.run_priming_eval:
            if rank == 0: logger.info("--- Running Priming Evaluation on Directory ---")
            # Run eval, which internally saves raw to CSV (on rank 0) and returns summary dict
            # Use global_step=0 or a fixed indicator for eval-only mode
            prime_metrics_all_summary = run_priming_evaluation_on_directory(
                args, model, tokenizer, device, rank, run, global_step=-1 # Use -1 to indicate eval-only step
            )
            # Console/Neptune summary logging handled inside helper
        else:
            if rank == 0: logger.info("--- Skipping Priming Evaluation (per arguments) ---")

        # --- Save Summary Results (JSON on Rank 0) ---
        if rank == 0:
            logger.info("***** Evaluation Complete *****")
            results_summary = {}
            if not args.skip_standard_eval and std_metrics:
                 logger.info(f"Standard Eval Summary Metrics: {std_metrics}")
                 results_summary["standard_summary"] = std_metrics
            if args.run_priming_eval and prime_metrics_all_summary:
                 logger.info(f"Priming Eval Summary Metrics (All Files): {prime_metrics_all_summary}")
                 results_summary["priming_summary"] = prime_metrics_all_summary

            if results_summary:
                # Save JSON summary to the checkpoint's directory if specified, otherwise output_dir
                # Needs args.output_dir to be set if checkpoint_path isn't! Let's use output_dir as primary.
                if args.output_dir:
                     eval_output_dir = Path(args.output_dir)
                     eval_output_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists
                     res_file = eval_output_dir / "evaluation_results.json"
                     try:
                         with open(res_file, "w", encoding='utf-8') as f:
                              json.dump(results_summary, f, indent=4)
                         logger.info(f"Evaluation summary results saved to: {res_file}")

                         # Log summary to Neptune if enabled
                         if run:
                             try: run["evaluation/final_summary_results"] = results_summary
                             except Exception as ne: logger.warning(f"Failed to log final eval summary results to Neptune: {ne}")

                     except IOError as e: logger.error(f"Failed to save evaluation summary JSON: {e}")
                     except TypeError as e: logger.error(f"Failed to serialize evaluation summary to JSON: {e}")
                else:
                    logger.warning("No --output_dir specified, cannot save evaluation_results.json.")

            else:
                logger.warning("No evaluation summary metrics were generated to save.")

            # Remind user where CSV results are (if priming was run)
            if args.run_priming_eval and args.output_dir:
                csv_path = Path(args.output_dir) / 'prime_measures' / 'results.csv'
                logger.info(f"Raw priming results (if generated) were appended to: {csv_path}")


        # Clean up DDP and Neptune
        if is_distributed:
            torch.distributed.destroy_process_group()
        if rank == 0 and run:
             run.stop()
        logger.info(f"Evaluation script finished (Rank {rank}).")
        sys.exit(0)


    # === Training Mode ===
    else: # if not args.evaluate_only
        if rank == 0: logger.info(f"***** STARTING TRAINING *****")

        # Load Training Data
        train_dataloader, train_sampler = None, None # Initialize
        try:
            train_dataloader, _, train_sampler = load_standard_data(args, is_distributed, rank, world_size, data_collator, mode='train')
            if rank == 0: logger.info(f"Loaded training dataset with {len(train_dataloader.dataset):,} samples.")
        except Exception as e:
            logger.critical(f"Fatal Error: Failed to load training data: {e}", exc_info=True)
            if run: run["error/message"] = f"Training data load failed: {e}"; run.stop()
            sys.exit(1)


        # Optimizer, Scheduler, Scaler
        if rank == 0: logger.info(f"Setting up Optimizer, Learning Rate Scheduler, and Gradient Scaler (Rank {rank})...")
        # Filter parameters for weight decay application
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"] # Common parameters to exclude from weight decay
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
        ]
        num_trainable_params = sum(p.numel() for group in optimizer_grouped_parameters for p in group['params'])
        total_params = sum(p.numel() for p in model.parameters())
        if rank == 0: logger.info(f"Model Parameters: Total={total_params:,}, Trainable={num_trainable_params:,}")
        if num_trainable_params == 0:
             logger.critical("No trainable parameters found. Check model configuration and requires_grad settings.")
             sys.exit(1)

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # Calculate total training steps for scheduler
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if num_update_steps_per_epoch == 0:
            logger.warning("Train dataloader seems empty or gradient_accumulation_steps is very large. Max training steps will be 0.")
            max_train_steps = 0
        else:
            max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            if rank == 0: logger.info(f"Estimated total training steps: {max_train_steps:,} ({args.num_train_epochs} epochs * {num_update_steps_per_epoch} steps/epoch)")

        # Adjust warmup steps if necessary
        eff_warmup_steps = min(args.num_warmup_steps, max_train_steps) if max_train_steps > 0 else 0
        if eff_warmup_steps < args.num_warmup_steps and rank == 0:
            logger.warning(f"Requested warmup steps ({args.num_warmup_steps}) exceeds max training steps ({max_train_steps}). Setting warmup steps to {eff_warmup_steps}.")
            args.num_warmup_steps = eff_warmup_steps # Use adjusted value

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=max_train_steps
        )

        # Gradient Scaler for AMP
        scaler_enabled = args.use_amp and device.type == 'cuda'
        if args.use_amp and not scaler_enabled and rank == 0:
             logger.warning("AMP requested (--use_amp) but CUDA is not available. AMP is disabled.")
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        if rank == 0: logger.info(f"Automatic Mixed Precision (AMP) enabled: {scaler.is_enabled()}.")


        # Resume from Checkpoint if specified
        start_epoch, global_step, resumed_from_checkpoint = 0, 0, False
        if args.checkpoint_path:
            state_file = Path(args.checkpoint_path) / "training_state.pt"
            if state_file.is_file():
                if rank == 0: logger.info(f"Attempting to load checkpoint state from: {state_file}")
                try:
                    # Load checkpoint state to the current device
                    checkpoint_state = torch.load(state_file, map_location=device)

                    # Load model state (handle DDP wrapper if necessary)
                    model_to_load = model.module if hasattr(model, 'module') else model
                    missing_keys, unexpected_keys = model_to_load.load_state_dict(checkpoint_state['model'], strict=False)
                    if rank == 0:
                        if missing_keys: logger.warning(f"Missing keys when loading model state: {missing_keys}")
                        if unexpected_keys: logger.warning(f"Unexpected keys when loading model state: {unexpected_keys}")
                        logger.info("Loaded model state from checkpoint.")

                    # Load optimizer and scheduler states
                    if 'optimizer' in checkpoint_state:
                         optimizer.load_state_dict(checkpoint_state['optimizer'])
                         if rank == 0: logger.info("Loaded optimizer state from checkpoint.")
                    else: logger.warning("Optimizer state not found in checkpoint.")
                    if 'lr_scheduler' in checkpoint_state:
                         lr_scheduler.load_state_dict(checkpoint_state['lr_scheduler'])
                         if rank == 0: logger.info("Loaded LR scheduler state from checkpoint.")
                    else: logger.warning("LR scheduler state not found in checkpoint.")

                    # Load AMP scaler state if it was enabled and saved
                    if 'scaler' in checkpoint_state and checkpoint_state['scaler'] and scaler.is_enabled():
                        try:
                            scaler.load_state_dict(checkpoint_state['scaler'])
                            if rank == 0: logger.info("Loaded AMP GradScaler state from checkpoint.")
                        except Exception as scaler_e: logger.warning(f"Could not load GradScaler state: {scaler_e}")
                    elif scaler.is_enabled(): logger.warning("AMP Scaler state not found in checkpoint, starting with default scale.")


                    # Load training progress
                    start_epoch = checkpoint_state.get('epoch', 0) + 1 # Start next epoch
                    global_step = checkpoint_state.get('global_step', 0) # Resume from this step count
                    resumed_from_checkpoint = True
                    if rank == 0: logger.info(f"Resuming training from Epoch {start_epoch}, Global Step {global_step}")

                    # Load RNG states for reproducibility
                    try:
                        if 'torch_rng_state' in checkpoint_state: torch.set_rng_state(checkpoint_state['torch_rng_state'].cpu()) # Load CPU state first
                        if device.type == 'cuda' and 'torch_cuda_rng_state_all' in checkpoint_state and checkpoint_state['torch_cuda_rng_state_all']:
                            # Ensure list of tensors is loaded correctly
                             torch.cuda.set_rng_state_all(checkpoint_state['torch_cuda_rng_state_all'])
                        if 'numpy_rng_state' in checkpoint_state: np.random.set_state(checkpoint_state['numpy_rng_state'])
                        if 'python_rng_state' in checkpoint_state: random.setstate(checkpoint_state['python_rng_state'])
                        if rank == 0: logger.info(f"Restored RNG states from checkpoint.")
                    except Exception as rng_e:
                         logger.warning(f"Could not restore RNG states from checkpoint: {rng_e}")

                    # Clear memory after loading
                    del checkpoint_state
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

                    if rank == 0: logger.info(f"Checkpoint loaded successfully from {args.checkpoint_path}.")
                    if is_distributed: torch.distributed.barrier() # Sync after loading checkpoint

                except Exception as e:
                    logger.error(f"Failed to load checkpoint from {state_file}: {e}", exc_info=True)
                    # Reset state if loading failed completely
                    start_epoch, global_step, resumed_from_checkpoint = 0, 0, False
                    logger.warning("Starting training from scratch due to checkpoint load failure.")
            else:
                 if rank == 0: logger.warning(f"Checkpoint path specified ({args.checkpoint_path}), but training_state.pt not found. Starting training from scratch.")
                 start_epoch, global_step, resumed_from_checkpoint = 0, 0, False
        else:
             if rank == 0: logger.info("No checkpoint specified, starting training from scratch.")


        # Initial Evaluation before training starts (if not resuming)
        if not resumed_from_checkpoint:
            if rank == 0: logger.info("--- Running Initial Evaluation Before Training ---")
            init_std_metrics, init_prime_metrics_summary = {}, {}

            # Run initial standard evaluation
            if not args.skip_standard_eval and eval_dataloader:
                 init_std_metrics = evaluate_standard(args, model, eval_dataloader, device, rank, world_size)
            elif rank == 0: logger.info("Skipping initial standard evaluation.")

            # Run initial priming evaluation
            if args.run_priming_eval:
                 init_prime_metrics_summary = run_priming_evaluation_on_directory(
                     args, model, tokenizer, device, rank, run, global_step=0 # Use step 0 for initial eval
                 )
            elif rank == 0: logger.info("Skipping initial priming evaluation.")

            if rank==0:
                logger.info(f"Initial Evaluation Summary Metrics: Standard={init_std_metrics}, Priming={init_prime_metrics_summary}")
                # Log initial metrics to Neptune if enabled
                if run:
                    try:
                        if init_std_metrics:
                             loss_val = init_std_metrics.get("loss", float('nan')); ppl_val = init_std_metrics.get("perplexity", float('nan'))
                             if math.isfinite(loss_val): run["eval/loss"].append(loss_val, step=0)
                             if math.isfinite(ppl_val): run["eval/perplexity"].append(ppl_val, step=0)
                        if init_prime_metrics_summary:
                            for fname, metrics in init_prime_metrics_summary.items():
                                log_prefix = f"eval/priming/{fname.replace('.', '_').replace('/','_')}"
                                for k, v in metrics.items():
                                    if isinstance(v, (int, float)) and math.isfinite(v):
                                         run[f"{log_prefix}/{k}"].append(v, step=0)
                        logger.info("Logged initial evaluation metrics to Neptune.")
                    except Exception as e: logger.warning(f"Neptune initial eval log failed: {e}")

            # Ensure model is back in training mode after initial eval
            model.train()
            if rank == 0: logger.info("--- Finished Initial Evaluation ---")


        # Training Start Logging (Rank 0 Only)
        if rank == 0:
            logger.info("***** Training Configuration *****")
            logger.info(f"   Total Epochs: {args.num_train_epochs}")
            logger.info(f"   Start Epoch: {start_epoch}")
            logger.info(f"   Start Global Step: {global_step}")
            logger.info(f"   Max Global Steps: {max_train_steps if max_train_steps > 0 else 'Not Limited'}")
            logger.info(f"   Train Dataset Size: {len(train_dataloader.dataset):,}")
            logger.info(f"   Standard Eval Dataset Size: {len(eval_dataloader.dataset) if eval_dataloader else 'N/A'}")
            if args.run_priming_eval: logger.info(f"   Priming Eval Directory: {args.priming_eval_dir_path}")
            eff_batch_size = args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps
            logger.info(f"   Batch Size Per Device: {args.per_device_train_batch_size}")
            logger.info(f"   Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
            logger.info(f"   Number of Devices: {world_size}")
            logger.info(f"   Effective Global Batch Size: {eff_batch_size}")
            logger.info(f"   Logging Steps: {args.logging_steps}")
            logger.info(f"   Save Checkpoint Steps: {args.save_steps}")
            logger.info(f"   Standard Eval Steps: {'Disabled' if args.skip_standard_eval else args.eval_steps}")
            logger.info(f"   Priming Eval Steps: {'Disabled' if not args.run_priming_eval else args.priming_eval_steps}")
            logger.info(f"   Output Directory: {args.output_dir}")
            logger.info(f"   AMP Enabled: {scaler.is_enabled()}")
            logger.info(f"   Learning Rate: {args.learning_rate}, Scheduler: {args.lr_scheduler_type}, Warmup Steps: {args.num_warmup_steps}")
            logger.info(f"   Optimizer: AdamW, Weight Decay: {args.weight_decay}")

        # Training Loop
        if is_distributed: torch.distributed.barrier() # Sync all processes before starting loop
        training_start_time = time.time()
        final_global_step = global_step # Keep track of the step count

        try:
            for epoch in range(start_epoch, args.num_train_epochs):
                if rank == 0: logger.info(f"--- Starting Epoch {epoch + 1}/{args.num_train_epochs} ---")

                # Ensure model is in training mode for the epoch
                model.train()

                # Call train_epoch function, which handles the inner loop, logging, eval, saving
                # It returns the updated global_step count
                final_global_step = train_epoch(
                    args=args, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, scaler=scaler,
                    train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, # Pass eval dataloader for periodic eval
                    train_sampler=train_sampler, epoch=epoch, global_step=final_global_step, device=device, rank=rank, world_size=world_size,
                    run=run, tokenizer=tokenizer,
                )

                # Check if max steps were reached within the epoch function
                if max_train_steps > 0 and final_global_step >= max_train_steps:
                     if rank == 0: logger.info(f"Max training steps ({max_train_steps}) reached. Stopping training.")
                     break # Exit the outer loop (epochs)

            # Training finished (either completed epochs or reached max steps)
            training_duration = time.time() - training_start_time
            if rank == 0: logger.info(f"***** Training Finished *****")
            if rank == 0: logger.info(f"   Total Training Time: {training_duration:.2f} seconds")
            if rank == 0: logger.info(f"   Final Global Step: {final_global_step}")


            # --- Final Saving (Rank 0 Only) ---
            if rank == 0:
                final_model_path = Path(args.output_dir) / "final_model"
                logger.info(f"Saving final model, tokenizer, and config to {final_model_path}")
                try:
                    final_model_path.mkdir(parents=True, exist_ok=True)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(final_model_path)
                    tokenizer.save_pretrained(final_model_path)
                    # Save the final training args used
                    args_dict_final = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
                    with open(final_model_path / "training_args.json", "w", encoding='utf-8') as f:
                         json.dump(args_dict_final, f, indent=4)
                    logger.info(f"Final model assets saved successfully.")

                    # Upload final model to Neptune if enabled
                    if run:
                        try:
                            run["final_model"].upload(str(final_model_path))
                            run["training/duration_seconds"] = training_duration
                            run["training/final_global_step"] = final_global_step
                            logger.info("Uploaded final model directory and training summary to Neptune.")
                        except Exception as e: logger.warning(f"Failed to upload final model to Neptune: {e}")

                except Exception as e:
                    logger.error(f"Failed to save final model assets: {e}", exc_info=True)


            # --- Final Evaluation ---
            if rank == 0: logger.info("--- Running Final Evaluation After Training ---")
            final_std_metrics, final_prime_metrics_summary = {}, {}

            # Final Standard Evaluation
            if not args.skip_standard_eval and eval_dataloader:
                final_std_metrics = evaluate_standard(args, model, eval_dataloader, device, rank, world_size)
            elif rank == 0: logger.info("Skipping final standard evaluation.")

            # Final Priming Evaluation
            if args.run_priming_eval:
                final_prime_metrics_summary = run_priming_evaluation_on_directory(
                    args, model, tokenizer, device, rank, run, global_step=final_global_step # Use final step count
                )
            elif rank == 0: logger.info("Skipping final priming evaluation.")


            # --- Save Final Summary Results (JSON on Rank 0) ---
            if rank == 0:
                 final_results_summary = {}
                 if not args.skip_standard_eval and final_std_metrics:
                      logger.info(f"Final Standard Eval Summary Metrics: {final_std_metrics}")
                      final_results_summary["standard_summary"] = final_std_metrics
                 if args.run_priming_eval and final_prime_metrics_summary:
                      logger.info(f"Final Priming Eval Summary Metrics (All Files): {final_prime_metrics_summary}")
                      final_results_summary["priming_summary"] = final_prime_metrics_summary

                 if final_results_summary:
                      final_res_file = Path(args.output_dir) / "final_evaluation_results.json"
                      try:
                           with open(final_res_file, "w", encoding='utf-8') as f:
                                json.dump(final_results_summary, f, indent=4)
                           logger.info(f"Final evaluation summary results saved to: {final_res_file}")
                           # Log final summary to Neptune
                           if run:
                               try: run["evaluation/final_summary_results"] = final_results_summary
                               except Exception as ne: logger.warning(f"Failed to log final eval summary results to Neptune: {ne}")
                      except IOError as e: logger.error(f"Failed to save final evaluation summary JSON: {e}")
                      except TypeError as e: logger.error(f"Failed to serialize final evaluation summary to JSON: {e}")
                 else:
                      logger.warning("No final evaluation summary metrics were generated to save.")

                 # Remind user about the CSV file
                 if args.run_priming_eval:
                    csv_path = Path(args.output_dir) / 'prime_measures' / 'results.csv'
                    logger.info(f"Raw priming results (if generated) were appended to: {csv_path}")


        except KeyboardInterrupt:
             logger.warning("Training interrupted by user (KeyboardInterrupt).")
             # Optionally save state here if desired
        except Exception as e:
            logger.critical(f"An unexpected error occurred during the training loop (Rank {rank}): {e}", exc_info=True)
            # Log error to Neptune if enabled
            if rank == 0 and run:
                try: run["error/message"] = f"Training loop error: {e}"; run["error/traceback"] = traceback.format_exc()
                except Exception as ne: logger.error(f"Failed to log training loop error to Neptune: {ne}")

        finally:
            # Clean up DDP and Neptune (Rank 0)
            if is_distributed:
                torch.distributed.destroy_process_group()
            if rank == 0 and run:
                 run.stop()
            logger.info(f"Training script finished (Rank {rank}).")


if __name__ == "__main__":
    main()