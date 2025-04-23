# src/train.py (Modified for Frequent Directory Eval)

# === Imports ===
import logging
import argparse
from pathlib import Path

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

    # === Essential Paths ===
    parser.add_argument("--train_dataset_path", type=str, default=None, help="Path to the training Arrow dataset. Required for training.")
    parser.add_argument("--validation_dataset_path", type=str, default=None, help="Path to the validation Arrow dataset. Required for standard eval.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for checkpoints, logs, final model. Required for training.")
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

    args = parser.parse_args()

    # Set defaults
    if args.priming_eval_steps is None: args.priming_eval_steps = args.eval_steps
    if args.priming_per_device_eval_batch_size is None: args.priming_per_device_eval_batch_size = args.per_device_eval_batch_size

    # Validation
    if args.evaluate_only:
        if not args.checkpoint_path: parser.error("--checkpoint_path required for --evaluate_only.")
        if not Path(args.checkpoint_path).is_dir(): parser.error(f"Checkpoint not found: {args.checkpoint_path}")
        if not args.skip_standard_eval and not args.validation_dataset_path: parser.error("--validation_dataset_path required for standard eval.")
        if args.run_priming_eval and not args.priming_eval_dir_path: parser.error("--priming_eval_dir_path required for priming eval.")
    else: # Training
        if not args.train_dataset_path: parser.error("--train_dataset_path required for training.")
        if not args.output_dir: parser.error("--output_dir required for training.")
        if not Path(args.train_dataset_path).is_dir(): parser.error(f"Train dataset not found: {args.train_dataset_path}")
        if not args.skip_standard_eval and not args.validation_dataset_path: parser.error("--validation_dataset_path required for standard eval.")
        if args.run_priming_eval and not args.priming_eval_dir_path: parser.error("--priming_eval_dir_path required for priming eval.")

    if args.validation_dataset_path and not Path(args.validation_dataset_path).is_dir(): parser.error(f"Validation dataset not found: {args.validation_dataset_path}")
    if args.priming_eval_dir_path and not Path(args.priming_eval_dir_path).is_dir(): parser.error(f"Priming dir not found: {args.priming_eval_dir_path}")

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
    else: logger.disabled = True

def set_seed(seed_value):
    """Sets random seeds."""
    import random; import numpy as np; import torch; global logger
    random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)
    try: logger.info(f"Set random seed: {seed_value}")
    except (NameError, AttributeError): print(f"Set random seed: {seed_value} (logger N/A)")

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
            try: logger.info(msg)
            except (NameError, AttributeError): print(f"Info: {msg}")
            torch.distributed.barrier()
        except Exception as e: print(f"ERROR: DDP init failed: {e}"); raise
    else:
        msg = "DDP not enabled."
        try: logger.info(msg)
        except (NameError, AttributeError): print(f"Info: {msg}")
    return is_dist, rank, world_size, local_rank

def load_standard_data(args, is_distributed, rank, world_size, data_collator, mode='train'):
    """Loads standard Arrow datasets."""
    global logger
    from datasets import load_from_disk
    from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
    train_dl, eval_dl, train_sampler = None, None, None

    if mode == 'train':
        if not args.train_dataset_path: logger.error("Missing training path."); raise ValueError("Missing path.")
        logger.info(f"Loading train data: {args.train_dataset_path}")
        try:
            ds = load_from_disk(args.train_dataset_path); logger.info(f"Train size: {len(ds):,}")
            sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if is_distributed else RandomSampler(ds)
            logger.info(f"Using {'DistributedSampler' if is_distributed else 'RandomSampler'} for training.")
            train_dl = DataLoader(ds, sampler=sampler, batch_size=args.per_device_train_batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=data_collator)
            logger.info("Train DataLoader created.")
        except Exception as e: logger.error(f"Train data load fail: {e}"); raise

    if not args.skip_standard_eval:
        if not args.validation_dataset_path: logger.error("Missing validation path."); raise ValueError("Missing path.")
        logger.info(f"Loading validation data: {args.validation_dataset_path}")
        try:
            ds = load_from_disk(args.validation_dataset_path); logger.info(f"Eval size: {len(ds):,}")
            sampler = SequentialSampler(ds) # Use Sequential for eval always
            logger.info("Using SequentialSampler for standard eval.")
            eval_dl = DataLoader(ds, sampler=sampler, batch_size=args.per_device_eval_batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=data_collator)
            logger.info("Standard Eval DataLoader created.")
        except Exception as e: logger.error(f"Validation data load fail: {e}"); raise
    else: logger.info("Skipping standard eval data loading.")

    return train_dl, eval_dl, train_sampler


def save_checkpoint(args, model, optimizer, lr_scheduler, scaler, epoch, global_step, rank, tokenizer=None):
    """Saves training state."""
    if rank != 0: return
    global logger
    import torch; import numpy as np; import random; from pathlib import Path
    ckpt_dir = Path(args.output_dir) / f"checkpoint-{global_step}"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving ckpt {global_step} to {ckpt_dir}")
    unwrapped = model.module if hasattr(model, 'module') else model
    state = {'model': unwrapped.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'scaler': scaler.state_dict() if scaler.is_enabled() else None, 'epoch': epoch, 'global_step': global_step, 'args': vars(args), 'torch_rng_state': torch.get_rng_state(), 'numpy_rng_state': np.random.get_state(), 'python_rng_state': random.getstate(), 'torch_cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    state_file = ckpt_dir / "training_state.pt"
    try: torch.save(state, state_file); logger.info(f"Saved state: {state_file}")
    except Exception as e: logger.error(f"State save fail: {e}")
    try: unwrapped.save_pretrained(ckpt_dir); logger.info(f"Saved model weights to {ckpt_dir}")
    except Exception as e: logger.error(f"Weights save fail: {e}")
    try:
        if tokenizer: tokenizer.save_pretrained(ckpt_dir); logger.info(f"Saved tokenizer to {ckpt_dir}")
        unwrapped.config.save_pretrained(ckpt_dir); logger.info(f"Saved config to {ckpt_dir}")
    except Exception as e: logger.error(f"Tokenizer/Config save fail: {e}")

def load_model_for_evaluation(model_class, checkpoint_path, base_model_name="gpt2"):
    """Loads components from checkpoint."""
    global logger
    from transformers import AutoTokenizer, AutoConfig; from pathlib import Path
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_dir(): raise FileNotFoundError(f"Not found: {ckpt_path}")
    logger.info(f"Loading tokenizer/config from: {ckpt_path}")
    try: tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=True); config = AutoConfig.from_pretrained(ckpt_path); logger.info("Loaded components from checkpoint.")
    except OSError:
        logger.warning(f"Fallback: Loading from base: {base_model_name}")
        try: tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True); config = AutoConfig.from_pretrained(base_model_name)
        except Exception as e: logger.error(f"Fallback failed: {e}"); raise
    logger.info(f"Loading model weights from: {ckpt_path}")
    try: model = model_class.from_pretrained(ckpt_path, config=config)
    except Exception as e: logger.error(f"Weight load fail: {e}"); raise
    if tokenizer.pad_token is None:
        if tokenizer.eos_token: tokenizer.pad_token = tokenizer.eos_token; logger.info(f"Set pad=eos ('{tokenizer.eos_token}')")
        else: added = tokenizer.add_special_tokens({'pad_token': '[PAD]'}); logger.warning(f"Added [PAD] ({added} new).")
        if added > 0: logger.info(f"Resize embed: {model.config.vocab_size}->{len(tokenizer)}"); model.resize_token_embeddings(len(tokenizer)); config.vocab_size = len(tokenizer); model.config.vocab_size=len(tokenizer)
    return model, tokenizer, config


def evaluate_standard(args, model, eval_dataloader, device, rank, world_size):
    """Runs standard eval (perplexity). Handles DDP."""
    if eval_dataloader is None: return {}
    global logger
    import torch; from tqdm.auto import tqdm; import numpy as np; import math
    original_mode = model.training; model.eval()
    total_loss = torch.tensor(0.0, device=device); total_items = torch.tensor(0, device=device)
    is_dist = torch.distributed.is_initialized(); is_rank0 = rank == 0; no_tqdm = not is_rank0
    pbar = tqdm(eval_dataloader, desc="Eval (Std)", leave=False, disable=no_tqdm)
    with torch.no_grad():
        for batch in pbar:
            try: b_dev = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            except RuntimeError as e: logger.error(f"Std eval move err: {e}"); continue
            try:
                with torch.amp.autocast(device_type=device.type, enabled=args.use_amp): out = model(**b_dev); loss = out.loss
            except Exception as e: logger.error(f"Std eval fwd err: {e}"); continue
            if torch.isfinite(loss): num = b_dev['input_ids'].size(0); total_loss += loss.detach() * num; total_items += num
            elif is_rank0: logger.warning(f"Non-finite std eval loss: {loss.item()}")
            if is_rank0: pbar.set_postfix({"loss": f"{loss.item():.4f}" if torch.isfinite(loss) else "NaN"})

    if is_dist: torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM); torch.distributed.all_reduce(total_items, op=torch.distributed.ReduceOp.SUM)
    metrics = {}
    if is_rank0:
        if total_items.item() > 0:
            loss = total_loss.item() / total_items.item()
            try: ppl = math.exp(loss) if loss < 700 else float('inf')
            except (OverflowError, ValueError): ppl = float('inf') if loss > 0 else float('nan')
            logger.info(f"Std Eval: Loss={loss:.4f}, PPL={ppl:.4f}"); metrics = {"loss": loss, "perplexity": ppl}
        else: logger.warning("Std eval: Zero items."); metrics = {"loss": float('nan'), "perplexity": float('nan')}
    if original_mode: model.train()
    else: model.eval()
    if not no_tqdm: pbar.close()
    if is_dist: torch.distributed.barrier()
    return metrics


# --- Helper Function for Multi-CSV Priming Eval (No Changes Needed Here) ---
def run_priming_evaluation_on_directory(args, model, tokenizer, device, rank, run, global_step):
    """Finds CSVs, creates dataloaders, runs eval, and aggregates results."""
    if not args.run_priming_eval or not args.priming_eval_dir_path: return {}

    global logger
    from pathlib import Path
    # Import priming libs here to keep them contained
    try:
        from priming_evaluation.data_loader import create_priming_dataloader
        from priming_evaluation.evaluator import run_native_priming_eval
    except ImportError as e:
        logger.error(f"Failed to import priming evaluation library: {e}. Cannot run priming eval.")
        return {"error": "Priming library import failed."}

    import math # For isfinite check
    import torch # For distributed check

    priming_dir = Path(args.priming_eval_dir_path)
    if not priming_dir.is_dir(): logger.error(f"Priming dir not found: {priming_dir}"); return {}

    csv_files = sorted(list(priming_dir.glob('*.csv')))
    if not csv_files: logger.warning(f"No CSVs found in: {priming_dir}"); return {}

    all_priming_results = {}
    logger.info(f"Found {len(csv_files)} CSVs for priming eval in {priming_dir}.")
    is_distributed = torch.distributed.is_initialized()
    original_mode = model.training # Store original mode

    for csv_path in csv_files:
        csv_filename = csv_path.name
        logger.info(f"--- Running Priming Eval for: {csv_filename} (Step {global_step}) ---")
        priming_dataloader_single = None
        if is_distributed: torch.distributed.barrier() # Sync before creating loader/running eval for each file
        try:
            priming_dataloader_single = create_priming_dataloader(
                csv_path=str(csv_path), tokenizer=tokenizer,
                batch_size=args.priming_per_device_eval_batch_size,
                delimiter=args.priming_delimiter, num_workers=args.num_workers, pin_memory=True)
        except Exception as e: logger.error(f"Dataloader fail for {csv_filename}: {e}", exc_info=True); all_priming_results[csv_filename] = {"error": str(e)}; continue
        if priming_dataloader_single is None: logger.warning(f"Dataloader None for {csv_filename}. Skipping."); all_priming_results[csv_filename] = {"error": "Dataloader None."}; continue

        try:
            # run_native_priming_eval sets model.eval() internally
            priming_metrics = run_native_priming_eval(model=model, priming_dataloader=priming_dataloader_single, device=device, tokenizer=tokenizer)
            all_priming_results[csv_filename] = priming_metrics
            logger.info(f"Priming Metrics for {csv_filename}: {priming_metrics}")

            if rank == 0 and run: # Neptune logging
                 log_prefix = f"eval/priming/{csv_filename.replace('.', '_')}"
                 try:
                     for k, v in priming_metrics.items():
                         if isinstance(v, (int, float)) and math.isfinite(v): run[f"{log_prefix}/{k}"].append(v, step=global_step)
                     logger.info(f"Logged priming metrics for {csv_filename} to Neptune @ step {global_step}.")
                 except Exception as e: logger.warning(f"Neptune log fail for {csv_filename}: {e}")

        except Exception as e: logger.error(f"Priming eval error for {csv_filename}: {e}", exc_info=True); all_priming_results[csv_filename] = {"error": str(e)}
        finally: del priming_dataloader_single # Cleanup

    if is_distributed: torch.distributed.barrier() # Sync after all files done
    # Restore original model mode (evaluator sets it to eval)
    if original_mode: model.train()
    else: model.eval()

    logger.info("--- Finished All Priming Evaluations for this step ---")
    return all_priming_results


# --- Modified train_epoch ---
def train_epoch(args, model, optimizer, lr_scheduler, scaler, train_dataloader, eval_dataloader, #<<< Removed priming_dataloader arg
                train_sampler, epoch, global_step, device, rank, world_size, run, tokenizer):
    """Runs one training epoch with periodic multi-CSV priming eval."""
    global logger
    import torch; from torch.utils.data import DistributedSampler; from tqdm.auto import tqdm; import math; import sys
    from torch.nn.utils import clip_grad_norm_
    # Priming evaluation helper is called directly below, no need for separate imports here

    model.train()
    is_distributed = train_sampler is not None and isinstance(train_sampler, DistributedSampler)
    if is_distributed: train_sampler.set_epoch(epoch)
    disable_tqdm = rank != 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=True, disable=disable_tqdm, position=0)
    total_loss_since_logging, steps_since_logging = 0.0, 0

    for step, batch in enumerate(progress_bar):
        try: batch_on_device = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        except RuntimeError as e: logger.error(f"Train move error step {global_step}: {e}"); continue
        try:
            with torch.amp.autocast(device_type=device.type, enabled=args.use_amp): outputs = model(**batch_on_device); loss = outputs.loss
        except Exception as e: logger.error(f"Train forward error step {global_step}: {e}"); optimizer.zero_grad(set_to_none=True); continue
        if not torch.isfinite(loss): logger.warning(f"Non-finite loss {loss.item()} step {global_step}. Skipping."); optimizer.zero_grad(set_to_none=True); continue

        scaled_loss = loss / args.gradient_accumulation_steps
        current_loss_value = loss.item()
        scaler.scale(scaled_loss).backward()
        total_loss_since_logging += current_loss_value; steps_since_logging += 1

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.max_grad_norm and args.max_grad_norm > 0:
                if scaler.is_enabled(): scaler.unscale_(optimizer)
                params = model.module.parameters() if is_distributed else model.parameters()
                clip_grad_norm_(params, args.max_grad_norm)
            scaler.step(optimizer); scaler.update(); lr_scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step += 1

            if rank == 0 and global_step % args.logging_steps == 0:
                 avg_loss = total_loss_since_logging / steps_since_logging if steps_since_logging > 0 else 0.0
                 lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, 'get_last_lr') and lr_scheduler.get_last_lr() else optimizer.param_groups[0]['lr']
                 logger.info(f"E{epoch+1} S{global_step}: Avg Loss={avg_loss:.4f}, LR={lr:.6e}")
                 if 'neptune' in sys.modules and sys.modules['neptune'] and run:
                     try:
                         if math.isfinite(avg_loss): run["train/step_loss"].append(avg_loss, step=global_step)
                         if math.isfinite(lr): run["train/learning_rate"].append(lr, step=global_step)
                         if torch.cuda.is_available(): run["train/gpu_mem_alloc_gb"].append(torch.cuda.memory_allocated(device)/1e9, step=global_step)
                         if scaler.is_enabled(): run["train/grad_scale"].append(scaler.get_scale(), step=global_step)
                     except Exception as e: logger.warning(f"Neptune train log fail step {global_step}: {e}")
                 total_loss_since_logging, steps_since_logging = 0.0, 0

            # --- Evaluation Step ---
            run_std_eval_now = not args.skip_standard_eval and eval_dataloader and global_step > 0 and global_step % args.eval_steps == 0
            # Check run_priming_eval flag and step count
            run_prime_eval_now = args.run_priming_eval and global_step > 0 and global_step % args.priming_eval_steps == 0

            if run_std_eval_now or run_prime_eval_now:
                logger.info(f"--- Starting Periodic Eval @ Step {global_step} ---")
                if is_distributed: torch.distributed.barrier() # Sync before eval starts
                original_mode = model.training # Remember current mode

                std_metrics = {}; priming_metrics_all = {} # Hold results

                # --- Standard Evaluation ---
                if run_std_eval_now:
                    logger.info("--- Running Periodic Standard Eval ---")
                    std_metrics = evaluate_standard(args, model, eval_dataloader, device, rank, world_size)
                    # Log standard metrics to Neptune (rank 0 only)
                    if rank == 0 and run and std_metrics:
                        try:
                            loss_val = std_metrics.get("loss", float('nan')); ppl_val = std_metrics.get("perplexity", float('nan'))
                            if math.isfinite(loss_val): run["eval/loss"].append(loss_val, step=global_step)
                            if math.isfinite(ppl_val): run["eval/perplexity"].append(ppl_val, step=global_step)
                            logger.info(f"Logged periodic standard eval metrics to Neptune @ step {global_step}.")
                        except Exception as e: logger.warning(f"Neptune periodic std eval log fail step {global_step}: {e}")
                elif global_step % args.eval_steps == 0: # Log skip only if it *would* have run
                    logger.info("--- Skipping Periodic Standard Eval ---")

                # --- Priming Evaluation (calls helper function) ---
                if run_prime_eval_now:
                    logger.info("--- Running Periodic Priming Eval on Directory ---")
                    # The helper function handles DDP syncs internally between files and restores model state
                    priming_metrics_all = run_priming_evaluation_on_directory(
                        args, model, tokenizer, device, rank, run, global_step
                    )
                    # Logging to console/Neptune happens *inside* the helper now
                elif global_step % args.priming_eval_steps == 0: # Log skip only if it *would* have run
                     logger.info("--- Skipping Periodic Priming Eval ---")

                # --- Post-Evaluation ---
                if is_distributed: torch.distributed.barrier() # Sync after all eval is done
                # Ensure model is back in training mode if it was originally
                if original_mode: model.train()
                logger.info(f"--- Finished Periodic Eval, Resuming Train ---")

            # --- Save Checkpoint ---
            if global_step > 0 and global_step % args.save_steps == 0:
                if is_distributed: torch.distributed.barrier()
                save_checkpoint(args, model, optimizer, lr_scheduler, scaler, epoch, global_step, rank, tokenizer)
                if is_distributed: torch.distributed.barrier()

        if rank == 0: progress_bar.set_postfix({"loss": f"{current_loss_value:.4f}", "step": global_step})

    if rank == 0: progress_bar.close(); logger.info(f"--- Epoch {epoch+1} Finished ---")
    return global_step


# --- Main Execution ---
def main():
    """Main function."""
    print("Importing standard libraries...")
    # ... (standard imports: math, torch, random, np, logging, json, sys, time, traceback, gc, tqdm, Path) ...
    import math; import torch; import random; import numpy as np; import logging
    import json; import sys; import time; import traceback; import gc
    from tqdm.auto import tqdm; from pathlib import Path
    print("Finished standard imports.")
    print("Importing ML/data libraries...")
    try:
        # ... (ML imports: DataLoader, Samplers, DDP, AdamW, HF models/tokenizers, GradScaler, amp, neptune, datasets) ...
        from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
        from torch.nn.utils import clip_grad_norm_; from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.optim import AdamW; from transformers import GPT2LMHeadModel, AutoConfig, get_scheduler, AutoTokenizer, DataCollatorForLanguageModeling
        from torch.cuda.amp import GradScaler; import torch.amp # Use torch.amp namespace
        try: import neptune
        except ImportError: print("Neptune not found, logging disabled."); sys.modules['neptune'] = None
        from datasets import load_from_disk
        # Priming imports NOT needed here, handled in helper function run_priming_evaluation_on_directory
        print("Finished ML/data imports.")
    except ImportError as e: print(f"ERROR: Import failed: {e}"); sys.exit(1)

    args = parse_args()
    is_distributed, rank, world_size, local_rank = setup_distributed(args)
    setup_logging(rank=rank)
    global logger; logger = logging.getLogger(__name__) # Assign logger

    if rank == 0: logger.info(f"Mode: {'EVAL ONLY' if args.evaluate_only else 'TRAINING'}")
    if rank == 0: logger.info(f"Args: {vars(args)}")

    device = get_device()
    if rank == 0: logger.info(f"Device: {device} {'('+torch.cuda.get_device_name(device)+')' if device.type == 'cuda' else ''}")
    set_seed(args.seed + rank)

    run = None # Neptune
    if rank == 0 and sys.modules['neptune'] and args.neptune_project:
        try:
            run = neptune.init_run(project=args.neptune_project, name=args.neptune_run_name, tags=args.neptune_tags)
            args_log = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            run["parameters"] = args_log; logger.info(f"Neptune Run: {run.get_url()}")
        except Exception as e: logger.error(f"Neptune init failed: {e}"); run = None
    elif rank == 0: logger.info("Neptune disabled.")

    # === Model & Tokenizer Loading ===
    try:
        if args.evaluate_only: model, tokenizer, config = load_model_for_evaluation(GPT2LMHeadModel, args.checkpoint_path, args.model)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
            if tokenizer.pad_token is None:
                if tokenizer.eos_token: tokenizer.pad_token = tokenizer.eos_token; logger.info(f"Set pad=eos")
                else: added = tokenizer.add_special_tokens({'pad_token': '[PAD]'}); logger.warning(f"Added [PAD] ({added} new).")
            config = AutoConfig.from_pretrained(args.model); config.use_cache = False
            model = GPT2LMHeadModel.from_pretrained(args.model, config=config)
            if len(tokenizer) > model.config.vocab_size: model.resize_token_embeddings(len(tokenizer)); config.vocab_size = len(tokenizer); model.config.vocab_size=len(tokenizer); logger.info("Resized embeddings.")
        model.to(device); logger.info(f"Model loaded (Rank {rank})")
        if is_distributed: model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False); logger.info(f"DDP Wrapped (Rank {rank}).")
    except Exception as e: logger.critical(f"Model/Tokenizer load failed: {e}", exc_info=True); sys.exit(1)

    # === Standard Eval Data Loading ===
    eval_dataloader = None; data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    if not args.skip_standard_eval:
        try: _, eval_dataloader, _ = load_standard_data(args, is_distributed, rank, world_size, data_collator, mode='eval')
        except Exception as e: logger.error(f"Std eval data setup fail: {e}. Skipping.", exc_info=True); args.skip_standard_eval = True
    else: logger.info("Skipping standard validation loading.")

    # === Evaluation Only Mode ===
    if args.evaluate_only:
        if rank == 0: logger.info(f"***** EVALUATION ONLY *****"); logger.info(f"Checkpoint: {args.checkpoint_path}")
        std_metrics = {}; prime_metrics_all = {}
        if not args.skip_standard_eval:
            if eval_dataloader: logger.info("--- Running Standard Eval ---"); std_metrics = evaluate_standard(args, model, eval_dataloader, device, rank, world_size)
            else: logger.warning("Std eval skipped (no dataloader).")
        else: logger.info("--- Skipping Standard Eval ---")
        if args.run_priming_eval:
            logger.info("--- Running Priming Eval on Directory ---")
            prime_metrics_all = run_priming_evaluation_on_directory(args, model, tokenizer, device, rank, run, global_step=0)
        else: logger.info("--- Skipping Priming Eval ---")
        if rank == 0:
            logger.info("***** Evaluation Complete *****"); results = {}
            if not args.skip_standard_eval: logger.info(f"Std Metrics: {std_metrics}"); results["standard"] = std_metrics
            if args.run_priming_eval: logger.info(f"Priming Metrics (All): {prime_metrics_all}"); results["priming"] = prime_metrics_all
            if results:
                res_file = Path(args.checkpoint_path) / "evaluation_results.json"
                try:
                    with open(res_file, "w") as f: json.dump(results, f, indent=4); logger.info(f"Results saved: {res_file}")
                    if run: run["evaluation/final_results"] = results; logger.info("Logged final eval results to Neptune.")
                except Exception as e: logger.error(f"Result save fail: {e}")
            else: logger.warning("No eval metrics generated.")
        if is_distributed: torch.distributed.destroy_process_group()
        if run: run.stop()
        logger.info("Evaluation finished."); sys.exit(0)

    # === Training Mode ===
    else:
        # Load Training Data
        try: train_dataloader, _, train_sampler = load_standard_data(args, is_distributed, rank, world_size, data_collator, mode='train')
        except Exception as e: logger.critical(f"Training data failed: {e}", exc_info=True); sys.exit(1)

        # Optimizer, Scheduler, Scaler
        logger.info(f"Setting up Optimizer/Scheduler/Scaler (Rank {rank})...")
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        opt_params = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": args.weight_decay}, {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0}]
        num_trainable = sum(p.numel() for g in opt_params for p in g['params']); total_params = sum(p.numel() for p in model.parameters())
        if num_trainable == 0: logger.critical("No trainable params."); sys.exit(1)
        logger.info(f"Params: Total={total_params:,}, Trainable={num_trainable:,}")
        optimizer = AdamW(opt_params, lr=args.learning_rate)
        num_batches_epoch = len(train_dataloader); max_train_steps = 0
        if num_batches_epoch > 0: max_train_steps = args.num_train_epochs * math.ceil(num_batches_epoch / args.gradient_accumulation_steps)
        else: logger.warning("Train DL len=0.")
        eff_warmup = min(args.num_warmup_steps, max_train_steps) if max_train_steps > 0 else 0
        if eff_warmup < args.num_warmup_steps: logger.warning(f"Warmup adjusted: {args.num_warmup_steps}->{eff_warmup}"); args.num_warmup_steps = eff_warmup
        lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=max_train_steps)
        logger.info(f"Est. steps: {max_train_steps:,}")
        scaler_enabled = args.use_amp and device.type == 'cuda'
        if args.use_amp and not scaler_enabled: logger.warning("AMP disabled (not CUDA).")
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled); logger.info(f"AMP Scaler (On: {scaler_enabled}).")

        # Resume from Checkpoint
        start_epoch, global_step, resumed = 0, 0, False
        if args.checkpoint_path:
            state_file = Path(args.checkpoint_path) / "training_state.pt"
            if state_file.is_file():
                logger.info(f"Loading checkpoint state: {state_file}")
                try:
                    ckpt = torch.load(state_file, map_location=device)
                    m_load = model.module if hasattr(model, 'module') else model; miss, unex = m_load.load_state_dict(ckpt['model'], strict=False)
                    if miss or unex: logger.warning(f"Model load state: Missing={miss}, Unexpected={unex}")
                    if 'optimizer' in ckpt: optimizer.load_state_dict(ckpt['optimizer']); logger.info("Loaded optimizer state.")
                    if 'lr_scheduler' in ckpt: lr_scheduler.load_state_dict(ckpt['lr_scheduler']); logger.info("Loaded scheduler state.")
                    if 'scaler' in ckpt and ckpt['scaler'] and scaler.is_enabled(): scaler.load_state_dict(ckpt['scaler']); logger.info("Loaded scaler state.")
                    start_epoch = ckpt.get('epoch', -1) + 1; global_step = ckpt.get('global_step', 0); logger.info(f"Resuming: Epoch {start_epoch}, Step {global_step}")
                    try: # RNG
                        if 'torch_rng_state' in ckpt: torch.set_rng_state(ckpt['torch_rng_state'].cpu())
                        if device.type == 'cuda' and 'torch_cuda_rng_state_all' in ckpt and ckpt['torch_cuda_rng_state_all']: torch.cuda.set_rng_state_all(ckpt['torch_cuda_rng_state_all'])
                        if 'numpy_rng_state' in ckpt: np.random.set_state(ckpt['numpy_rng_state'])
                        if 'python_rng_state' in ckpt: random.setstate(ckpt['python_rng_state'])
                        logger.info(f"Restored RNG states.")
                    except Exception as rng_e: logger.warning(f"RNG restore error: {rng_e}")
                    resumed = True; logger.info(f"Checkpoint loaded."); gc.collect(); torch.cuda.empty_cache()
                    if is_distributed: torch.distributed.barrier()
                except Exception as e: logger.error(f"Ckpt load failed: {e}", exc_info=True); start_epoch, global_step = 0, 0
            else: logger.warning(f"Ckpt specified but not found: {state_file}. Starting fresh."); start_epoch, global_step = 0, 0
        else: logger.info("No checkpoint, starting fresh.")

        # Initial Evaluation
        if not resumed:
            logger.info("--- Initial Evaluation ---")
            init_std_metrics, init_prime_metrics = {}, {}
            if not args.skip_standard_eval and eval_dataloader: init_std_metrics = evaluate_standard(args, model, eval_dataloader, device, rank, world_size)
            if args.run_priming_eval: init_prime_metrics = run_priming_evaluation_on_directory(args, model, tokenizer, device, rank, run, global_step=0)
            if rank==0: logger.info(f"Initial Metrics: Standard={init_std_metrics}, Priming={init_prime_metrics}")
            model.train(); logger.info("--- Finished Initial Eval ---")

        # Training Start Logging
        if rank == 0:
            logger.info("***** Starting Training *****")
            # ... (log config details) ...
            logger.info(f" Config: Epochs={args.num_train_epochs}, StartEpoch={start_epoch}, StartStep={global_step}, MaxSteps={max_train_steps}")
            logger.info(f" Data: Train={len(train_dataloader.dataset):,}, StdEval={len(eval_dataloader.dataset) if eval_dataloader else 'N/A'}")
            if args.run_priming_eval: logger.info(f" Data: PrimingDir={args.priming_eval_dir_path}")
            eff_bs = args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps
            logger.info(f" Batch: Device={args.per_device_train_batch_size}, Accum={args.gradient_accumulation_steps}, Devices={world_size}, Effective={eff_bs}")
            logger.info(f" Steps: Log={args.logging_steps}, Save={args.save_steps}, StdEval={args.eval_steps if not args.skip_standard_eval else 'Off'}, PrimeEval={args.priming_eval_steps if args.run_priming_eval else 'Off'}")
            logger.info(f" Output: {args.output_dir}")
            try: Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            except OSError as e: logger.critical(f"Output dir error {args.output_dir}: {e}"); sys.exit(1)

        # Training Loop
        if is_distributed: torch.distributed.barrier() # Sync before loop
        start_time = time.time()
        try:
            for epoch in range(start_epoch, args.num_train_epochs):
                if rank == 0: logger.info(f"--- Starting Epoch {epoch + 1}/{args.num_train_epochs} ---")
                # Call train_epoch (which now handles periodic multi-CSV priming eval inside)
                global_step = train_epoch(
                    args=args, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, scaler=scaler,
                    train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, #<<< No priming_dataloader here
                    train_sampler=train_sampler, epoch=epoch, global_step=global_step, device=device, rank=rank, world_size=world_size,
                    run=run, tokenizer=tokenizer,
                )
                if max_train_steps > 0 and global_step >= max_train_steps: logger.info(f"Max steps {max_train_steps} reached."); break
            duration = time.time() - start_time
            if rank == 0: logger.info(f"Training finished. Time: {duration:.2f}s")

            # Final Saving
            if rank == 0:
                final_path = Path(args.output_dir) / "final_model"; logger.info(f"Saving final model to {final_path}")
                try:
                    final_path.mkdir(parents=True, exist_ok=True); m_save = model.module if hasattr(model, 'module') else model
                    m_save.save_pretrained(final_path); tokenizer.save_pretrained(final_path)
                    args_d = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
                    with open(final_path / "training_args.json", "w") as f: json.dump(args_d, f, indent=4)
                    logger.info(f"Final model saved.")
                    if run: run["final_model"].upload(str(final_path)); run["training/duration_seconds"]=duration; run["training/final_global_step"]=global_step; logger.info("Uploaded final model to Neptune.")
                except Exception as e: logger.error(f"Final save fail: {e}", exc_info=True)

            # Final Evaluation
            logger.info("--- Running Final Evaluation ---")
            final_std_metrics, final_prime_metrics = {}, {}
            if not args.skip_standard_eval and eval_dataloader: final_std_metrics = evaluate_standard(args, model, eval_dataloader, device, rank, world_size)
            if args.run_priming_eval: final_prime_metrics = run_priming_evaluation_on_directory(args, model, tokenizer, device, rank, run, global_step=global_step)

            # Save Final Results
            if rank == 0:
                 final_results = {}
                 if not args.skip_standard_eval: logger.info(f"Final Std Metrics: {final_std_metrics}"); final_results["standard"] = final_std_metrics
                 if args.run_priming_eval: logger.info(f"Final Priming Metrics: {final_prime_metrics}"); final_results["priming"] = final_prime_metrics
                 if final_results:
                      res_file = Path(args.output_dir) / "final_evaluation_results.json"
                      try:
                           with open(res_file, "w") as f: json.dump(final_results, f, indent=4); logger.info(f"Final results saved: {res_file}")
                           if run: run["evaluation/final_results"] = final_results; logger.info("Logged final eval results to Neptune.")
                      except Exception as e: logger.error(f"Final result save fail: {e}")
                 else: logger.warning("No final eval metrics.")

        except Exception as e:
            logger.critical(f"Training loop error (Rank {rank}): {e}", exc_info=True)
            if rank == 0 and run: run["error/message"] = str(e); run["error/traceback"] = traceback.format_exc()

        finally:
            if rank == 0 and run: run.stop()
            if is_distributed: torch.distributed.destroy_process_group()
            logger.info(f"Script finished (Rank {rank}).")

if __name__ == "__main__":
    main()