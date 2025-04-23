# src/priming_evaluation/evaluator.py (Revised SEM Calculation)

import logging
import math # Keep math for isfinite checks
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import numpy as np # Import numpy for sqrt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Define the structure for individual item results
ResultItem = Dict[str, float] # {'pe': float, 'logp_con': float, 'logp_incon': float}
# Define the return type for the batch calculation function
BatchResults = Dict[str, List[ResultItem]]

# --- calculate_priming_effect (No changes needed here) ---
# (Keep the function as it was in the previous version with AMP context manager if added)
def calculate_priming_effect(
    model: PreTrainedModel,
    batch: Dict[str, Any],
    device: torch.device,
    use_amp: bool = False # Optionally pass AMP flag if needed
) -> BatchResults:
    """
    Calculates the Priming Effect (PE) and intermediate log probabilities
    for a batch of data.
    PE = log P(Target | CongruentPrime) - log P(Target | IncongruentPrime)
    Uses corrected label indexing for log probability calculation.

    Returns:
        Dict[str, List[ResultItem]]: A dictionary where keys are target
        structures and values are lists of dictionaries. Each inner
        dictionary contains 'pe', 'logp_con', and 'logp_incon' for one item.
    """
    # Move tensors to device
    try:
        congruent_input_ids = batch['congruent_input_ids'].to(device)
        congruent_attention_mask = batch['congruent_attention_mask'].to(device)
        incongruent_input_ids = batch['incongruent_input_ids'].to(device)
        incongruent_attention_mask = batch['incongruent_attention_mask'].to(device)
        labels = batch['labels'].to(device) # Padded target token IDs, -100 elsewhere
        target_starts_con = batch['target_start_congruent'] # Tensor on CPU/Device
        target_ends_con = batch['target_end_congruent']     # Tensor on CPU/Device
        target_starts_incon = batch['target_start_incongruent'] # Tensor on CPU/Device
        target_ends_incon = batch['target_end_incongruent']   # Tensor on CPU/Device
        target_structures = batch['target_structure'] # List of strings
        # Convert indices to tensors on device if they aren't already (usually done by collate)
        if isinstance(target_starts_con, list): target_starts_con = torch.tensor(target_starts_con, dtype=torch.long, device=device)
        if isinstance(target_ends_con, list): target_ends_con = torch.tensor(target_ends_con, dtype=torch.long, device=device)
        if isinstance(target_starts_incon, list): target_starts_incon = torch.tensor(target_starts_incon, dtype=torch.long, device=device)
        if isinstance(target_ends_incon, list): target_ends_incon = torch.tensor(target_ends_incon, dtype=torch.long, device=device)

    except KeyError as e:
        logger.error(f"Batch missing key: {e}. Cannot calculate PE metrics.")
        return {}
    except Exception as e:
        logger.error(f"Error moving/preparing batch for device {device}: {e}")
        return {}

    batch_size = congruent_input_ids.size(0)
    batch_results: BatchResults = defaultdict(list)
    nan_result: ResultItem = {'pe': float('nan'), 'logp_con': float('nan'), 'logp_incon': float('nan')}

    # Perform forward passes WITH AUTOCAST
    try:
        amp_enabled = use_amp and device.type == 'cuda' # Enable AMP only if requested AND on CUDA
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs_con = model(input_ids=congruent_input_ids, attention_mask=congruent_attention_mask)
                logits_con = outputs_con.logits # (batch, seq_len_con, vocab)
                outputs_incon = model(input_ids=incongruent_input_ids, attention_mask=incongruent_attention_mask)
                logits_incon = outputs_incon.logits # (batch, seq_len_incon, vocab)
        # Optionally cast logits if needed, usually fine
        # logits_con = logits_con.float()
        # logits_incon = logits_incon.float()
    except Exception as e:
        logger.error(f"Model forward pass error (AMP enabled: {amp_enabled}): {e}", exc_info=True)
        return {}

    # Calculate log probabilities and PE for each item in the batch
    for i in range(batch_size):
        target_structure = target_structures[i]
        log_prob_con_val = float('nan')
        log_prob_incon_val = float('nan')
        priming_effect = float('nan')

        try:
            start_con = target_starts_con[i].item()
            end_con = target_ends_con[i].item()
            start_incon = target_starts_incon[i].item()
            end_incon = target_ends_incon[i].item()

            logits_for_target_con = logits_con[i, start_con-1 : end_con-1, :]
            logits_for_target_incon = logits_incon[i, start_incon-1 : end_incon-1, :]
            target_labels = labels[i, start_con : end_con]

            if logits_for_target_con.shape[0] != target_labels.shape[0] or \
               logits_for_target_incon.shape[0] != target_labels.shape[0] or \
               target_labels.shape[0] == 0: # Also check for zero length target
                # If target length is zero, log probabilities are undefined/zero
                if target_labels.shape[0] == 0:
                    logger.warning(f"Zero length target sequence for item {i}, target {target_structure}. Skipping.")
                else:
                    logger.warning(f"Logit/Label length mismatch for item {i}, target {target_structure}. "
                                f"LogitCon:{logits_for_target_con.shape[0]}, LogitIncon:{logits_for_target_incon.shape[0]}, Label:{target_labels.shape[0]}. Skipping.")
                batch_results[target_structure].append(nan_result)
                continue

            vocab_size = logits_for_target_con.size(-1)
            # Use .view instead of reshape for potential memory efficiency
            log_prob_con_tensor = -F.cross_entropy(
                logits_for_target_con.view(-1, vocab_size),
                target_labels.view(-1),
                ignore_index=-100,
                reduction='sum'
            )
            log_prob_incon_tensor = -F.cross_entropy(
                logits_for_target_incon.view(-1, vocab_size),
                target_labels.view(-1),
                ignore_index=-100,
                reduction='sum'
            )

            log_prob_con_val = log_prob_con_tensor.item()
            log_prob_incon_val = log_prob_incon_tensor.item()

            if math.isfinite(log_prob_con_val) and math.isfinite(log_prob_incon_val):
                priming_effect = log_prob_con_val - log_prob_incon_val
            else:
                priming_effect = float('nan')

            current_result: ResultItem = {
                'pe': priming_effect,
                'logp_con': log_prob_con_val,
                'logp_incon': log_prob_incon_val
            }

            if not math.isfinite(priming_effect):
                 logger.warning(f"Non-finite PE/LogP: item {i}, target {target_structure}. LogP_con={log_prob_con_val}, LogP_incon={log_prob_incon_val}")
                 batch_results[target_structure].append(nan_result)
            else:
                 batch_results[target_structure].append(current_result)

        except IndexError as e:
            logger.error(f"IndexError calculating metrics for item {i}, target {target_structure}. Indices:{start_con=},{end_con=},{start_incon=},{end_incon=}. Err:{e}")
            batch_results[target_structure].append(nan_result)
        except Exception as e:
            logger.error(f"Unexpected error calculating metrics for item {i}, target {target_structure}: {e}", exc_info=True)
            batch_results[target_structure].append(nan_result)

    return dict(batch_results)


# --- run_native_priming_eval (MODIFIED FOR SEM) ---
def run_native_priming_eval(
    model: PreTrainedModel, priming_dataloader: DataLoader, device: torch.device,
    tokenizer: PreTrainedTokenizer, # Keep tokenizer for potential debugging
    use_amp: bool = False # Pass AMP flag from args if available
) -> Dict[str, float]:
    """
    Runs the full native priming evaluation loop, calculating Priming Effect (PE),
    congruent log probability (LogP_con), and incongruent log probability (LogP_incon).
    Returns average, standard deviation, and standard error (SEM) for PE,
    and avg/std for LogP metrics per target structure.
    """
    logger.info("Starting native priming evaluation (PE, LogP_con, LogP_incon Metrics)...")
    original_mode = model.training
    model.eval()

    all_results: Dict[str, List[ResultItem]] = defaultdict(list)

    progress_bar = tqdm(priming_dataloader, desc="Priming Eval", leave=False)
    for batch in progress_bar:
        if not batch:
            logger.warning("Skipping empty batch from collate function.")
            continue
        try:
            # Pass use_amp flag to the calculation function
            batch_metrics: BatchResults = calculate_priming_effect(model, batch, device, use_amp=use_amp)
            for target_structure, result_list in batch_metrics.items():
                all_results[target_structure].extend(result_list)
        except Exception as e:
            logger.error(f"Error processing priming batch: {e}", exc_info=True)

    # --- Calculate Final Aggregate Metrics ---
    final_metrics: Dict[str, float] = {}
    logger.info("Calculating final priming metric averages:")

    for target_structure, structure_results_list in all_results.items():
        # Extract finite values for each metric separately
        finite_pe_values = [r['pe'] for r in structure_results_list if isinstance(r, dict) and 'pe' in r and math.isfinite(r['pe'])]
        finite_logp_con_values = [r['logp_con'] for r in structure_results_list if isinstance(r, dict) and 'logp_con' in r and math.isfinite(r['logp_con'])]
        finite_logp_incon_values = [r['logp_incon'] for r in structure_results_list if isinstance(r, dict) and 'logp_incon' in r and math.isfinite(r['logp_incon'])]

        total_items = len(structure_results_list)
        logger.info(f"  Target '{target_structure}' (Total Items Processed: {total_items}):") # Clarified log

        # --- Calculate and store metrics for PE (Avg, Std, SEM) ---
        if finite_pe_values:
            count_pe = len(finite_pe_values)
            mean_pe = np.mean(finite_pe_values)
            std_pe = np.std(finite_pe_values)
            # --- Calculate SEM ---
            sem_pe = float('nan') # Default to NaN
            if count_pe > 0:
                # Use np.sqrt since numpy is imported
                sem_pe = std_pe / np.sqrt(count_pe)
            # --- End SEM Calculation ---

            final_metrics[f"avg_PE_{target_structure}"] = mean_pe
            final_metrics[f"std_PE_{target_structure}"] = std_pe
            final_metrics[f"sem_PE_{target_structure}"] = sem_pe # Store SEM
            final_metrics[f"count_PE_{target_structure}"] = count_pe
            # --- Update Logging ---
            logger.info(f"    PE       : Avg = {mean_pe:.4f} (Std = {std_pe:.4f}, SEM = {sem_pe:.4f}, N = {count_pe})")
        else:
            final_metrics[f"avg_PE_{target_structure}"] = float('nan')
            final_metrics[f"std_PE_{target_structure}"] = float('nan')
            final_metrics[f"sem_PE_{target_structure}"] = float('nan') # Store NaN SEM
            final_metrics[f"count_PE_{target_structure}"] = 0
            logger.warning(f"    PE       : No finite values found.")

        # --- Calculate and store metrics for LogP Congruent (Avg, Std) ---
        if finite_logp_con_values:
            count_logp_con = len(finite_logp_con_values)
            mean_logp_con = np.mean(finite_logp_con_values)
            std_logp_con = np.std(finite_logp_con_values)
            final_metrics[f"avg_LogP_con_{target_structure}"] = mean_logp_con
            final_metrics[f"std_LogP_con_{target_structure}"] = std_logp_con
            final_metrics[f"count_LogP_con_{target_structure}"] = count_logp_con
            logger.info(f"    LogP_con : Avg = {mean_logp_con:.4f} (Std = {std_logp_con:.4f}, N = {count_logp_con})")
        else:
            final_metrics[f"avg_LogP_con_{target_structure}"] = float('nan')
            final_metrics[f"std_LogP_con_{target_structure}"] = float('nan')
            final_metrics[f"count_LogP_con_{target_structure}"] = 0
            logger.warning(f"    LogP_con : No finite values found.")

        # --- Calculate and store metrics for LogP Incongruent (Avg, Std) ---
        if finite_logp_incon_values:
            count_logp_incon = len(finite_logp_incon_values)
            mean_logp_incon = np.mean(finite_logp_incon_values)
            std_logp_incon = np.std(finite_logp_incon_values)
            final_metrics[f"avg_LogP_incon_{target_structure}"] = mean_logp_incon
            final_metrics[f"std_LogP_incon_{target_structure}"] = std_logp_incon
            final_metrics[f"count_LogP_incon_{target_structure}"] = count_logp_incon
            logger.info(f"    LogP_incon: Avg = {mean_logp_incon:.4f} (Std = {std_logp_incon:.4f}, N = {count_logp_incon})")
        else:
            final_metrics[f"avg_LogP_incon_{target_structure}"] = float('nan')
            final_metrics[f"std_LogP_incon_{target_structure}"] = float('nan')
            final_metrics[f"count_LogP_incon_{target_structure}"] = 0
            logger.warning(f"    LogP_incon: No finite values found.")

    # Restore model mode
    if original_mode:
        model.train()
    logger.info("Native priming evaluation finished.")
    return final_metrics