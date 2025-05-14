# src/priming_evaluation/evaluator.py (Modified for Baseline, assumes valid indices from loader)

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader # Not directly used in these functions but good practice
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer # Tokenizer for type hint

logger = logging.getLogger(__name__)

# Define the structure for individual item results
# ADDED 'logp_baseline'
ResultItem = Dict[str, float] # {'pe': float, 'logp_con': float, 'logp_incon': float, 'logp_baseline': float}
# Define the return type for the batch calculation function
BatchResults = Dict[str, List[ResultItem]]
# Define the return type for the main eval function
EvalResults = Tuple[Dict[str, float], Dict[str, List[ResultItem]]]


def calculate_priming_effect(
    model: PreTrainedModel,
    batch: Dict[str, Any],
    device: torch.device,
    use_amp: bool = False
) -> BatchResults:
    """
    Calculates Priming Effect (PE), congruent, incongruent, and baseline log probabilities.
    PE = log P(Target | CongruentPrime) - log P(Target | IncongruentPrime)
    It relies on the data loader to provide target_start_X >= 1 for all conditions.
    """
    try:
        congruent_input_ids = batch['congruent_input_ids'].to(device)
        congruent_attention_mask = batch['congruent_attention_mask'].to(device)
        incongruent_input_ids = batch['incongruent_input_ids'].to(device)
        incongruent_attention_mask = batch['incongruent_attention_mask'].to(device)
        labels = batch['labels'].to(device)
        target_starts_con = batch['target_start_congruent']
        target_ends_con = batch['target_end_congruent']
        target_starts_incon = batch['target_start_incongruent']
        target_ends_incon = batch['target_end_incongruent']
        target_structures = batch['target_structure']

        # ADDED: Baseline inputs
        baseline_input_ids = batch['baseline_input_ids'].to(device)
        baseline_attention_mask = batch['baseline_attention_mask'].to(device)
        target_starts_base = batch['target_start_baseline']
        target_ends_base = batch['target_end_baseline']

        # Convert indices to tensors on device if they aren't already
        if isinstance(target_starts_con, list): target_starts_con = torch.tensor(target_starts_con, dtype=torch.long, device=device)
        if isinstance(target_ends_con, list): target_ends_con = torch.tensor(target_ends_con, dtype=torch.long, device=device)
        if isinstance(target_starts_incon, list): target_starts_incon = torch.tensor(target_starts_incon, dtype=torch.long, device=device)
        if isinstance(target_ends_incon, list): target_ends_incon = torch.tensor(target_ends_incon, dtype=torch.long, device=device)
        if isinstance(target_starts_base, list): target_starts_base = torch.tensor(target_starts_base, dtype=torch.long, device=device) # ADDED
        if isinstance(target_ends_base, list): target_ends_base = torch.tensor(target_ends_base, dtype=torch.long, device=device) # ADDED

    except KeyError as e:
        logger.error(f"Batch missing key: {e}. Cannot calculate PE and baseline metrics.")
        return {}
    except Exception as e:
        logger.error(f"Error moving/preparing batch for device {device}: {e}")
        return {}

    batch_size = congruent_input_ids.size(0)
    batch_results: BatchResults = defaultdict(list)
    # ADDED 'logp_baseline' to nan_result
    nan_result: ResultItem = {'pe': float('nan'), 'logp_con': float('nan'), 'logp_incon': float('nan'), 'logp_baseline': float('nan')}

    try:
        amp_enabled = use_amp and device.type == 'cuda'
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs_con = model(input_ids=congruent_input_ids, attention_mask=congruent_attention_mask)
                logits_con = outputs_con.logits
                outputs_incon = model(input_ids=incongruent_input_ids, attention_mask=incongruent_attention_mask)
                logits_incon = outputs_incon.logits
                # ADDED: Baseline forward pass
                outputs_base = model(input_ids=baseline_input_ids, attention_mask=baseline_attention_mask)
                logits_base = outputs_base.logits
    except Exception as e:
        logger.error(f"Model forward pass error (AMP enabled: {amp_enabled}): {e}", exc_info=True)
        return {}

    for i in range(batch_size):
        target_structure = target_structures[i]
        log_prob_con_val = float('nan')
        log_prob_incon_val = float('nan')
        log_prob_baseline_val = float('nan') # ADDED
        priming_effect = float('nan')

        try:
            start_con = target_starts_con[i].item()
            end_con = target_ends_con[i].item()
            start_incon = target_starts_incon[i].item()
            end_incon = target_ends_incon[i].item()
            start_base = target_starts_base[i].item() # ADDED
            end_base = target_ends_base[i].item()     # ADDED

            len_con = logits_con.shape[1]
            len_incon = logits_incon.shape[1]
            len_base = logits_base.shape[1] # ADDED
            label_len = labels.shape[1]

            # Existing validation logic: start_X - 1 must be a valid index.
            # This means start_X must be >= 1.
            # The data loader MUST ensure this for start_base as well.
            valid_con_indices = (0 <= start_con - 1 < end_con - 1 < len_con and
                                 start_con < end_con <= label_len)
            valid_incon_indices = (0 <= start_incon - 1 < end_incon - 1 < len_incon and
                                   start_incon < end_incon <= label_len)
            valid_base_indices = (0 <= start_base - 1 < end_base - 1 < len_base and    # ADDED
                                  start_base < end_base <= label_len)                 # ADDED

            if not (valid_con_indices and valid_incon_indices and valid_base_indices): # MODIFIED Check
                 logger.warning(
                     f"Index out of bounds for item {i}, target {target_structure}. "
                     f"Con: ({start_con},{end_con}) vs LogitLen={len_con}, LabelLen={label_len}. "
                     f"Incon: ({start_incon},{end_incon}) vs LogitLen={len_incon}, LabelLen={label_len}. "
                     f"Base: ({start_base},{end_base}) vs LogitLen={len_base}, LabelLen={label_len}. Skipping." # ADDED Base info
                 )
                 batch_results[target_structure].append(nan_result)
                 continue

            # Logits slices
            logits_for_target_con = logits_con[i, start_con-1 : end_con-1, :]
            logits_for_target_incon = logits_incon[i, start_incon-1 : end_incon-1, :]
            logits_for_target_base = logits_base[i, start_base-1 : end_base-1, :] # ADDED

            # Label slices - using the same start/end indices on the common 'labels' tensor
            # This assumes the target tokens are identical and occupy the same relative span
            # if aligned to the start of the target sequence.
            target_labels_con = labels[i, start_con : end_con]
            target_labels_incon = labels[i, start_incon : end_incon] # Potentially different slice if alignment shifts
            target_labels_base = labels[i, start_base : end_base]    # ADDED

            # It's crucial that target_labels_con, _incon, _base refer to the *same actual target tokens*.
            # The most robust way for this is if labels[i, common_target_start:common_target_end] holds the target,
            # and start_con/incon/base are all relative to the start of this common_target_start.
            # However, the current script implies start_X refers to a position in the *full* labels tensor.
            # For simplicity, let's assume the target length is consistent.
            # The most critical part is `target_labels_con.shape[0]` matching others.
            # We'll use target_labels_con as the reference for labels in CE, assuming its indices are correct for the target.

            if not (logits_for_target_con.shape[0] == target_labels_con.shape[0] and \
                    logits_for_target_incon.shape[0] == target_labels_con.shape[0] and \
                    logits_for_target_base.shape[0] == target_labels_con.shape[0] and \
                    target_labels_con.shape[0] > 0):
                if target_labels_con.shape[0] == 0:
                     logger.warning(f"Zero length target sequence derived from congruent for item {i}, target {target_structure}. Indices: ({start_con}:{end_con}). Skipping.")
                else:
                    logger.warning(f"Logit/Label length mismatch for item {i}, target {target_structure}. "
                                   f"LogitCon:{logits_for_target_con.shape[0]}, "
                                   f"LogitIncon:{logits_for_target_incon.shape[0]}, "
                                   f"LogitBase:{logits_for_target_base.shape[0]}, "
                                   f"Expected LabelLen (from con):{target_labels_con.shape[0]}. "
                                   f"Indices: Con({start_con}:{end_con}), Incon({start_incon}:{end_incon}), Base({start_base}:{end_base}). Skipping.")
                batch_results[target_structure].append(nan_result)
                continue

            vocab_size = logits_for_target_con.size(-1)

            log_prob_con_tensor = -F.cross_entropy(
                logits_for_target_con.reshape(-1, vocab_size),
                target_labels_con.reshape(-1), # Use target_labels_con as the reference
                ignore_index=-100,
                reduction='sum'
            )
            log_prob_incon_tensor = -F.cross_entropy(
                logits_for_target_incon.reshape(-1, vocab_size),
                target_labels_con.reshape(-1), # Use target_labels_con as the reference
                ignore_index=-100,
                reduction='sum'
            )
            log_prob_baseline_tensor = -F.cross_entropy( # ADDED
                logits_for_target_base.reshape(-1, vocab_size),
                target_labels_con.reshape(-1), # Use target_labels_con as the reference
                ignore_index=-100,
                reduction='sum'
            )

            log_prob_con_val = log_prob_con_tensor.item()
            log_prob_incon_val = log_prob_incon_tensor.item()
            log_prob_baseline_val = log_prob_baseline_tensor.item() # ADDED

            if math.isfinite(log_prob_con_val) and math.isfinite(log_prob_incon_val):
                priming_effect = log_prob_con_val - log_prob_incon_val
            else:
                priming_effect = float('nan')

            current_result: ResultItem = {
                'pe': priming_effect,
                'logp_con': log_prob_con_val,
                'logp_incon': log_prob_incon_val,
                'logp_baseline': log_prob_baseline_val # ADDED
            }

            # Check all values for finiteness now
            if not all(math.isfinite(v) for v in current_result.values()):
                 logger.warning(f"Non-finite value calculated: item {i}, target {target_structure}. "
                                f"PE={current_result['pe']:.4f}, LogP_con={current_result['logp_con']:.4f}, "
                                f"LogP_incon={current_result['logp_incon']:.4f}, LogP_base={current_result['logp_baseline']:.4f}. Storing NaNs.")
                 # Replace non-finite values with NaN for this specific item
                 final_item_result = {k: v if math.isfinite(v) else float('nan') for k, v in current_result.items()}
                 batch_results[target_structure].append(final_item_result)
            else:
                 batch_results[target_structure].append(current_result)

        except IndexError as e: # Catching potential index errors if validation wasn't enough
            logger.error(f"IndexError during metric calculation for item {i}, target {target_structure}. Err:{e}", exc_info=True)
            batch_results[target_structure].append(nan_result)
        except Exception as e:
            logger.error(f"Unexpected error processing metrics for item {i}, target {target_structure}: {e}", exc_info=True)
            batch_results[target_structure].append(nan_result)

    return dict(batch_results)


def run_native_priming_eval(
    model: PreTrainedModel, priming_dataloader: DataLoader, device: torch.device,
    tokenizer: PreTrainedTokenizer, # Keep tokenizer for potential debugging
    use_amp: bool = False
) -> EvalResults:
    logger.info("Starting native priming evaluation (PE, LogP_con, LogP_incon, LogP_baseline Metrics)...") # MODIFIED
    original_mode = model.training
    model.eval()

    all_results_raw: Dict[str, List[ResultItem]] = defaultdict(list)

    progress_bar = tqdm(priming_dataloader, desc="Priming Eval", leave=False)
    for batch in progress_bar:
        if not batch:
            logger.warning("Skipping empty batch from collate function.")
            continue
        try:
            batch_metrics_raw: BatchResults = calculate_priming_effect(model, batch, device, use_amp=use_amp)
            for target_structure, result_list in batch_metrics_raw.items():
                all_results_raw[target_structure].extend(result_list)
        except Exception as e:
            logger.error(f"Error processing priming batch: {e}", exc_info=True)

    final_aggregated_metrics: Dict[str, float] = {}
    logger.info("Calculating final priming metric aggregates:")

    for target_structure, structure_results_list in all_results_raw.items():
        finite_pe_values = [r['pe'] for r in structure_results_list if isinstance(r, dict) and 'pe' in r and math.isfinite(r['pe'])]
        finite_logp_con_values = [r['logp_con'] for r in structure_results_list if isinstance(r, dict) and 'logp_con' in r and math.isfinite(r['logp_con'])]
        finite_logp_incon_values = [r['logp_incon'] for r in structure_results_list if isinstance(r, dict) and 'logp_incon' in r and math.isfinite(r['logp_incon'])]
        finite_logp_baseline_values = [r['logp_baseline'] for r in structure_results_list if isinstance(r, dict) and 'logp_baseline' in r and math.isfinite(r['logp_baseline'])] # ADDED

        total_items = len(structure_results_list)
        logger.info(f"  Target '{target_structure}' (Total Items Processed: {total_items}):")

        # PE
        if finite_pe_values:
            count = len(finite_pe_values)
            mean_val, std_val = np.mean(finite_pe_values), np.std(finite_pe_values)
            sem_val = std_val / np.sqrt(count) if count > 0 else float('nan')
            final_aggregated_metrics[f"avg_PE_{target_structure}"] = mean_val
            final_aggregated_metrics[f"std_PE_{target_structure}"] = std_val
            final_aggregated_metrics[f"sem_PE_{target_structure}"] = sem_val
            final_aggregated_metrics[f"count_PE_{target_structure}"] = count
            logger.info(f"    PE          : Avg = {mean_val:.4f} (Std = {std_val:.4f}, SEM = {sem_val:.4f}, N = {count})")
        else:
            for metric in ["avg_PE", "std_PE", "sem_PE", "count_PE"]: final_aggregated_metrics[f"{metric}_{target_structure}"] = float('nan') if metric != "count_PE" else 0
            logger.warning(f"    PE          : No finite values found.")

        # LogP Congruent (Added SEM)
        if finite_logp_con_values:
            count = len(finite_logp_con_values)
            mean_val, std_val = np.mean(finite_logp_con_values), np.std(finite_logp_con_values)
            sem_val = std_val / np.sqrt(count) if count > 0 else float('nan') # ADDED SEM
            final_aggregated_metrics[f"avg_LogP_con_{target_structure}"] = mean_val
            final_aggregated_metrics[f"std_LogP_con_{target_structure}"] = std_val
            final_aggregated_metrics[f"sem_LogP_con_{target_structure}"] = sem_val # ADDED SEM
            final_aggregated_metrics[f"count_LogP_con_{target_structure}"] = count
            logger.info(f"    LogP_con    : Avg = {mean_val:.4f} (Std = {std_val:.4f}, SEM = {sem_val:.4f}, N = {count})")
        else:
            for metric in ["avg_LogP_con", "std_LogP_con", "sem_LogP_con", "count_LogP_con"]: final_aggregated_metrics[f"{metric}_{target_structure}"] = float('nan') if metric != "count_LogP_con" else 0
            logger.warning(f"    LogP_con    : No finite values found.")

        # LogP Incongruent (Added SEM)
        if finite_logp_incon_values:
            count = len(finite_logp_incon_values)
            mean_val, std_val = np.mean(finite_logp_incon_values), np.std(finite_logp_incon_values)
            sem_val = std_val / np.sqrt(count) if count > 0 else float('nan') # ADDED SEM
            final_aggregated_metrics[f"avg_LogP_incon_{target_structure}"] = mean_val
            final_aggregated_metrics[f"std_LogP_incon_{target_structure}"] = std_val
            final_aggregated_metrics[f"sem_LogP_incon_{target_structure}"] = sem_val # ADDED SEM
            final_aggregated_metrics[f"count_LogP_incon_{target_structure}"] = count
            logger.info(f"    LogP_incon  : Avg = {mean_val:.4f} (Std = {std_val:.4f}, SEM = {sem_val:.4f}, N = {count})")
        else:
            for metric in ["avg_LogP_incon", "std_LogP_incon", "sem_LogP_incon", "count_LogP_incon"]: final_aggregated_metrics[f"{metric}_{target_structure}"] = float('nan') if metric != "count_LogP_incon" else 0
            logger.warning(f"    LogP_incon  : No finite values found.")

        # ADDED: LogP Baseline (Avg, Std, SEM)
        if finite_logp_baseline_values:
            count = len(finite_logp_baseline_values)
            mean_val, std_val = np.mean(finite_logp_baseline_values), np.std(finite_logp_baseline_values)
            sem_val = std_val / np.sqrt(count) if count > 0 else float('nan')
            final_aggregated_metrics[f"avg_LogP_baseline_{target_structure}"] = mean_val
            final_aggregated_metrics[f"std_LogP_baseline_{target_structure}"] = std_val
            final_aggregated_metrics[f"sem_LogP_baseline_{target_structure}"] = sem_val
            final_aggregated_metrics[f"count_LogP_baseline_{target_structure}"] = count
            logger.info(f"    LogP_baseline: Avg = {mean_val:.4f} (Std = {std_val:.4f}, SEM = {sem_val:.4f}, N = {count})")
        else:
            for metric in ["avg_LogP_baseline", "std_LogP_baseline", "sem_LogP_baseline", "count_LogP_baseline"]: final_aggregated_metrics[f"{metric}_{target_structure}"] = float('nan') if metric != "count_LogP_baseline" else 0
            logger.warning(f"    LogP_baseline: No finite values found.")

    if original_mode:
        model.train()
    logger.info("Native priming evaluation finished.")
    return final_aggregated_metrics, dict(all_results_raw)