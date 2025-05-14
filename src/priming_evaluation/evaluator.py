# src/priming_evaluation/evaluator.py (Revised SEM, Baseline Prob, & Return Raw Data)

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Define the structure for individual item results
ResultItem = Dict[str, float] # {'pe': float, 'logp_con': float, 'logp_incon': float, 'logp_baseline': float}
# Define the return type for the batch calculation function
BatchResults = Dict[str, List[ResultItem]]
# Define the return type for the main eval function
# Returns (Aggregated Metrics, Raw Item Results)
EvalResults = Tuple[Dict[str, float], Dict[str, List[ResultItem]]]


def calculate_priming_effect(
    model: PreTrainedModel,
    batch: Dict[str, Any],
    device: torch.device,
    use_amp: bool = False
) -> BatchResults:
    """
    Calculates the Priming Effect (PE), intermediate log probabilities for
    congruent, incongruent, and baseline conditions.
    PE = log P(Target | CongruentPrime) - log P(Target | IncongruentPrime)
    Uses corrected label indexing for log probability calculation.

    Returns:
        Dict[str, List[ResultItem]]: A dictionary where keys are target
        structures and values are lists of dictionaries. Each inner
        dictionary contains 'pe', 'logp_con', 'logp_incon', and 'logp_baseline' for one item.
    """
    # Move tensors to device
    try:
        congruent_input_ids = batch['congruent_input_ids'].to(device)
        congruent_attention_mask = batch['congruent_attention_mask'].to(device)
        incongruent_input_ids = batch['incongruent_input_ids'].to(device)
        incongruent_attention_mask = batch['incongruent_attention_mask'].to(device)
        baseline_input_ids = batch['baseline_input_ids'].to(device) # New baseline inputs
        baseline_attention_mask = batch['baseline_attention_mask'].to(device) # New baseline attention

        labels = batch['labels'].to(device)
        target_starts_con = batch['target_start_congruent']
        target_ends_con = batch['target_end_congruent']
        target_starts_incon = batch['target_start_incongruent']
        target_ends_incon = batch['target_end_incongruent']
        target_starts_base = batch['target_start_baseline'] # New baseline target starts
        target_ends_base = batch['target_end_baseline']     # New baseline target ends

        target_structures = batch['target_structure']

        # Convert indices to tensors on device if they aren't already
        if isinstance(target_starts_con, list): target_starts_con = torch.tensor(target_starts_con, dtype=torch.long, device=device)
        if isinstance(target_ends_con, list): target_ends_con = torch.tensor(target_ends_con, dtype=torch.long, device=device)
        if isinstance(target_starts_incon, list): target_starts_incon = torch.tensor(target_starts_incon, dtype=torch.long, device=device)
        if isinstance(target_ends_incon, list): target_ends_incon = torch.tensor(target_ends_incon, dtype=torch.long, device=device)
        if isinstance(target_starts_base, list): target_starts_base = torch.tensor(target_starts_base, dtype=torch.long, device=device)
        if isinstance(target_ends_base, list): target_ends_base = torch.tensor(target_ends_base, dtype=torch.long, device=device)

    except KeyError as e:
        logger.error(f"Batch missing key: {e}. Cannot calculate PE and baseline metrics.")
        return {}
    except Exception as e:
        logger.error(f"Error moving/preparing batch for device {device}: {e}")
        return {}

    batch_size = congruent_input_ids.size(0)
    batch_results: BatchResults = defaultdict(list)
    nan_result: ResultItem = {'pe': float('nan'), 'logp_con': float('nan'), 'logp_incon': float('nan'), 'logp_baseline': float('nan')}

    # Perform forward passes WITH AUTOCAST
    try:
        amp_enabled = use_amp and device.type == 'cuda'
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs_con = model(input_ids=congruent_input_ids, attention_mask=congruent_attention_mask)
                logits_con = outputs_con.logits
                outputs_incon = model(input_ids=incongruent_input_ids, attention_mask=incongruent_attention_mask)
                logits_incon = outputs_incon.logits
                outputs_base = model(input_ids=baseline_input_ids, attention_mask=baseline_attention_mask) # Baseline forward pass
                logits_base = outputs_base.logits
    except Exception as e:
        logger.error(f"Model forward pass error (AMP enabled: {amp_enabled}): {e}", exc_info=True)
        return {}

    # Calculate log probabilities and PE for each item in the batch
    for i in range(batch_size):
        target_structure = target_structures[i]
        log_prob_con_val = float('nan')
        log_prob_incon_val = float('nan')
        log_prob_baseline_val = float('nan') # Initialize baseline
        priming_effect = float('nan')

        try:
            start_con = target_starts_con[i].item()
            end_con = target_ends_con[i].item()
            start_incon = target_starts_incon[i].item()
            end_incon = target_ends_incon[i].item()
            start_base = target_starts_base[i].item() # Get baseline indices
            end_base = target_ends_base[i].item()     # Get baseline indices

            len_con = logits_con.shape[1]
            len_incon = logits_incon.shape[1]
            len_base = logits_base.shape[1] # Length of baseline logits
            label_len = labels.shape[1]

            # Validate indices (assuming target_starts are 1-based relative to the sequence start in labels)
            # Logits for position t predict token t+1.
            # So, logits[start-1:end-1] predict labels[start:end]
            # Or, more directly, logits for target tokens from `start` to `end-1` are at indices `start-1` to `end-2` of the logits tensor.
            # The target labels themselves are at `start` to `end-1` of the labels tensor.

            # Corrected logic: logits for position `t` predict token `t+1`.
            # We need logits from `start-1` up to `end-1` (exclusive for end) to predict labels from `start` to `end` (exclusive for end).
            # Example: Target is tokens at labels[s_idx...e_idx-1]. We need logits from logit_tensor[s_idx-1 ... e_idx-2]
            # target_labels = labels[i, actual_start_in_label_tensor : actual_end_in_label_tensor]
            # logits_for_target = logits[i, actual_start_in_label_tensor-1 : actual_end_in_label_tensor-1, :]

            # Let's stick to the previous interpretation: target_starts and target_ends define the slice in `labels` directly.
            # `labels[i, start:end]` are the target tokens.
            # `logits[i, start-1:end-1, :]` are the logits predicting these target tokens.

            valid_con_indices = (0 <= start_con -1 < end_con -1 < len_con and # Logit indices
                                 0 <= start_con < end_con <= label_len)      # Label indices
            valid_incon_indices = (0 <= start_incon -1 < end_incon -1 < len_incon and
                                   0 <= start_incon < end_incon <= label_len)
            valid_base_indices = (0 <= start_base -1 < end_base -1 < len_base and   # Logit indices for baseline
                                  0 <= start_base < end_base <= label_len)        # Label indices for baseline

            if not (valid_con_indices and valid_incon_indices and valid_base_indices):
                 logger.warning(
                     f"Index out of bounds for item {i}, target {target_structure}. "
                     f"Con: L({start_con},{end_con}) vs LogitLen={len_con}, LabelLen={label_len}. "
                     f"Incon: L({start_incon},{end_incon}) vs LogitLen={len_incon}, LabelLen={label_len}. "
                     f"Base: L({start_base},{end_base}) vs LogitLen={len_base}, LabelLen={label_len}. Skipping."
                 )
                 batch_results[target_structure].append(nan_result)
                 continue

            # Slice logits and labels
            # IMPORTANT: Assuming start_con/incon/base and end_con/incon/base refer to the same segment in the *labels* tensor
            # but different segments in their respective *logit* tensors.
            # The target tokens are fixed, their probability changes based on preceding context.
            # Thus, `target_labels` should be sliced consistently.
            # The key is that `start_con` and `end_con` define the target sequence in the `labels` tensor.
            # This same definition applies to `start_incon`, `end_incon` and `start_base`, `end_base` *with respect to the labels tensor*.
            # However, the code uses `start_con` to index into `labels` for `target_labels`,
            # and then `start_con-1` to index `logits_con`. This implies `start_con` is consistent.

            target_labels_con = labels[i, start_con : end_con]
            logits_for_target_con = logits_con[i, start_con-1 : end_con-1, :]

            target_labels_incon = labels[i, start_incon : end_incon] # Should be same as target_labels_con if aligned
            logits_for_target_incon = logits_incon[i, start_incon-1 : end_incon-1, :]

            target_labels_base = labels[i, start_base : end_base] # Should be same as target_labels_con if aligned
            logits_for_target_base = logits_base[i, start_base-1 : end_base-1, :]


            # Validate shapes after slicing
            # The number of tokens in the target sequence must be the same across all conditions for a given item.
            if not (logits_for_target_con.shape[0] == target_labels_con.shape[0] and \
                    logits_for_target_incon.shape[0] == target_labels_incon.shape[0] and \
                    logits_for_target_base.shape[0] == target_labels_base.shape[0] and \
                    target_labels_con.shape[0] > 0 and \
                    target_labels_con.shape[0] == target_labels_incon.shape[0] and \
                    target_labels_con.shape[0] == target_labels_base.shape[0]):

                if target_labels_con.shape[0] == 0 or target_labels_incon.shape[0] == 0 or target_labels_base.shape[0] == 0:
                    logger.warning(f"Zero length target sequence for item {i}, target {target_structure}. "
                                   f"Con Lbl len: {target_labels_con.shape[0]}, Incon Lbl len: {target_labels_incon.shape[0]}, Base Lbl len: {target_labels_base.shape[0]}. "
                                   f"Skipping.")
                else:
                    logger.warning(f"Logit/Label length mismatch or inconsistent target lengths for item {i}, target {target_structure}. "
                                f"LogitCon:{logits_for_target_con.shape[0]} vs LabelCon:{target_labels_con.shape[0]}. "
                                f"LogitIncon:{logits_for_target_incon.shape[0]} vs LabelIncon:{target_labels_incon.shape[0]}. "
                                f"LogitBase:{logits_for_target_base.shape[0]} vs LabelBase:{target_labels_base.shape[0]}. "
                                f"Skipping.")
                batch_results[target_structure].append(nan_result)
                continue

            vocab_size = logits_for_target_con.size(-1) # Assume vocab size is consistent

            # LogP Congruent
            log_prob_con_tensor = -F.cross_entropy(
                logits_for_target_con.reshape(-1, vocab_size),
                target_labels_con.reshape(-1),
                ignore_index=-100,
                reduction='sum'
            )
            log_prob_con_val = log_prob_con_tensor.item()

            # LogP Incongruent
            log_prob_incon_tensor = -F.cross_entropy(
                logits_for_target_incon.reshape(-1, vocab_size),
                target_labels_incon.reshape(-1), # Use labels corresponding to incongruent span
                ignore_index=-100,
                reduction='sum'
            )
            log_prob_incon_val = log_prob_incon_tensor.item()

            # LogP Baseline
            log_prob_baseline_tensor = -F.cross_entropy(
                logits_for_target_base.reshape(-1, vocab_size),
                target_labels_base.reshape(-1), # Use labels corresponding to baseline span
                ignore_index=-100,
                reduction='sum'
            )
            log_prob_baseline_val = log_prob_baseline_tensor.item()


            if math.isfinite(log_prob_con_val) and math.isfinite(log_prob_incon_val):
                priming_effect = log_prob_con_val - log_prob_incon_val
            else:
                priming_effect = float('nan')

            current_result: ResultItem = {
                'pe': priming_effect,
                'logp_con': log_prob_con_val,
                'logp_incon': log_prob_incon_val,
                'logp_baseline': log_prob_baseline_val # Add baseline logp
            }

            if not all(math.isfinite(v) for v in [priming_effect, log_prob_con_val, log_prob_incon_val, log_prob_baseline_val]):
                 logger.warning(f"Non-finite value calculated: item {i}, target {target_structure}. PE={priming_effect}, LogP_con={log_prob_con_val}, LogP_incon={log_prob_incon_val}, LogP_base={log_prob_baseline_val}. Storing NaNs where appropriate.")
                 # Ensure nan_result reflects the new key, or fill selectively
                 current_result = {k: v if math.isfinite(v) else float('nan') for k, v in current_result.items()}
                 # If PE became nan due to con/incon being nan, it's already nan.
                 # If baseline is nan, that's fine.
                 # The nan_result should be the fallback if something major goes wrong earlier.
                 if not (math.isfinite(current_result['pe']) and
                         math.isfinite(current_result['logp_con']) and
                         math.isfinite(current_result['logp_incon']) and
                         math.isfinite(current_result['logp_baseline'])):
                    # If any of the primary calculated values are still NaN, log and use a fully NaN result to be safe
                    # (though current_result already has NaNs for non-finite parts).
                    # This specific log line might be redundant if the above warning is sufficient.
                    pass # Values are already NaN if they were non-finite.

            batch_results[target_structure].append(current_result)


        except IndexError as e:
            logger.error(f"IndexError processing metrics for item {i}, target {target_structure}. Err:{e}", exc_info=True)
            batch_results[target_structure].append(nan_result) # Use pre-defined nan_result
        except Exception as e:
            logger.error(f"Unexpected error processing metrics for item {i}, target {target_structure}: {e}", exc_info=True)
            batch_results[target_structure].append(nan_result) # Use pre-defined nan_result

    return dict(batch_results)


def run_native_priming_eval(
    model: PreTrainedModel, priming_dataloader: DataLoader, device: torch.device,
    tokenizer: PreTrainedTokenizer,
    use_amp: bool = False
) -> EvalResults:
    """
    Runs the full native priming evaluation loop, calculating Priming Effect (PE),
    congruent log probability (LogP_con), incongruent log probability (LogP_incon),
    and baseline log probability (LogP_baseline).
    Returns:
        Tuple[Dict[str, float], Dict[str, List[ResultItem]]]:
            - Aggregated metrics (avg, std, sem for PE and LogPs per structure).
            - Raw per-item results (list of ResultItem dicts per structure).
    """
    logger.info("Starting native priming evaluation (PE, LogP_con, LogP_incon, LogP_baseline Metrics)...")
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
        # Ensure item is a dict and key exists before checking isfinite
        finite_pe_values = [r['pe'] for r in structure_results_list if isinstance(r, dict) and 'pe' in r and math.isfinite(r['pe'])]
        finite_logp_con_values = [r['logp_con'] for r in structure_results_list if isinstance(r, dict) and 'logp_con' in r and math.isfinite(r['logp_con'])]
        finite_logp_incon_values = [r['logp_incon'] for r in structure_results_list if isinstance(r, dict) and 'logp_incon' in r and math.isfinite(r['logp_incon'])]
        finite_logp_baseline_values = [r['logp_baseline'] for r in structure_results_list if isinstance(r, dict) and 'logp_baseline' in r and math.isfinite(r['logp_baseline'])] # New

        total_items = len(structure_results_list)
        logger.info(f"  Target '{target_structure}' (Total Items Processed: {total_items}):")

        # --- PE (Avg, Std, SEM) ---
        if finite_pe_values:
            count = len(finite_pe_values)
            mean_val = np.mean(finite_pe_values)
            std_val = np.std(finite_pe_values)
            sem_val = std_val / np.sqrt(count) if count > 0 else float('nan')
            final_aggregated_metrics[f"avg_PE_{target_structure}"] = mean_val
            final_aggregated_metrics[f"std_PE_{target_structure}"] = std_val
            final_aggregated_metrics[f"sem_PE_{target_structure}"] = sem_val
            final_aggregated_metrics[f"count_PE_{target_structure}"] = count
            logger.info(f"    PE          : Avg = {mean_val:.4f} (Std = {std_val:.4f}, SEM = {sem_val:.4f}, N = {count})")
        else:
            final_aggregated_metrics[f"avg_PE_{target_structure}"] = float('nan')
            final_aggregated_metrics[f"std_PE_{target_structure}"] = float('nan')
            final_aggregated_metrics[f"sem_PE_{target_structure}"] = float('nan')
            final_aggregated_metrics[f"count_PE_{target_structure}"] = 0
            logger.warning(f"    PE          : No finite values found.")

        # --- LogP Congruent (Avg, Std, SEM) ---
        if finite_logp_con_values:
            count = len(finite_logp_con_values)
            mean_val = np.mean(finite_logp_con_values)
            std_val = np.std(finite_logp_con_values)
            sem_val = std_val / np.sqrt(count) if count > 0 else float('nan') # Added SEM
            final_aggregated_metrics[f"avg_LogP_con_{target_structure}"] = mean_val
            final_aggregated_metrics[f"std_LogP_con_{target_structure}"] = std_val
            final_aggregated_metrics[f"sem_LogP_con_{target_structure}"] = sem_val # Added SEM
            final_aggregated_metrics[f"count_LogP_con_{target_structure}"] = count
            logger.info(f"    LogP_con    : Avg = {mean_val:.4f} (Std = {std_val:.4f}, SEM = {sem_val:.4f}, N = {count})")
        else:
            final_aggregated_metrics[f"avg_LogP_con_{target_structure}"] = float('nan')
            final_aggregated_metrics[f"std_LogP_con_{target_structure}"] = float('nan')
            final_aggregated_metrics[f"sem_LogP_con_{target_structure}"] = float('nan') # Added SEM
            final_aggregated_metrics[f"count_LogP_con_{target_structure}"] = 0
            logger.warning(f"    LogP_con    : No finite values found.")

        # --- LogP Incongruent (Avg, Std, SEM) ---
        if finite_logp_incon_values:
            count = len(finite_logp_incon_values)
            mean_val = np.mean(finite_logp_incon_values)
            std_val = np.std(finite_logp_incon_values)
            sem_val = std_val / np.sqrt(count) if count > 0 else float('nan') # Added SEM
            final_aggregated_metrics[f"avg_LogP_incon_{target_structure}"] = mean_val
            final_aggregated_metrics[f"std_LogP_incon_{target_structure}"] = std_val
            final_aggregated_metrics[f"sem_LogP_incon_{target_structure}"] = sem_val # Added SEM
            final_aggregated_metrics[f"count_LogP_incon_{target_structure}"] = count
            logger.info(f"    LogP_incon  : Avg = {mean_val:.4f} (Std = {std_val:.4f}, SEM = {sem_val:.4f}, N = {count})")
        else:
            final_aggregated_metrics[f"avg_LogP_incon_{target_structure}"] = float('nan')
            final_aggregated_metrics[f"std_LogP_incon_{target_structure}"] = float('nan')
            final_aggregated_metrics[f"sem_LogP_incon_{target_structure}"] = float('nan') # Added SEM
            final_aggregated_metrics[f"count_LogP_incon_{target_structure}"] = 0
            logger.warning(f"    LogP_incon  : No finite values found.")

        # --- LogP Baseline (Avg, Std, SEM) --- New Section
        if finite_logp_baseline_values:
            count = len(finite_logp_baseline_values)
            mean_val = np.mean(finite_logp_baseline_values)
            std_val = np.std(finite_logp_baseline_values)
            sem_val = std_val / np.sqrt(count) if count > 0 else float('nan')
            final_aggregated_metrics[f"avg_LogP_baseline_{target_structure}"] = mean_val
            final_aggregated_metrics[f"std_LogP_baseline_{target_structure}"] = std_val
            final_aggregated_metrics[f"sem_LogP_baseline_{target_structure}"] = sem_val
            final_aggregated_metrics[f"count_LogP_baseline_{target_structure}"] = count
            logger.info(f"    LogP_baseline: Avg = {mean_val:.4f} (Std = {std_val:.4f}, SEM = {sem_val:.4f}, N = {count})")
        else:
            final_aggregated_metrics[f"avg_LogP_baseline_{target_structure}"] = float('nan')
            final_aggregated_metrics[f"std_LogP_baseline_{target_structure}"] = float('nan')
            final_aggregated_metrics[f"sem_LogP_baseline_{target_structure}"] = float('nan')
            final_aggregated_metrics[f"count_LogP_baseline_{target_structure}"] = 0
            logger.warning(f"    LogP_baseline: No finite values found.")


    if original_mode:
        model.train()
    logger.info("Native priming evaluation finished.")
    return final_aggregated_metrics, dict(all_results_raw)