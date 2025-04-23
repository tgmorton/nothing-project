# src/priming_evaluation/evaluator.py (Revised PE Calculation)

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

def calculate_priming_effect(
    model: PreTrainedModel,
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, List[float]]:
    """
    Calculates the Priming Effect (PE) for a batch of data.
    PE = log P(Target | CongruentPrime) - log P(Target | IncongruentPrime)
    Uses corrected label indexing for log probability calculation.
    """
    # Move tensors to device
    try:
        congruent_input_ids = batch['congruent_input_ids'].to(device)
        congruent_attention_mask = batch['congruent_attention_mask'].to(device)
        incongruent_input_ids = batch['incongruent_input_ids'].to(device)
        incongruent_attention_mask = batch['incongruent_attention_mask'].to(device)
        labels = batch['labels'].to(device) # Padded target token IDs, -100 elsewhere
        target_starts_con = batch['target_start_congruent'] # Tensor on CPU
        target_ends_con = batch['target_end_congruent']     # Tensor on CPU
        target_starts_incon = batch['target_start_incongruent'] # Tensor on CPU
        target_ends_incon = batch['target_end_incongruent']   # Tensor on CPU
        target_structures = batch['target_structure'] # List of strings
    except KeyError as e: logger.error(f"Batch missing key: {e}. Cannot calc PE."); return {}
    except Exception as e: logger.error(f"Error moving batch to device {device}: {e}"); return {}

    batch_size = congruent_input_ids.size(0)
    batch_pe_results = defaultdict(list)

    # Perform forward passes
    try:
        with torch.no_grad():
            outputs_con = model(input_ids=congruent_input_ids, attention_mask=congruent_attention_mask)
            logits_con = outputs_con.logits # (batch, seq_len_con, vocab)
            outputs_incon = model(input_ids=incongruent_input_ids, attention_mask=incongruent_attention_mask)
            logits_incon = outputs_incon.logits # (batch, seq_len_incon, vocab)
    except Exception as e: logger.error(f"Model forward pass error: {e}", exc_info=True); return {}

    # Calculate log probabilities
    for i in range(batch_size):
        target_structure = target_structures[i]
        try:
            # --- Indices for Target Sequence ---
            start_con = target_starts_con[i].item()
            end_con = target_ends_con[i].item()
            start_incon = target_starts_incon[i].item()
            end_incon = target_ends_incon[i].item()

            # --- Logits for predicting target tokens ---
            # Logits at position `t` predict token `t+1`.
            # To predict tokens from `start` to `end-1`, we need logits from `start-1` to `end-2`.
            logits_for_target_con = logits_con[i, start_con-1 : end_con-1, :] # Shape: (target_len, vocab_size)
            logits_for_target_incon = logits_incon[i, start_incon-1 : end_incon-1, :] # Shape: (target_len, vocab_size)

            # --- Labels for the target tokens ---
            # These are the actual token IDs at positions `start` to `end-1`.
            # The `labels` tensor already has -100 padding outside this range.
            # We need the slice corresponding to the target tokens.
            target_labels = labels[i, start_con : end_con] # Shape: (target_len)

            # --- Verify lengths match ---
            if logits_for_target_con.shape[0] != target_labels.shape[0] or \
               logits_for_target_incon.shape[0] != target_labels.shape[0]:
                logger.warning(f"Logit/Label length mismatch for item {i}, target {target_structure}. "
                               f"LogitCon:{logits_for_target_con.shape[0]}, LogitIncon:{logits_for_target_incon.shape[0]}, Label:{target_labels.shape[0]}. Skipping.")
                batch_pe_results[target_structure].append(float('nan'))
                continue

            # --- Calculate Log Prob using cross_entropy ---
            # cross_entropy calculates -log(softmax(logits))[label] summed over sequence
            # It handles ignore_index=-100 internally.
            # Reshape for cross_entropy: (N, C) and (N)
            vocab_size = logits_for_target_con.size(-1)

            log_prob_con = -F.cross_entropy(
                logits_for_target_con.reshape(-1, vocab_size), # (target_len, vocab) -> (target_len, vocab)
                target_labels.reshape(-1), # (target_len) -> (target_len)
                ignore_index=-100,
                reduction='sum' # Sum log probabilities over the target sequence
            )

            log_prob_incon = -F.cross_entropy(
                logits_for_target_incon.reshape(-1, vocab_size),
                target_labels.reshape(-1), # Same target labels
                ignore_index=-100,
                reduction='sum'
            )

            # --- Priming Effect (PE) ---
            priming_effect = log_prob_con.item() - log_prob_incon.item()

            if not math.isfinite(priming_effect):
                 logger.warning(f"Non-finite PE: item {i}, target {target_structure}. LogP_con={log_prob_con.item()}, LogP_incon={log_prob_incon.item()}")
                 batch_pe_results[target_structure].append(float('nan'))
            else:
                 # Log detailed calculation for debugging one item
                 # if i == 0 and logger.isEnabledFor(logging.DEBUG):
                 #      logger.debug(f"Item 0 ({target_structure}): LogP_con={log_prob_con.item():.4f}, LogP_incon={log_prob_incon.item():.4f}, PE={priming_effect:.4f}")
                 batch_pe_results[target_structure].append(priming_effect)

        except IndexError as e:
            logger.error(f"IndexError calc PE item {i}, target {target_structure}. Indices:{start_con=},{end_con=},{start_incon=},{end_incon=}. Err:{e}")
            batch_pe_results[target_structure].append(float('nan'))
        except Exception as e:
            logger.error(f"Unexpected error calc PE item {i}, target {target_structure}: {e}", exc_info=True)
            batch_pe_results[target_structure].append(float('nan'))

    return dict(batch_pe_results)

# --- run_native_priming_eval (No changes needed from previous version) ---
def run_native_priming_eval(
    model: PreTrainedModel, priming_dataloader: DataLoader, device: torch.device,
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, float]:
    """ Runs the full native priming evaluation loop using the Priming Effect metric. """
    logger.info("Starting native priming evaluation (PE Metric)...")
    original_mode = model.training; model.eval()
    all_pe_results = defaultdict(list)
    progress_bar = tqdm(priming_dataloader, desc="Priming Eval (PE)", leave=False)
    for batch in progress_bar:
        if not batch: logger.warning("Skipping empty batch from collate."); continue
        try:
            batch_metrics = calculate_priming_effect(model, batch, device)
            for target_structure, pe_values in batch_metrics.items(): all_pe_results[target_structure].extend(pe_values)
        except Exception as e: logger.error(f"Error processing priming batch: {e}", exc_info=True)

    final_metrics = {}; logger.info("Calculating final Priming Effect averages:")
    for target_structure, pe_list in all_pe_results.items():
        finite_pe_values = [v for v in pe_list if isinstance(v, (int, float)) and math.isfinite(v)]
        if finite_pe_values:
            mean_pe, std_pe, count = np.mean(finite_pe_values), np.std(finite_pe_values), len(finite_pe_values)
            final_metrics[f"avg_PE_{target_structure}"] = mean_pe; final_metrics[f"std_PE_{target_structure}"] = std_pe; final_metrics[f"count_{target_structure}"] = count
            logger.info(f"  Target '{target_structure}': Avg PE = {mean_pe:.4f} (Std: {std_pe:.4f}, N = {count} / {len(pe_list)})")
        else:
            final_metrics[f"avg_PE_{target_structure}"] = float('nan'); final_metrics[f"std_PE_{target_structure}"] = float('nan'); final_metrics[f"count_{target_structure}"] = 0
            logger.warning(f"  Target '{target_structure}': No finite PE values found (Total items: {len(pe_list)}).")
    if original_mode: model.train()
    logger.info("Native priming evaluation (PE Metric) finished.")
    return final_metrics