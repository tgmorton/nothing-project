# src/priming_evaluation/evaluator.py (Revised to calculate PE, Normalized Probs, and Baseline)

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

# --- NEW: Define more detailed result structures ---
# Raw log-probabilities calculated for each item
RawLogProbs = Dict[str, float]

# A dictionary to hold lists of raw log-probs, keyed by target structure
BatchResults = Dict[str, List[RawLogProbs]]

# Final return type: (Aggregated Metrics, Raw Individual Item Metrics)
FinalResultItem = Dict[str, float]
EvalResults = Tuple[Dict[str, float], Dict[str, List[FinalResultItem]]]


def calculate_full_metrics_for_batch(
        model: PreTrainedModel,
        batch: Dict[str, Any],
        device: torch.device,
        tokenizer: PreTrainedTokenizer,
        use_amp: bool = False
) -> BatchResults:
    """
    Calculates all required log-probabilities for a batch of data.
    This includes P(Target|Prime) for all four combinations, plus baseline P(Target).
    """
    # This function now expects input_ids and masks for all four sentence parts,
    # plus the standalone targets for baseline calculation.
    # e.g., batch['congruent_prime_input_ids'], batch['congruent_target_input_ids'], etc.

    # We will compute log-probabilities for 6 scenarios per item.
    # To do this efficiently, we create one large mega-batch, run the model once,
    # and then split the results.

    try:
        # --- NEW: Concatenate all inputs for a single mega-batch forward pass ---
        # This assumes your dataloader prepares these tensors.
        # Primes + Targets for contextualized probabilities
        con_prime_con_target = torch.cat([batch['con_prime_input_ids'], batch['con_target_input_ids']], dim=1)
        con_prime_incon_target = torch.cat([batch['con_prime_input_ids'], batch['incon_target_input_ids']], dim=1)
        incon_prime_con_target = torch.cat([batch['incon_prime_input_ids'], batch['con_target_input_ids']], dim=1)
        incon_prime_incon_target = torch.cat([batch['incon_prime_input_ids'], batch['incon_target_input_ids']], dim=1)

        # Standalone targets for baseline probabilities
        # Add BOS token to the beginning of standalone targets for proper probability calculation
        bos_tensor = torch.tensor([[tokenizer.bos_token_id]], device=device).expand(
            batch['con_target_input_ids'].size(0), -1)
        base_con_target = torch.cat([bos_tensor, batch['con_target_input_ids']], dim=1)
        base_incon_target = torch.cat([bos_tensor, batch['incon_target_input_ids']], dim=1)

        # Combine all 6 into a single tensor for one forward pass
        all_input_ids = torch.cat([
            con_prime_con_target, con_prime_incon_target,
            incon_prime_con_target, incon_prime_incon_target,
            base_con_target, base_incon_target
        ], dim=0).to(device)

        # Create corresponding attention masks
        all_attention_mask = (all_input_ids != tokenizer.pad_token_id).long()

    except (KeyError, AttributeError) as e:
        logger.error(
            f"Batch missing a required key or attribute (e.g., 'con_prime_input_ids'): {e}. Check your dataloader.")
        return {}
    except Exception as e:
        logger.error(f"Error preparing mega-batch for device {device}: {e}")
        return {}

    # Get batch size from one of the original tensors
    original_batch_size = batch['con_prime_input_ids'].size(0)

    # Perform a single forward pass on the combined batch
    try:
        amp_enabled = use_amp and device.type == 'cuda'
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(input_ids=all_input_ids, attention_mask=all_attention_mask)
                logits = outputs.logits
    except Exception as e:
        logger.error(f"Model forward pass error on mega-batch (AMP enabled: {amp_enabled}): {e}", exc_info=True)
        return {}

    # Create labels by shifting inputs, ignoring padding
    labels = all_input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    # Split the logits and labels back into 6 parts
    logits_list = torch.chunk(logits, 6, dim=0)
    labels_list = torch.chunk(labels, 6, dim=0)

    # Define slices for calculating target log-probabilities
    # These indices refer to the position *within the concatenated sequence*
    # This requires your dataloader to provide these start indices.
    # Example: 'con_target_start_in_con_prime_con_target_seq'
    target_starts = [
        batch['con_target_start_in_con_prime_context'], batch['incon_target_start_in_con_prime_context'],
        batch['con_target_start_in_incon_prime_context'], batch['incon_target_start_in_incon_prime_context'],
        1, 1  # For baseline targets, the target starts at index 1 (after BOS)
    ]

    log_prob_keys = [
        'logp_conT_conP', 'logp_inconT_conP',
        'logp_conT_inconP', 'logp_inconT_inconP',
        'logp_conT_base', 'logp_inconT_base'
    ]

    # Calculate log-probabilities for each of the 6 scenarios
    log_probs_per_item = [defaultdict(float) for _ in range(original_batch_size)]

    for k, (key, logit_tensor, label_tensor, start_indices) in enumerate(
            zip(log_prob_keys, logits_list, labels_list, target_starts)):
        vocab_size = logit_tensor.size(-1)
        for i in range(original_batch_size):
            try:
                # The logits needed are from one position before the target starts, up to one before the end
                target_start_idx = start_indices[i].item()
                # Find the end of the sequence (first -100 label after start)
                item_labels = label_tensor[i, target_start_idx:]
                non_padding_len = (item_labels != -100).sum().item()

                if non_padding_len == 0:
                    logger.warning(f"Item {i} has zero-length target for '{key}'. Skipping.")
                    log_probs_per_item[i][key] = float('nan')
                    continue

                target_end_idx = target_start_idx + non_padding_len

                logits_for_target = logit_tensor[i, target_start_idx - 1: target_end_idx - 1, :]
                labels_for_target = label_tensor[i, target_start_idx: target_end_idx]

                # Ensure shapes match
                if logits_for_target.shape[0] != labels_for_target.shape[0]:
                    raise ValueError("Logit and label shapes do not match after slicing.")

                log_prob_tensor = -F.cross_entropy(
                    logits_for_target.view(-1, vocab_size),
                    labels_for_target.view(-1),
                    ignore_index=-100,
                    reduction='sum'
                )
                log_probs_per_item[i][key] = log_prob_tensor.item()

            except Exception as e:
                logger.error(f"Error calculating log-prob for item {i}, key '{key}': {e}", exc_info=True)
                log_probs_per_item[i][key] = float('nan')

    # Organize results by target structure from the original batch
    batch_results: BatchResults = defaultdict(list)
    target_structures = batch.get('target_structure', ['default_structure'] * original_batch_size)
    for i in range(original_batch_size):
        batch_results[target_structures[i]].append(log_probs_per_item[i])

    return dict(batch_results)


def run_native_priming_eval(
        model: PreTrainedModel, priming_dataloader: DataLoader, device: torch.device,
        tokenizer: PreTrainedTokenizer,
        use_amp: bool = False
) -> EvalResults:
    """
    Runs the full native priming evaluation loop.

    NOW CALCULATES:
    1. Priming Effect (PE) via log-prob subtraction ("Sinclair" method).
    2. Normalized Probabilities (Paper's method).
    3. Prime-less Baseline Preference.
    """
    logger.info("Starting expanded native priming evaluation...")
    original_mode = model.training
    model.eval()

    all_raw_log_probs: Dict[str, List[RawLogProbs]] = defaultdict(list)

    progress_bar = tqdm(priming_dataloader, desc="Priming Eval", leave=False)
    for batch in progress_bar:
        if not batch:
            logger.warning("Skipping empty batch from collate function.")
            continue
        try:
            batch_metrics_raw = calculate_full_metrics_for_batch(model, batch, device, tokenizer, use_amp=use_amp)
            for target_structure, result_list in batch_metrics_raw.items():
                all_raw_log_probs[target_structure].extend(result_list)
        except Exception as e:
            logger.error(f"Error processing priming batch: {e}", exc_info=True)

    # --- NEW: Calculate all final metrics from the collected raw log-probs ---
    final_aggregated_metrics = {}
    final_raw_item_metrics: Dict[str, List[FinalResultItem]] = defaultdict(list)

    for structure, results_list in all_raw_log_probs.items():
        pe_scores = []
        norm_p_con_target_given_con_prime = []
        norm_p_con_target_given_incon_prime = []
        baseline_prefs_for_con_target = []

        for raw_probs in results_list:
            item_metrics = {}

            # Use nan-safe retrieval
            logp_ct_cp = raw_probs.get('logp_conT_conP', float('nan'))
            logp_it_cp = raw_probs.get('logp_inconT_conP', float('nan'))
            logp_ct_ip = raw_probs.get('logp_conT_inconP', float('nan'))
            logp_it_ip = raw_probs.get('logp_inconT_inconP', float('nan'))
            logp_ct_base = raw_probs.get('logp_conT_base', float('nan'))
            logp_it_base = raw_probs.get('logp_inconT_base', float('nan'))

            # --- 1. Calculate Priming Effect (PE) Score ---
            if math.isfinite(logp_ct_cp) and math.isfinite(logp_ct_ip):
                pe = logp_ct_cp - logp_ct_ip
                pe_scores.append(pe)
                item_metrics['pe_sinclair'] = pe

            # --- 2. Calculate Normalized Probabilities ---
            p_ct_cp = math.exp(logp_ct_cp) if math.isfinite(logp_ct_cp) else 0
            p_it_cp = math.exp(logp_it_cp) if math.isfinite(logp_it_cp) else 0
            if p_ct_cp + p_it_cp > 0:
                norm_p = p_ct_cp / (p_ct_cp + p_it_cp)
                norm_p_con_target_given_con_prime.append(norm_p)
                item_metrics['norm_p_conT_given_conP'] = norm_p

            p_ct_ip = math.exp(logp_ct_ip) if math.isfinite(logp_ct_ip) else 0
            p_it_ip = math.exp(logp_it_ip) if math.isfinite(logp_it_ip) else 0
            if p_ct_ip + p_it_ip > 0:
                norm_p = p_ct_ip / (p_ct_ip + p_it_ip)
                norm_p_con_target_given_incon_prime.append(norm_p)
                item_metrics['norm_p_conT_given_inconP'] = norm_p

            # --- 3. Calculate Baseline Preference ---
            p_ct_base = math.exp(logp_ct_base) if math.isfinite(logp_ct_base) else 0
            p_it_base = math.exp(logp_it_base) if math.isfinite(logp_it_base) else 0
            if p_ct_base + p_it_base > 0:
                baseline_pref = p_ct_base / (p_ct_base + p_it_base)
                baseline_prefs_for_con_target.append(baseline_pref)
                item_metrics['baseline_pref_for_conT'] = baseline_pref

            final_raw_item_metrics[structure].append(item_metrics)

        # --- Aggregate and store all metrics ---
        # PE ("Sinclair") metrics
        final_aggregated_metrics[f'avg_pe_sinclair_{structure}'] = np.mean(pe_scores) if pe_scores else float('nan')
        final_aggregated_metrics[f'std_pe_sinclair_{structure}'] = np.std(pe_scores) if pe_scores else float('nan')

        # Normalized Probability metrics
        final_aggregated_metrics[f'avg_norm_p_conT_given_conP_{structure}'] = np.mean(
            norm_p_con_target_given_con_prime) if norm_p_con_target_given_con_prime else float('nan')
        final_aggregated_metrics[f'avg_norm_p_conT_given_inconP_{structure}'] = np.mean(
            norm_p_con_target_given_incon_prime) if norm_p_con_target_given_incon_prime else float('nan')

        # Baseline Preference metric
        final_aggregated_metrics[f'avg_baseline_pref_for_conT_{structure}'] = np.mean(
            baseline_prefs_for_con_target) if baseline_prefs_for_con_target else float('nan')
        final_aggregated_metrics[f'count_{structure}'] = len(results_list)

    if original_mode:
        model.train()
    logger.info("Expanded native priming evaluation finished.")

    # Return both the summary aggregates and the raw per-item calculated metrics
    return final_aggregated_metrics, dict(final_raw_item_metrics)