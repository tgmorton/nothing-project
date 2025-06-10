# src/priming_evaluation/evaluator.py (Corrected for device placement)

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

# (The result structures definitions remain the same)
RawLogProbs = Dict[str, float]
BatchResults = Dict[str, List[RawLogProbs]]
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
    Calculates log-probabilities from pre-assembled and padded tensors.
    """
    try:
        # --- KEY CHANGE: The batch now contains fully-formed sequences. We just move them to the device. ---
        sequence_keys = [
            'con_prime_con_target_ids', 'con_prime_incon_target_ids',
            'incon_prime_con_target_ids', 'incon_prime_incon_target_ids',
            'base_con_target_ids', 'base_incon_target_ids'
        ]
        all_input_ids_list = []
        for key in sequence_keys:
            if key in batch and isinstance(batch[key], torch.Tensor):
                all_input_ids_list.append(batch[key].to(device))
            else:
                raise KeyError(f"Batch from DataLoader is missing required tensor key: {key}")

        # Now we create the single mega-batch tensor. This should not error.
        all_input_ids = torch.cat(all_input_ids_list, dim=0)
        all_attention_mask = (all_input_ids != tokenizer.pad_token_id).long()

    except Exception as e:
        logger.error(f"Error preparing mega-batch for device {device}: {e}", exc_info=True)
        return {}

    original_batch_size = batch['con_prime_con_target_ids'].size(0)

    try:
        amp_enabled = use_amp and device.type == 'cuda'
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(input_ids=all_input_ids, attention_mask=all_attention_mask)
                logits = outputs.logits
    except Exception as e:
        logger.error(f"Model forward pass error on mega-batch (AMP enabled: {amp_enabled}): {e}", exc_info=True)
        return {}

    labels = all_input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    logits_list = torch.chunk(logits, 6, dim=0)
    labels_list = torch.chunk(labels, 6, dim=0)

    # These start indices are now simple integers since the logic is in the dataloader
    target_starts_map = {
        'logp_conT_conP': batch['con_target_start_in_con_prime_context'],
        'logp_inconT_conP': batch['incon_target_start_in_con_prime_context'],
        'logp_conT_inconP': batch['con_target_start_in_incon_prime_context'],
        'logp_inconT_inconP': batch['incon_target_start_in_incon_prime_context'],
        'logp_conT_base': torch.full((original_batch_size,), 1, dtype=torch.long),
        'logp_inconT_base': torch.full((original_batch_size,), 1, dtype=torch.long),
    }

    log_prob_keys = list(target_starts_map.keys())
    log_probs_per_item = [defaultdict(float) for _ in range(original_batch_size)]

    for k, key in enumerate(log_prob_keys):
        logit_tensor, label_tensor = logits_list[k], labels_list[k]
        start_indices = target_starts_map[key].to(device)
        vocab_size = logit_tensor.size(-1)

        for i in range(original_batch_size):
            try:
                target_start_idx = start_indices[i].item()
                if target_start_idx == 0:  # 0 is an invalid start index (must be > BOS)
                    log_probs_per_item[i][key] = float('nan')
                    continue

                item_labels = label_tensor[i, target_start_idx:]
                non_padding_len = (item_labels != -100).sum().item()

                if non_padding_len == 0:
                    log_probs_per_item[i][key] = float('nan')
                    continue

                target_end_idx = target_start_idx + non_padding_len
                logits_for_target = logit_tensor[i, target_start_idx - 1: target_end_idx - 1, :]
                labels_for_target = label_tensor[i, target_start_idx: target_end_idx]

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

    final_aggregated_metrics = {}
    final_raw_item_metrics: Dict[str, List[FinalResultItem]] = defaultdict(list)

    for structure, results_list in all_raw_log_probs.items():
        pe_scores = []
        norm_p_con_target_given_con_prime = []
        norm_p_con_target_given_incon_prime = []
        baseline_prefs_for_con_target = []

        for raw_probs in results_list:
            item_metrics = {}
            logp_ct_cp = raw_probs.get('logp_conT_conP', float('nan'))
            logp_it_cp = raw_probs.get('logp_inconT_conP', float('nan'))
            logp_ct_ip = raw_probs.get('logp_conT_inconP', float('nan'))
            logp_it_ip = raw_probs.get('logp_inconT_inconP', float('nan'))
            logp_ct_base = raw_probs.get('logp_conT_base', float('nan'))
            logp_it_base = raw_probs.get('logp_inconT_base', float('nan'))

            # (Calculations for PE, Normalized Probs, and Baseline remain the same)
            # ...
            if math.isfinite(logp_ct_cp) and math.isfinite(logp_ct_ip):
                pe = logp_ct_cp - logp_ct_ip
                pe_scores.append(pe)
                item_metrics['pe_sinclair'] = pe

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

            p_ct_base = math.exp(logp_ct_base) if math.isfinite(logp_ct_base) else 0
            p_it_base = math.exp(logp_it_base) if math.isfinite(logp_it_base) else 0
            if p_ct_base + p_it_base > 0:
                baseline_pref = p_ct_base / (p_ct_base + p_it_base)
                baseline_prefs_for_con_target.append(baseline_pref)
                item_metrics['baseline_pref_for_conT'] = baseline_pref

            final_raw_item_metrics[structure].append(item_metrics)

        # Aggregate and store all metrics
        final_aggregated_metrics[f'avg_pe_sinclair_{structure}'] = np.mean(pe_scores) if pe_scores else float('nan')
        final_aggregated_metrics[f'std_pe_sinclair_{structure}'] = np.std(pe_scores) if pe_scores else float('nan')

        avg_norm_p_con = np.mean(norm_p_con_target_given_con_prime) if norm_p_con_target_given_con_prime else float(
            'nan')
        avg_norm_p_incon = np.mean(
            norm_p_con_target_given_incon_prime) if norm_p_con_target_given_incon_prime else float('nan')

        final_aggregated_metrics[f'avg_norm_p_conT_given_conP_{structure}'] = avg_norm_p_con
        final_aggregated_metrics[f'avg_norm_p_conT_given_inconP_{structure}'] = avg_norm_p_incon

        # --- NEW: Calculate the direct comparison metric ---
        if math.isfinite(avg_norm_p_con) and math.isfinite(avg_norm_p_incon):
            final_aggregated_metrics[f'priming_effect_normalized_{structure}'] = avg_norm_p_con - avg_norm_p_incon
        else:
            final_aggregated_metrics[f'priming_effect_normalized_{structure}'] = float('nan')
        # --- End of New Code ---

        final_aggregated_metrics[f'avg_baseline_pref_for_conT_{structure}'] = np.mean(
            baseline_prefs_for_con_target) if baseline_prefs_for_con_target else float('nan')
        final_aggregated_metrics[f'count_{structure}'] = len(results_list)

    if original_mode:
        model.train()
    logger.info("Expanded native priming evaluation finished.")

    return final_aggregated_metrics, dict(final_raw_item_metrics)