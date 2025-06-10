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
    Calculates all required log-probabilities for a batch of data.
    """
    try:
        # --- KEY CHANGE: Move all required tensors from the batch to the correct device FIRST ---
        tensor_keys = [
            'congruent_prime_input_ids', 'incongruent_prime_input_ids',
            'congruent_target_input_ids', 'incongruent_target_input_ids'
        ]
        for key in tensor_keys:
            if key in batch and isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
            else:
                raise KeyError(f"Batch from DataLoader is missing required tensor key: {key}")
        # --- End of Change ---

        # Concatenate all inputs for a single mega-batch forward pass
        con_prime_con_target = torch.cat([batch['congruent_prime_input_ids'], batch['congruent_target_input_ids']],
                                         dim=1)
        con_prime_incon_target = torch.cat([batch['congruent_prime_input_ids'], batch['incongruent_target_input_ids']],
                                           dim=1)
        incon_prime_con_target = torch.cat([batch['incongruent_prime_input_ids'], batch['congruent_target_input_ids']],
                                           dim=1)
        incon_prime_incon_target = torch.cat(
            [batch['incongruent_prime_input_ids'], batch['incongruent_target_input_ids']], dim=1)

        # Now that batch tensors are on the correct device, this will also be on the correct device
        bos_tensor = torch.tensor([[tokenizer.bos_token_id]], device=device).expand(
            batch['congruent_target_input_ids'].size(0), -1)
        base_con_target = torch.cat([bos_tensor, batch['congruent_target_input_ids']], dim=1)
        base_incon_target = torch.cat([bos_tensor, batch['incongruent_target_input_ids']], dim=1)

        all_input_ids = torch.cat([
            con_prime_con_target, con_prime_incon_target,
            incon_prime_con_target, incon_prime_incon_target,
            base_con_target, base_incon_target
        ], dim=0)

        all_attention_mask = (all_input_ids != tokenizer.pad_token_id).long()

    except (KeyError, AttributeError) as e:
        logger.error(f"Batch missing a required key or attribute: {e}. Check your dataloader.")
        return {}
    except Exception as e:
        logger.error(f"Error preparing mega-batch for device {device}: {e}", exc_info=True)
        return {}

    original_batch_size = batch['congruent_prime_input_ids'].size(0)

    # (The rest of the function for forward pass and log-prob calculation remains the same)

    # ... (rest of function is identical to the one I sent previously) ...
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

    target_starts = [
        batch['con_target_start_in_con_prime_context'], batch['incon_target_start_in_con_prime_context'],
        batch['con_target_start_in_incon_prime_context'], batch['incon_target_start_in_incon_prime_context'],
        torch.full((original_batch_size,), 1, dtype=torch.long, device=device),  # Use full tensor
        torch.full((original_batch_size,), 1, dtype=torch.long, device=device)  # Use full tensor
    ]

    log_prob_keys = [
        'logp_conT_conP', 'logp_inconT_conP',
        'logp_conT_inconP', 'logp_inconT_inconP',
        'logp_conT_base', 'logp_inconT_base'
    ]

    log_probs_per_item = [defaultdict(float) for _ in range(original_batch_size)]

    for k, (key, logit_tensor, label_tensor, start_indices) in enumerate(
            zip(log_prob_keys, logits_list, labels_list, target_starts)):
        vocab_size = logit_tensor.size(-1)
        # Move start_indices to the correct device if it's not already
        start_indices = start_indices.to(device)
        for i in range(original_batch_size):
            try:
                target_start_idx = start_indices[i].item()
                item_labels = label_tensor[i, target_start_idx:]
                non_padding_len = (item_labels != -100).sum().item()

                if non_padding_len == 0:
                    logger.warning(f"Item {i} has zero-length target for '{key}'. Skipping.")
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


# (run_native_priming_eval function remains the same as the previous version)
def run_native_priming_eval(
        model: PreTrainedModel, priming_dataloader: DataLoader, device: torch.device,
        tokenizer: PreTrainedTokenizer,
        use_amp: bool = False
) -> EvalResults:
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

        final_aggregated_metrics[f'avg_pe_sinclair_{structure}'] = np.mean(pe_scores) if pe_scores else float('nan')
        final_aggregated_metrics[f'std_pe_sinclair_{structure}'] = np.std(pe_scores) if pe_scores else float('nan')
        final_aggregated_metrics[f'avg_norm_p_conT_given_conP_{structure}'] = np.mean(
            norm_p_con_target_given_con_prime) if norm_p_con_target_given_con_prime else float('nan')
        final_aggregated_metrics[f'avg_norm_p_conT_given_inconP_{structure}'] = np.mean(
            norm_p_con_target_given_incon_prime) if norm_p_con_target_given_incon_prime else float('nan')
        final_aggregated_metrics[f'avg_baseline_pref_for_conT_{structure}'] = np.mean(
            baseline_prefs_for_con_target) if baseline_prefs_for_con_target else float('nan')
        final_aggregated_metrics[f'count_{structure}'] = len(results_list)

    if original_mode:
        model.train()
    logger.info("Expanded native priming evaluation finished.")

    return final_aggregated_metrics, dict(final_raw_item_metrics)