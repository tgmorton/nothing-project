# src/priming_evaluation/evaluator.py

import logging
import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Define the structure for individual item results
ResultItem = Dict[
    str, float]  # {'pe': float, 'logp_con': float, 'logp_incon': float, 'logp_baseline': float, 'logp_con_random_baseline': float, 'logp_incon_random_baseline': float}
# Define the return type for the batch calculation function
BatchResults = Dict[str, List[ResultItem]]
# Define the return type for the main eval function
EvalResults = Tuple[Dict[str, float], Dict[str, List[ResultItem]]]


def calculate_priming_effect(
        model: PreTrainedModel,
        batch: Dict[str, Any],
        device: torch.device,
        torch_rng: torch.Generator,  # For reproducible shuffling of con/incon primes
        use_amp: bool = False
) -> BatchResults:
    """
    Calculates Priming Effect (PE), congruent, incongruent, original baseline,
    and randomized-prime baseline log probabilities.
    PE = log P(Target | CongruentPrime) - log P(Target | IncongruentPrime)
    logp_baseline = log P(Target | UserDefinedBaselinePrime)
    logp_con_random_baseline = log P(Target | RandomizedCongruentPrime)
    logp_incon_random_baseline = log P(Target | RandomizedIncongruentPrime)
    """
    try:
        congruent_input_ids = batch['congruent_input_ids'].to(device)
        congruent_attention_mask = batch['congruent_attention_mask'].to(device)
        incongruent_input_ids = batch['incongruent_input_ids'].to(device)
        incongruent_attention_mask = batch['incongruent_attention_mask'].to(device)

        # Original baseline inputs
        baseline_input_ids = batch['baseline_input_ids'].to(device)
        baseline_attention_mask = batch['baseline_attention_mask'].to(device)

        labels = batch['labels'].to(device)

        target_starts_con = batch['target_start_congruent']
        target_ends_con = batch['target_end_congruent']
        target_starts_incon = batch['target_start_incongruent']
        target_ends_incon = batch['target_end_incongruent']
        target_starts_base = batch['target_start_baseline']  # Original baseline
        target_ends_base = batch['target_end_baseline']  # Original baseline
        target_structures = batch['target_structure']

        # Convert indices to tensors on device if they aren't already
        if isinstance(target_starts_con, list): target_starts_con = torch.tensor(target_starts_con, dtype=torch.long,
                                                                                 device=device)
        if isinstance(target_ends_con, list): target_ends_con = torch.tensor(target_ends_con, dtype=torch.long,
                                                                             device=device)
        if isinstance(target_starts_incon, list): target_starts_incon = torch.tensor(target_starts_incon,
                                                                                     dtype=torch.long, device=device)
        if isinstance(target_ends_incon, list): target_ends_incon = torch.tensor(target_ends_incon, dtype=torch.long,
                                                                                 device=device)
        if isinstance(target_starts_base, list): target_starts_base = torch.tensor(target_starts_base, dtype=torch.long,
                                                                                   device=device)
        if isinstance(target_ends_base, list): target_ends_base = torch.tensor(target_ends_base, dtype=torch.long,
                                                                               device=device)

    except KeyError as e:
        logger.error(f"Batch missing key: {e}. Cannot calculate all metrics.")
        return {}
    except Exception as e:
        logger.error(f"Error moving/preparing batch for device {device}: {e}")
        return {}

    batch_size = congruent_input_ids.size(0)
    batch_results: BatchResults = defaultdict(list)
    nan_result: ResultItem = {
        'pe': float('nan'),
        'logp_con': float('nan'),
        'logp_incon': float('nan'),
        'logp_baseline': float('nan'),  # Original baseline
        'logp_con_random_baseline': float('nan'),
        'logp_incon_random_baseline': float('nan')
    }

    # Prepare inputs for randomized baselines (derived from con/incon)
    con_random_baseline_input_ids = torch.empty_like(congruent_input_ids)
    incon_random_baseline_input_ids = torch.empty_like(incongruent_input_ids)

    for i in range(batch_size):
        # Congruent Random Baseline
        prime_len_con = target_starts_con[i].item() - 1
        if prime_len_con < 0:
            con_random_baseline_input_ids[i] = congruent_input_ids[i]
        elif prime_len_con == 0:
            con_random_baseline_input_ids[i] = congruent_input_ids[i]
        else:
            prime_tokens_con = congruent_input_ids[i, :prime_len_con]
            target_and_suffix_con = congruent_input_ids[i, prime_len_con:]
            perm_con = torch.randperm(prime_tokens_con.size(0), generator=torch_rng, device=device)
            shuffled_prime_con = prime_tokens_con[perm_con]
            con_random_baseline_input_ids[i] = torch.cat((shuffled_prime_con, target_and_suffix_con))

        # Incongruent Random Baseline
        prime_len_incon = target_starts_incon[i].item() - 1
        if prime_len_incon < 0:
            incon_random_baseline_input_ids[i] = incongruent_input_ids[i]
        elif prime_len_incon == 0:
            incon_random_baseline_input_ids[i] = incongruent_input_ids[i]
        else:
            prime_tokens_incon = incongruent_input_ids[i, :prime_len_incon]
            target_and_suffix_incon = incongruent_input_ids[i, prime_len_incon:]
            perm_incon = torch.randperm(prime_tokens_incon.size(0), generator=torch_rng, device=device)
            shuffled_prime_incon = prime_tokens_incon[perm_incon]
            incon_random_baseline_input_ids[i] = torch.cat((shuffled_prime_incon, target_and_suffix_incon))

    con_random_baseline_attention_mask = congruent_attention_mask  # Reuses original mask
    incon_random_baseline_attention_mask = incongruent_attention_mask  # Reuses original mask

    try:
        amp_enabled = use_amp and device.type == 'cuda'
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs_con = model(input_ids=congruent_input_ids, attention_mask=congruent_attention_mask)
                logits_con = outputs_con.logits
                outputs_incon = model(input_ids=incongruent_input_ids, attention_mask=incongruent_attention_mask)
                logits_incon = outputs_incon.logits
                outputs_base = model(input_ids=baseline_input_ids,
                                     attention_mask=baseline_attention_mask)  # Original baseline
                logits_base = outputs_base.logits

                outputs_con_rand = model(input_ids=con_random_baseline_input_ids,
                                         attention_mask=con_random_baseline_attention_mask)
                logits_con_rand = outputs_con_rand.logits
                outputs_incon_rand = model(input_ids=incon_random_baseline_input_ids,
                                           attention_mask=incon_random_baseline_attention_mask)
                logits_incon_rand = outputs_incon_rand.logits

    except Exception as e:
        logger.error(f"Model forward pass error (AMP enabled: {amp_enabled}): {e}", exc_info=True)
        for i in range(batch_size):
            target_structure = target_structures[i] if i < len(target_structures) else "unknown_structure_fwd_fail"
            batch_results[target_structure].append(nan_result)
        return dict(batch_results)

    for i in range(batch_size):
        target_structure = target_structures[i]
        log_prob_con_val = float('nan')
        log_prob_incon_val = float('nan')
        log_prob_baseline_val = float('nan')  # Original baseline
        log_prob_con_rand_val = float('nan')
        log_prob_incon_rand_val = float('nan')
        priming_effect = float('nan')

        try:
            start_con = target_starts_con[i].item()
            end_con = target_ends_con[i].item()
            start_incon = target_starts_incon[i].item()
            end_incon = target_ends_incon[i].item()
            start_base = target_starts_base[i].item()  # Original baseline
            end_base = target_ends_base[i].item()  # Original baseline

            len_logits_con = logits_con.shape[1]
            len_logits_incon = logits_incon.shape[1]
            len_logits_base = logits_base.shape[1]  # Original baseline
            len_logits_con_rand = logits_con_rand.shape[1]
            len_logits_incon_rand = logits_incon_rand.shape[1]
            label_len = labels.shape[1]

            valid_con_indices = (0 <= start_con - 1 < end_con - 1 < len_logits_con and start_con < end_con <= label_len)
            valid_incon_indices = (
                        0 <= start_incon - 1 < end_incon - 1 < len_logits_incon and start_incon < end_incon <= label_len)
            valid_base_indices = (
                        0 <= start_base - 1 < end_base - 1 < len_logits_base and start_base < end_base <= label_len)  # Original baseline
            valid_con_rand_indices = (
                        0 <= start_con - 1 < end_con - 1 < len_logits_con_rand and start_con < end_con <= label_len)
            valid_incon_rand_indices = (
                        0 <= start_incon - 1 < end_incon - 1 < len_logits_incon_rand and start_incon < end_incon <= label_len)

            if not (
                    valid_con_indices and valid_incon_indices and valid_base_indices and valid_con_rand_indices and valid_incon_rand_indices):
                logger.warning(
                    f"Index out of bounds for item {i}, target {target_structure}. "
                    f"Con: ({start_con},{end_con}) vs LogitLen={len_logits_con}. "
                    f"Incon: ({start_incon},{end_incon}) vs LogitLen={len_logits_incon}. "
                    f"Base: ({start_base},{end_base}) vs LogitLen={len_logits_base}. "  # Original baseline info
                    f"ConRand: LogitLen={len_logits_con_rand}. InconRand: LogitLen={len_logits_incon_rand}. LabelLen={label_len}. Skipping."
                )
                batch_results[target_structure].append(nan_result)
                continue

            # Logits slices for target prediction (logit at t-1 predicts token at t)
            logits_for_target_con = logits_con[i, start_con - 1: end_con - 1, :]
            logits_for_target_incon = logits_incon[i, start_incon - 1: end_incon - 1, :]
            logits_for_target_base = logits_base[i, start_base - 1: end_base - 1, :]  # Original baseline
            logits_for_target_con_rand = logits_con_rand[i, start_con - 1: end_con - 1, :]
            logits_for_target_incon_rand = logits_incon_rand[i, start_incon - 1: end_incon - 1, :]

            # Label slices (token at t)
            target_labels_con_segment = labels[i, start_con: end_con]
            target_labels_incon_segment = labels[i, start_incon: end_incon]
            target_labels_base_segment = labels[i, start_base: end_base]  # Original baseline

            vocab_size = logits_con.size(-1)  # Assume vocab size is consistent

            # --- Congruent ---
            if logits_for_target_con.shape[0] == target_labels_con_segment.shape[0] and target_labels_con_segment.shape[
                0] > 0:
                log_prob_con_tensor = -F.cross_entropy(logits_for_target_con.reshape(-1, vocab_size),
                                                       target_labels_con_segment.reshape(-1), ignore_index=-100,
                                                       reduction='sum')
                log_prob_con_val = log_prob_con_tensor.item()
            else:
                logger.warning(
                    f"Logit/Label mismatch for CON: item {i}, target {target_structure}, LogitShape0 {logits_for_target_con.shape[0]}, LabelShape0 {target_labels_con_segment.shape[0]}")

            # --- Incongruent ---
            if logits_for_target_incon.shape[0] == target_labels_incon_segment.shape[0] and \
                    target_labels_incon_segment.shape[0] > 0:
                log_prob_incon_tensor = -F.cross_entropy(logits_for_target_incon.reshape(-1, vocab_size),
                                                         target_labels_incon_segment.reshape(-1), ignore_index=-100,
                                                         reduction='sum')
                log_prob_incon_val = log_prob_incon_tensor.item()
            else:
                logger.warning(
                    f"Logit/Label mismatch for INCON: item {i}, target {target_structure}, LogitShape0 {logits_for_target_incon.shape[0]}, LabelShape0 {target_labels_incon_segment.shape[0]}")

            # --- Original Baseline ---
            if logits_for_target_base.shape[0] == target_labels_base_segment.shape[0] and \
                    target_labels_base_segment.shape[0] > 0:
                log_prob_baseline_tensor = -F.cross_entropy(logits_for_target_base.reshape(-1, vocab_size),
                                                            target_labels_base_segment.reshape(-1), ignore_index=-100,
                                                            reduction='sum')
                log_prob_baseline_val = log_prob_baseline_tensor.item()
            else:
                logger.warning(
                    f"Logit/Label mismatch for BASELINE: item {i}, target {target_structure}, LogitShape0 {logits_for_target_base.shape[0]}, LabelShape0 {target_labels_base_segment.shape[0]}")

            # --- Congruent Random Baseline ---
            # Uses target_labels_con_segment as the target is the same as for the congruent condition
            if logits_for_target_con_rand.shape[0] == target_labels_con_segment.shape[0] and \
                    target_labels_con_segment.shape[0] > 0:
                log_prob_con_rand_tensor = -F.cross_entropy(logits_for_target_con_rand.reshape(-1, vocab_size),
                                                            target_labels_con_segment.reshape(-1), ignore_index=-100,
                                                            reduction='sum')
                log_prob_con_rand_val = log_prob_con_rand_tensor.item()
            else:
                logger.warning(
                    f"Logit/Label mismatch for CON_RAND: item {i}, target {target_structure}, LogitShape0 {logits_for_target_con_rand.shape[0]}, LabelShape0 {target_labels_con_segment.shape[0]}")

            # --- Incongruent Random Baseline ---
            # Uses target_labels_incon_segment as the target is the same as for the incongruent condition
            if logits_for_target_incon_rand.shape[0] == target_labels_incon_segment.shape[0] and \
                    target_labels_incon_segment.shape[0] > 0:
                log_prob_incon_rand_tensor = -F.cross_entropy(logits_for_target_incon_rand.reshape(-1, vocab_size),
                                                              target_labels_incon_segment.reshape(-1),
                                                              ignore_index=-100, reduction='sum')
                log_prob_incon_rand_val = log_prob_incon_rand_tensor.item()
            else:
                logger.warning(
                    f"Logit/Label mismatch for INCON_RAND: item {i}, target {target_structure}, LogitShape0 {logits_for_target_incon_rand.shape[0]}, LabelShape0 {target_labels_incon_segment.shape[0]}")

            if math.isfinite(log_prob_con_val) and math.isfinite(log_prob_incon_val):
                priming_effect = log_prob_con_val - log_prob_incon_val
            # PE is NaN if con or incon is NaN. Other metrics can still be valid.

            current_result: ResultItem = {
                'pe': priming_effect,
                'logp_con': log_prob_con_val,
                'logp_incon': log_prob_incon_val,
                'logp_baseline': log_prob_baseline_val,  # Original baseline
                'logp_con_random_baseline': log_prob_con_rand_val,
                'logp_incon_random_baseline': log_prob_incon_rand_val
            }

            if not all(math.isfinite(v) for v in current_result.values()):
                logger.debug(f"Item {i}, target {target_structure} has some non-finite values: "
                             f"PE={current_result['pe']:.4f}, LogP_con={current_result['logp_con']:.4f}, "
                             f"LogP_incon={current_result['logp_incon']:.4f}, LogP_base={current_result['logp_baseline']:.4f}, "
                             f"LogP_con_rand={current_result['logp_con_random_baseline']:.4f}, "
                             f"LogP_incon_rand={current_result['logp_incon_random_baseline']:.4f}. Storing (NaNs where applicable).")
                final_item_result = {k: (v if math.isfinite(v) else float('nan')) for k, v in current_result.items()}
                batch_results[target_structure].append(final_item_result)
            else:
                batch_results[target_structure].append(current_result)

        except IndexError as e:
            logger.error(f"IndexError during metric calculation for item {i}, target {target_structure}. Err:{e}",
                         exc_info=True)
            batch_results[target_structure].append(nan_result)
        except Exception as e:
            logger.error(f"Unexpected error processing metrics for item {i}, target {target_structure}: {e}",
                         exc_info=True)
            batch_results[target_structure].append(nan_result)

    return dict(batch_results)


def run_native_priming_eval(
        model: PreTrainedModel, priming_dataloader: DataLoader, device: torch.device,
        tokenizer: PreTrainedTokenizer,
        random_seed: Optional[int] = None,
        use_amp: bool = False
) -> EvalResults:
    logger.info(
        "Starting native priming evaluation (PE, LogPs for con, incon, baseline, con_random_baseline, incon_random_baseline)...")
    original_mode = model.training
    model.eval()

    torch_rng = torch.Generator(device=device)
    if random_seed is not None:
        torch.manual_seed(random_seed)
        torch_rng.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        logger.info(f"Random seed set to {random_seed} for evaluation.")
    else:
        logger.warning(
            "No random_seed provided for priming evaluation. Randomization of con/incon primes will not be reproducible.")

    all_results_raw: Dict[str, List[ResultItem]] = defaultdict(list)

    progress_bar = tqdm(priming_dataloader, desc="Priming Eval", leave=False)
    for batch_idx, batch in enumerate(progress_bar):
        if not batch:
            logger.warning(f"Skipping empty batch {batch_idx} from collate function.")
            continue
        try:
            batch_metrics_raw: BatchResults = calculate_priming_effect(model, batch, device, torch_rng, use_amp=use_amp)
            for target_structure, result_list in batch_metrics_raw.items():
                all_results_raw[target_structure].extend(result_list)
        except Exception as e:
            logger.error(f"Error processing priming batch {batch_idx}: {e}", exc_info=True)

    final_aggregated_metrics: Dict[str, float] = {}
    logger.info("Calculating final priming metric aggregates:")

    for target_structure, structure_results_list in all_results_raw.items():
        def get_finite_values(key):
            return [r[key] for r in structure_results_list if
                    isinstance(r, dict) and key in r and math.isfinite(r[key])]

        finite_pe_values = get_finite_values('pe')
        finite_logp_con_values = get_finite_values('logp_con')
        finite_logp_incon_values = get_finite_values('logp_incon')
        finite_logp_baseline_values = get_finite_values('logp_baseline')  # Original baseline
        finite_logp_con_rand_values = get_finite_values('logp_con_random_baseline')
        finite_logp_incon_rand_values = get_finite_values('logp_incon_random_baseline')

        total_items = len(structure_results_list)
        logger.info(f"  Target '{target_structure}' (Total Items Processed: {total_items}):")

        def aggregate_and_log(metric_name_display: str, metric_key_base: str, values: List[float]):
            if values:
                count = len(values)
                mean_val, std_val = np.mean(values), np.std(values)
                sem_val = std_val / np.sqrt(count) if count > 0 else float('nan')
                final_aggregated_metrics[f"avg_{metric_key_base}_{target_structure}"] = mean_val
                final_aggregated_metrics[f"std_{metric_key_base}_{target_structure}"] = std_val
                final_aggregated_metrics[f"sem_{metric_key_base}_{target_structure}"] = sem_val
                final_aggregated_metrics[f"count_{metric_key_base}_{target_structure}"] = count
                logger.info(
                    f"    {metric_name_display:<28}: Avg = {mean_val:.4f} (Std = {std_val:.4f}, SEM = {sem_val:.4f}, N = {count})")
            else:
                for metric_suffix in ["avg", "std", "sem", "count"]:
                    final_aggregated_metrics[f"{metric_suffix}_{metric_key_base}_{target_structure}"] = float(
                        'nan') if metric_suffix != "count" else 0
                logger.warning(f"    {metric_name_display:<28}: No finite values found.")

        aggregate_and_log("PE", "PE", finite_pe_values)
        aggregate_and_log("LogP_con", "LogP_con", finite_logp_con_values)
        aggregate_and_log("LogP_incon", "LogP_incon", finite_logp_incon_values)
        aggregate_and_log("LogP_baseline", "LogP_baseline", finite_logp_baseline_values)  # Original baseline
        aggregate_and_log("LogP_con_random_baseline", "LogP_con_random_baseline", finite_logp_con_rand_values)
        aggregate_and_log("LogP_incon_random_baseline", "LogP_incon_random_baseline", finite_logp_incon_rand_values)

    if original_mode:
        model.train()
    logger.info("Native priming evaluation finished.")
    return final_aggregated_metrics, dict(all_results_raw)