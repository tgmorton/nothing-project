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

ResultItem = Dict[str, float]
BatchResults = Dict[str, List[ResultItem]]
EvalResults = Tuple[Dict[str, float], Dict[str, List[ResultItem]]]


def calculate_priming_effect(
        model: PreTrainedModel,
        batch: Dict[str, Any],
        device: torch.device,
        torch_rng: torch.Generator,
        tokenizer: PreTrainedTokenizer,  # Added for logging
        use_amp: bool = False,
        is_first_batch_for_corpus: bool = False,  # For targeted verbose logging
        max_verbose_items_per_batch: int = 2  # For targeted verbose logging
) -> BatchResults:
    try:
        congruent_input_ids = batch['congruent_input_ids'].to(device)
        congruent_attention_mask = batch['congruent_attention_mask'].to(device)
        incongruent_input_ids = batch['incongruent_input_ids'].to(device)
        incongruent_attention_mask = batch['incongruent_attention_mask'].to(device)
        baseline_input_ids = batch['baseline_input_ids'].to(device)
        baseline_attention_mask = batch['baseline_attention_mask'].to(device)
        labels = batch['labels'].to(device)  # This label tensor is based on congruent path

        target_starts_con = batch['target_start_congruent'].to(device)  # Ensure on device
        target_ends_con = batch['target_end_congruent'].to(device)
        target_starts_incon = batch['target_start_incongruent'].to(device)
        target_ends_incon = batch['target_end_incongruent'].to(device)
        target_starts_base = batch['target_start_baseline'].to(device)
        target_ends_base = batch['target_end_baseline'].to(device)
        target_structures = batch['target_structure']

        # These might not always be present if collate_fn changes or for older data
        source_csvs = batch.get('source_csv', ["N/A"] * congruent_input_ids.size(0))
        csv_rows = batch.get('csv_row', [-1] * congruent_input_ids.size(0))

    except KeyError as e:
        logger.error(f"Batch missing key: {e}. Cannot calculate all metrics.")
        return {}
    except Exception as e:
        logger.error(f"Error moving/preparing batch for device {device}: {e}", exc_info=True)
        return {}

    batch_size = congruent_input_ids.size(0)
    batch_results: BatchResults = defaultdict(list)
    nan_result: ResultItem = {
        'pe': float('nan'), 'logp_con': float('nan'), 'logp_incon': float('nan'),
        'logp_baseline': float('nan'), 'logp_con_random_baseline': float('nan'),
        'logp_incon_random_baseline': float('nan')
    }

    con_random_baseline_input_ids = torch.empty_like(congruent_input_ids)
    incon_random_baseline_input_ids = torch.empty_like(incongruent_input_ids)

    for i in range(batch_size):
        prime_len_con = target_starts_con[i].item() - 1  # BOS is at index 0, prime starts at 1
        if prime_len_con <= 0:  # No prime tokens (only BOS) or invalid
            con_random_baseline_input_ids[i] = congruent_input_ids[i]
        else:
            bos_token_id_tensor = congruent_input_ids[i, 0:1]  # Keep BOS
            prime_tokens_con = congruent_input_ids[i, 1:prime_len_con + 1]  # Prime tokens after BOS
            target_and_suffix_con = congruent_input_ids[i, prime_len_con + 1:]
            perm_con = torch.randperm(prime_tokens_con.size(0), generator=torch_rng, device=device)
            shuffled_prime_con = prime_tokens_con[perm_con]
            con_random_baseline_input_ids[i] = torch.cat(
                (bos_token_id_tensor, shuffled_prime_con, target_and_suffix_con))

        prime_len_incon = target_starts_incon[i].item() - 1
        if prime_len_incon <= 0:
            incon_random_baseline_input_ids[i] = incongruent_input_ids[i]
        else:
            bos_token_id_tensor_incon = incongruent_input_ids[i, 0:1]
            prime_tokens_incon = incongruent_input_ids[i, 1:prime_len_incon + 1]
            target_and_suffix_incon = incongruent_input_ids[i, prime_len_incon + 1:]
            perm_incon = torch.randperm(prime_tokens_incon.size(0), generator=torch_rng, device=device)
            shuffled_prime_incon = prime_tokens_incon[perm_incon]
            incon_random_baseline_input_ids[i] = torch.cat(
                (bos_token_id_tensor_incon, shuffled_prime_incon, target_and_suffix_incon))

    con_random_baseline_attention_mask = congruent_attention_mask
    incon_random_baseline_attention_mask = incongruent_attention_mask

    try:
        amp_enabled = use_amp and device.type == 'cuda'
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs_con = model(input_ids=congruent_input_ids, attention_mask=congruent_attention_mask)
                logits_con = outputs_con.logits
                outputs_incon = model(input_ids=incongruent_input_ids, attention_mask=incongruent_attention_mask)
                logits_incon = outputs_incon.logits
                outputs_base = model(input_ids=baseline_input_ids, attention_mask=baseline_attention_mask)
                logits_base = outputs_base.logits
                outputs_con_rand = model(input_ids=con_random_baseline_input_ids,
                                         attention_mask=con_random_baseline_attention_mask)
                logits_con_rand = outputs_con_rand.logits
                outputs_incon_rand = model(input_ids=incon_random_baseline_input_ids,
                                           attention_mask=incon_random_baseline_attention_mask)
                logits_incon_rand = outputs_incon_rand.logits
    except Exception as e:
        logger.error(f"Model forward pass error (AMP enabled: {amp_enabled}): {e}", exc_info=True)
        for i_err in range(batch_size):
            err_target_structure = target_structures[i_err] if i_err < len(
                target_structures) else "unknown_structure_fwd_fail"
            batch_results[err_target_structure].append(nan_result)
        return dict(batch_results)

    for i in range(batch_size):
        target_structure = target_structures[i]
        source_csv_item = source_csvs[i] if i < len(source_csvs) else "N/A"
        csv_row_item = csv_rows[i] if i < len(csv_rows) else -1

        do_verbose_logging = logger.isEnabledFor(
            logging.DEBUG) and is_first_batch_for_corpus and i < max_verbose_items_per_batch

        if do_verbose_logging:
            logger.debug(
                f"\n--- evaluator.py ITEM LOG (BatchIdxItem:{i}, CSV:{source_csv_item}, Row:{csv_row_item}, Struct:{target_structure}) ---")

        log_prob_con_val, log_prob_incon_val, log_prob_baseline_val = float('nan'), float('nan'), float('nan')
        log_prob_con_rand_val, log_prob_incon_rand_val = float('nan'), float('nan')
        priming_effect = float('nan')

        try:
            s_con, e_con = target_starts_con[i].item(), target_ends_con[i].item()
            s_incon, e_incon = target_starts_incon[i].item(), target_ends_incon[i].item()
            s_base, e_base = target_starts_base[i].item(), target_ends_base[i].item()

            # For random baselines, the target segment indices are relative to *their own* sequences,
            # but should correspond to the *same target tokens* as their non-random counterparts.
            # So, con_rand uses s_con, e_con but on con_random_baseline_input_ids.
            # And incon_rand uses s_incon, e_incon but on incon_random_baseline_input_ids.
            # The item_effective_target_len is implicitly e_con - s_con.

            if do_verbose_logging:
                logger.debug(
                    f"  Indices CON: s={s_con}, e={e_con} (len={e_con - s_con}). LogitSeqLen={logits_con.shape[1]}. InputIDsLen={congruent_input_ids.shape[1]}")
                logger.debug(
                    f"  Indices INCON: s={s_incon}, e={e_incon} (len={e_incon - s_incon}). LogitSeqLen={logits_incon.shape[1]}. InputIDsLen={incongruent_input_ids.shape[1]}")
                logger.debug(
                    f"  Indices BASE: s={s_base}, e={e_base} (len={e_base - s_base}). LogitSeqLen={logits_base.shape[1]}. InputIDsLen={baseline_input_ids.shape[1]}")
                logger.debug(f"  Labels tensor shape for item: {labels[i].shape}")

            # Validation: Ensure slice indices are valid for logits AND that target segment length is positive
            # Logits are for predicting token AT an index, so we need logits up to `end_idx - 1`.
            # Labels are tokens AT an index.
            valid_con = (0 <= s_con - 1 < e_con - 1 < logits_con.shape[1]) and (s_con < e_con)
            valid_incon = (0 <= s_incon - 1 < e_incon - 1 < logits_incon.shape[1]) and (s_incon < e_incon)
            valid_base = (0 <= s_base - 1 < e_base - 1 < logits_base.shape[1]) and (s_base < e_base)
            valid_con_rand = (0 <= s_con - 1 < e_con - 1 < logits_con_rand.shape[1]) and (s_con < e_con)
            valid_incon_rand = (0 <= s_incon - 1 < e_incon - 1 < logits_incon_rand.shape[1]) and (s_incon < e_incon)

            if not (valid_con and valid_incon and valid_base and valid_con_rand and valid_incon_rand):
                if do_verbose_logging or logger.isEnabledFor(logging.WARNING):  # Log always if warning, or if verbose
                    logger.warning(
                        f"Idx validation fail item {i}, {target_structure}. C({valid_con}),I({valid_incon}),B({valid_base}),CR({valid_con_rand}),IR({valid_incon_rand})"
                        f"  Con: ({s_con},{e_con}) vs LogitL={logits_con.shape[1]}. Incon: ({s_incon},{e_incon}) vs LogitL={logits_incon.shape[1]}."
                        f"  Base: ({s_base},{e_base}) vs LogitL={logits_base.shape[1]}."
                    )
                batch_results[target_structure].append(nan_result)
                continue

            logits_target_con = logits_con[i, s_con - 1:e_con - 1, :]
            labels_target_con = congruent_input_ids[i, s_con:e_con]  # Labels are the actual tokens in that segment

            logits_target_incon = logits_incon[i, s_incon - 1:e_incon - 1, :]
            labels_target_incon = incongruent_input_ids[i, s_incon:e_incon]

            logits_target_base = logits_base[i, s_base - 1:e_base - 1, :]
            labels_target_base = baseline_input_ids[i, s_base:e_base]

            logits_target_con_rand = logits_con_rand[i, s_con - 1:e_con - 1, :]
            labels_target_con_rand = con_random_baseline_input_ids[i, s_con:e_con]  # Target tokens from this sequence

            logits_target_incon_rand = logits_incon_rand[i, s_incon - 1:e_incon - 1, :]
            labels_target_incon_rand = incon_random_baseline_input_ids[i, s_incon:e_incon]

            vocab_size = logits_con.size(-1)

            def get_logp(logits_slice, labels_slice, cond_name_log):
                val = float('nan')
                if logits_slice.shape[0] == labels_slice.shape[0] and labels_slice.shape[0] > 0:
                    if do_verbose_logging:
                        actual_ids = labels_slice[labels_slice != -100].tolist()  # Should not be -100 here
                        actual_tokens = tokenizer.convert_ids_to_tokens(actual_ids) if actual_ids else []
                        logger.debug(
                            f"    {cond_name_log}: Scoring {len(actual_ids)} tokens: {actual_tokens} (IDs: {actual_ids})")
                        logger.debug(
                            f"    {cond_name_log}: Logit slice shape: {logits_slice.shape}, Label slice shape: {labels_slice.shape}")

                    # Ensure labels_slice doesn't have pad_token_id if that's also ignore_index
                    # F.cross_entropy handles ignore_index=-100 internally if labels are constructed that way.
                    # Here, labels_slice are actual token IDs from input_ids.
                    logp_tensor = -F.cross_entropy(
                        logits_slice.reshape(-1, vocab_size),
                        labels_slice.reshape(-1),
                        reduction='sum'
                        # No ignore_index needed if labels_slice contains actual target tokens
                    )
                    val = logp_tensor.item()
                elif do_verbose_logging or logger.isEnabledFor(logging.WARNING):
                    logger.warning(
                        f"Logit/Label len mismatch for {cond_name_log}: item {i}, {target_structure}, LogitShape0 {logits_slice.shape[0]}, LabelShape0 {labels_slice.shape[0]}"
                    )
                return val

            log_prob_con_val = get_logp(logits_target_con, labels_target_con, "CON")
            log_prob_incon_val = get_logp(logits_target_incon, labels_target_incon, "INCON")
            log_prob_baseline_val = get_logp(logits_target_base, labels_target_base, "BASE")
            log_prob_con_rand_val = get_logp(logits_target_con_rand, labels_target_con_rand, "CON_RAND")
            log_prob_incon_rand_val = get_logp(logits_target_incon_rand, labels_target_incon_rand, "INCON_RAND")

            if do_verbose_logging:
                logger.debug(
                    f"    LogP Vals: Con={log_prob_con_val:.4f}, Incon={log_prob_incon_val:.4f}, Base={log_prob_baseline_val:.4f}, CRand={log_prob_con_rand_val:.4f}, IRand={log_prob_incon_rand_val:.4f}")

            if math.isfinite(log_prob_con_val) and math.isfinite(log_prob_incon_val):
                priming_effect = log_prob_con_val - log_prob_incon_val

            current_result: ResultItem = {
                'pe': priming_effect, 'logp_con': log_prob_con_val, 'logp_incon': log_prob_incon_val,
                'logp_baseline': log_prob_baseline_val, 'logp_con_random_baseline': log_prob_con_rand_val,
                'logp_incon_random_baseline': log_prob_incon_rand_val
            }
            batch_results[target_structure].append(
                {k: (v if math.isfinite(v) else float('nan')) for k, v in current_result.items()})

        except IndexError as e:  # ... (error handling as before)
            logger.error(f"IndexError for item {i}, {target_structure}. Err:{e}", exc_info=True)
            batch_results[target_structure].append(nan_result)
        except Exception as e:  # ... (error handling as before)
            logger.error(f"Unexpected error for item {i}, {target_structure}: {e}", exc_info=True)
            batch_results[target_structure].append(nan_result)

    return dict(batch_results)


def run_native_priming_eval(
        model: PreTrainedModel, priming_dataloader: DataLoader, device: torch.device,
        tokenizer: PreTrainedTokenizer,  # Pass tokenizer
        random_seed: Optional[int] = None,
        use_amp: bool = False
) -> EvalResults:
    logger.info(
        "Starting native priming evaluation (PE, LogPs for con, incon, baseline, con_random_baseline, incon_random_baseline)...")
    original_mode = model.training
    model.eval()

    torch_rng = torch.Generator(device=device)
    if random_seed is not None:
        torch.manual_seed(random_seed)  # Seed CPU for reproducibility if ops are on CPU first
        torch_rng.manual_seed(random_seed)
        # np.random.seed(random_seed) # Not directly used by torch_rng, but good practice
        # random.seed(random_seed) # Python's random, not directly used by torch_rng
        logger.info(f"Torch RNG seed set to {random_seed} for priming evaluation's prime shuffling.")
    else:
        logger.warning(
            "No random_seed provided for priming evaluation. Randomization of primes for random baselines will not be reproducible across runs.")

    all_results_raw: Dict[str, List[ResultItem]] = defaultdict(list)
    progress_bar = tqdm(priming_dataloader, desc="Priming Eval", leave=False,
                        disable=not sys.stdout.isatty() or not tqdm_module)

    for batch_idx, batch in enumerate(progress_bar):
        if not batch:
            logger.warning(f"Skipping empty batch {batch_idx} from collate function.")
            continue
        try:
            batch_metrics_raw: BatchResults = calculate_priming_effect(
                model, batch, device, torch_rng, tokenizer, use_amp=use_amp,  # Pass tokenizer
                is_first_batch_for_corpus=(batch_idx == 0),
                max_verbose_items_per_batch=getattr(priming_dataloader.dataset, "max_verbose_items_to_log", 2)
                # Example, needs better way to pass
            )
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
        # ... (get other finite values) ...
        finite_logp_con_values = get_finite_values('logp_con')
        finite_logp_incon_values = get_finite_values('logp_incon')
        finite_logp_baseline_values = get_finite_values('logp_baseline')
        finite_logp_con_rand_values = get_finite_values('logp_con_random_baseline')
        finite_logp_incon_rand_values = get_finite_values('logp_incon_random_baseline')

        total_items = len(structure_results_list)
        logger.info(f"  Target '{target_structure}' (Total Items Processed in evaluator: {total_items}):")

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
        aggregate_and_log("LogP_baseline", "LogP_baseline", finite_logp_baseline_values)
        aggregate_and_log("LogP_con_random_baseline", "LogP_con_random_baseline", finite_logp_con_rand_values)
        aggregate_and_log("LogP_incon_random_baseline", "LogP_incon_random_baseline", finite_logp_incon_rand_values)

    if original_mode: model.train()
    logger.info("Native priming evaluation finished.")
    return final_aggregated_metrics, dict(all_results_raw)