# src/priming_evaluation/data_loader.py (Revised Collate for Baseline Indexing)

import logging
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import random

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import PreTrainedTokenizer, BatchEncoding

logger = logging.getLogger(__name__)

# Define prefixes
PRIME_PREFIX = 'p'
TARGET_PREFIX = 't'


class PrimingEvaluationDataset(Dataset):
    def __init__(self, processed_data: List[Dict[str, Any]]):
        if not isinstance(processed_data, list): raise TypeError(f"Expected list, got {type(processed_data)}")
        self.data = processed_data
        if not self.data: logger.warning("PrimingEvaluationDataset initialized with no data.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not isinstance(idx, int): raise TypeError(f"Index must be int, got {type(idx)}")
        if idx >= len(self.data): raise IndexError(f"Index {idx} out of range for len {len(self.data)}")
        return self.data[idx]


def get_structure_alternations(columns: List[str]) -> Optional[Tuple[str, str]]:
    structures = set()
    for col in columns:
        if col.startswith(PRIME_PREFIX) or col.startswith(TARGET_PREFIX):
            structure_name = col[1:]
            structures.add(structure_name)
    if len(structures) == 2:
        return tuple(sorted(list(structures)))
    elif len(structures) > 2:
        logger.warning(f"Found >2 structures: {structures}. Cannot determine pair."); return None
    else:
        logger.warning(f"Could not find 2 structures from: {structures}"); return None


def load_and_process_priming_data(
        csv_path: Path,
        tokenizer: PreTrainedTokenizer,  # Keep for consistency, though not directly used here
        delimiter: str = ".",
) -> List[Dict[str, Any]]:
    processed_data = [];
    csv_filename = csv_path.name
    try:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            logger.warning(f"Pandas C engine failed for {csv_filename}. Trying Python."); df = pd.read_csv(csv_path,
                                                                                                           engine='python')
        df.columns = df.columns.str.strip();
        logger.debug(f"Cleaned columns for {csv_filename}: {list(df.columns)}")
    except FileNotFoundError:
        logger.error(f"CSV not found: {csv_path}"); return []
    except Exception as e:
        logger.error(f"Error loading CSV {csv_filename}: {e}"); return []

    alternation = get_structure_alternations(list(df.columns))
    if alternation is None: logger.error(
        f"Cannot determine alternation for {csv_filename}. Cols: {list(df.columns)}"); return []
    struct_x, struct_y = alternation;
    logger.info(f"Alternation for {csv_filename}: {struct_x} / {struct_y}")

    prime_col_x, prime_col_y = f"{PRIME_PREFIX}{struct_x}", f"{PRIME_PREFIX}{struct_y}"
    target_col_x, target_col_y = f"{TARGET_PREFIX}{struct_x}", f"{TARGET_PREFIX}{struct_y}"
    required_cols = [prime_col_x, prime_col_y, target_col_x, target_col_y]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns];
        logger.error(f"CSV {csv_filename} missing required columns: {missing}");
        return []

    logger.info(f"Processing {len(df)} rows from {csv_filename} for {struct_x}/{struct_y}...")
    items_created_count = 0
    for index, row in df.iterrows():
        try:
            prime_x_sent, prime_y_sent = str(row[prime_col_x]), str(row[prime_col_y])
            target_x_sent, target_y_sent = str(row[target_col_x]), str(row[target_col_y])
            if not all(s and isinstance(s, str) and s.strip() and s.lower() != 'nan' for s in
                       [prime_x_sent, prime_y_sent, target_x_sent, target_y_sent]):
                continue

            processed_data.append(
                {"target_sentence": target_x_sent, "congruent_prime": prime_x_sent, "incongruent_prime": prime_y_sent,
                 "target_structure": target_col_x, "congruent_prime_structure": prime_col_x,
                 "incongruent_prime_structure": prime_col_y, "source_csv": csv_filename, "csv_row": index})
            processed_data.append(
                {"target_sentence": target_y_sent, "congruent_prime": prime_y_sent, "incongruent_prime": prime_x_sent,
                 "target_structure": target_col_y, "congruent_prime_structure": prime_col_y,
                 "incongruent_prime_structure": prime_col_x, "source_csv": csv_filename, "csv_row": index})
            items_created_count += 2
        except Exception as e:
            logger.warning(f"Error processing row {index} in {csv_filename}: {e}. Skip."); continue
    logger.info(f"Finished processing {csv_filename}. Created {items_created_count} valid items from {len(df)} rows.");
    if items_created_count == 0: logger.warning(f"No valid items processed from {csv_filename}.")
    return processed_data


def collate_priming_eval_batch(
        batch: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        join_string: str = ". ",
        max_length: Optional[int] = None,
) -> Dict[str, Any]:
    batch = [item for item in batch if item is not None]
    if not batch: return {}

    # Determine effective max_length
    effective_max_length = max_length or getattr(tokenizer, 'model_max_length', None)
    if effective_max_length is None:  # Should not happen with typical tokenizers if max_length not set
        logger.warning(
            "Effective max_length is None; truncation will be disabled. This might lead to very long sequences.")
        # tokenizer will handle truncation=False if max_length is None

    collated_batch = defaultdict(list)
    items_kept_count = 0  # Changed from items_to_keep_indices as we are not using it to index `batch` later

    # --- Determine Tokenizer Properties (Robust check) ---
    test_text = "hello"
    tokens_with_special = tokenizer.encode(test_text, add_special_tokens=True)
    tokens_without_special = tokenizer.encode(test_text, add_special_tokens=False)
    adds_bos = False
    if tokenizer.bos_token_id is not None and \
            len(tokens_with_special) > 0 and \
            tokens_with_special[0] == tokenizer.bos_token_id:
        if len(tokens_without_special) == 0 or tokens_with_special[0] != tokens_without_special[0] or len(
                tokens_with_special) > len(tokens_without_special):
            adds_bos = True
    bos_offset = 1 if adds_bos else 0
    adds_eos = False  # Determine if EOS is typically added at the end
    if tokenizer.eos_token_id is not None and \
            len(tokens_with_special) > 0 and \
            tokens_with_special[-1] == tokenizer.eos_token_id:
        if len(tokens_without_special) == 0 or tokens_with_special[-1] != tokens_without_special[-1] or len(
                tokens_with_special) > len(tokens_without_special):
            adds_eos = True

    # --- Process each item ---
    for idx, item in enumerate(batch):
        target_sentence = item['target_sentence']
        tokens_target_only_no_special = \
        tokenizer(target_sentence, add_special_tokens=False, return_attention_mask=False)['input_ids']
        original_num_target_tokens = len(tokens_target_only_no_special)

        if original_num_target_tokens == 0:
            logger.warning(
                f"Item {idx} (CSV: {item.get('source_csv', 'N/A')}, Row: {item.get('csv_row', 'N/A')}) has empty target sentence. Skipping.")
            continue

        # --- Congruent Condition ---
        prime_congruent_context_str = item['congruent_prime'] + join_string
        tokens_prime_congruent_no_special = \
        tokenizer(prime_congruent_context_str, add_special_tokens=False, return_attention_mask=False)['input_ids']
        len_prime_congruent_tokens = len(tokens_prime_congruent_no_special)
        full_congruent_text = prime_congruent_context_str + target_sentence
        congruent_encoding = tokenizer(full_congruent_text, add_special_tokens=True,
                                       truncation=True if effective_max_length else False,
                                       max_length=effective_max_length, return_attention_mask=True)
        len_con_ids = len(congruent_encoding['input_ids'])

        target_start_congruent = bos_offset + len_prime_congruent_tokens

        # Determine effective target length for this item based on what fits in congruent
        # Space available for target tokens in congruent_encoding
        space_for_target_con = len_con_ids - target_start_congruent
        # If EOS is present and at the end, it takes one slot from this space if not already part of target
        # However, len_con_ids already includes EOS. The target must end *before or at* the last non-PAD token.
        # Simpler: the target runs from target_start_congruent for some length.
        # The maximum index it can reach is len_con_ids - 1.
        # So, target_end_congruent (exclusive) can be at most len_con_ids.

        item_effective_target_len = 0
        if target_start_congruent < len_con_ids:  # Target must at least start within bounds
            # Max possible length of target given the start and total length
            max_fit_target_len_con = len_con_ids - target_start_congruent
            item_effective_target_len = min(original_num_target_tokens, max_fit_target_len_con)

        if item_effective_target_len <= 0:
            logger.warning(
                f"Item {idx} (Congruent): Target does not fit or is empty. original_target_len={original_num_target_tokens}, effective_len={item_effective_target_len}, s_con={target_start_congruent}, len_con_ids={len_con_ids}. Skipping item.")
            continue

        target_end_congruent = target_start_congruent + item_effective_target_len

        # --- Incongruent Condition (uses item_effective_target_len from congruent) ---
        prime_incongruent_context_str = item['incongruent_prime'] + join_string
        tokens_prime_incongruent_no_special = \
        tokenizer(prime_incongruent_context_str, add_special_tokens=False, return_attention_mask=False)['input_ids']
        len_prime_incongruent_tokens = len(tokens_prime_incongruent_no_special)
        full_incongruent_text = prime_incongruent_context_str + target_sentence
        incongruent_encoding = tokenizer(full_incongruent_text, add_special_tokens=True,
                                         truncation=True if effective_max_length else False,
                                         max_length=effective_max_length, return_attention_mask=True)
        len_incon_ids = len(incongruent_encoding['input_ids'])
        target_start_incongruent = bos_offset + len_prime_incongruent_tokens
        target_end_incongruent = target_start_incongruent + item_effective_target_len  # Use consistent length

        # --- Baseline Condition (uses item_effective_target_len from congruent) ---
        baseline_encoding_dict = None
        target_start_baseline = 0  # Initialize to ensure it's set
        if bos_offset == 1:
            _baseline_encoding_obj = tokenizer(target_sentence, add_special_tokens=True,
                                               truncation=True if effective_max_length else False,
                                               max_length=effective_max_length, return_attention_mask=True)
            baseline_encoding_dict = {'input_ids': _baseline_encoding_obj['input_ids'],
                                      'attention_mask': _baseline_encoding_obj['attention_mask']}
            target_start_baseline = bos_offset
        else:
            if tokenizer.bos_token_id is not None:
                manual_baseline_ids = [tokenizer.bos_token_id] + tokens_target_only_no_special[
                                                                 :item_effective_target_len]  # Use potentially shortened target
                if adds_eos and tokenizer.eos_token_id is not None:
                    manual_baseline_ids.append(tokenizer.eos_token_id)
                if effective_max_length and len(manual_baseline_ids) > effective_max_length:
                    manual_baseline_ids = manual_baseline_ids[:effective_max_length]

                target_start_baseline = 1
                # Re-evaluate effective length for this manually constructed seq
                # This ensures the baseline encoding itself is valid for item_effective_target_len
                if len(manual_baseline_ids) < target_start_baseline + item_effective_target_len:
                    # If after manual construction and truncation, target doesn't fit, this baseline is problematic
                    # This path means this item will likely fail validation for baseline
                    pass  # Rely on validation below

                baseline_encoding_dict = {'input_ids': manual_baseline_ids,
                                          'attention_mask': [1] * len(manual_baseline_ids)}
            else:
                _baseline_encoding_obj = tokenizer(target_sentence, add_special_tokens=True,
                                                   truncation=True if effective_max_length else False,
                                                   max_length=effective_max_length, return_attention_mask=True)
                baseline_encoding_dict = {'input_ids': _baseline_encoding_obj['input_ids'],
                                          'attention_mask': _baseline_encoding_obj['attention_mask']}
                target_start_baseline = 0  # Will fail validation

        target_end_baseline = target_start_baseline + item_effective_target_len  # Use consistent length
        len_base_ids = len(baseline_encoding_dict['input_ids'])

        # --- Validation of indices for each item ---
        valid_congruent = (target_start_congruent >= 1 and
                           0 <= target_start_congruent < target_end_congruent <= len_con_ids and
                           (
                                       target_end_congruent - target_start_congruent) == item_effective_target_len)  # Check against effective length
        valid_incongruent = (target_start_incongruent >= 1 and
                             0 <= target_start_incongruent < target_end_incongruent <= len_incon_ids and
                             (target_end_incongruent - target_start_incongruent) == item_effective_target_len)
        valid_baseline = (target_start_baseline >= 1 and
                          0 <= target_start_baseline < target_end_baseline <= len_base_ids and
                          (target_end_baseline - target_start_baseline) == item_effective_target_len)

        if valid_congruent and valid_incongruent and valid_baseline:
            items_kept_count += 1
            collated_batch["_congruent_encoding"].append(congruent_encoding)
            collated_batch["_incongruent_encoding"].append(incongruent_encoding)
            collated_batch["_baseline_encoding"].append(baseline_encoding_dict)
            collated_batch["target_start_congruent"].append(target_start_congruent)
            collated_batch["target_end_congruent"].append(target_end_congruent)
            collated_batch["target_start_incongruent"].append(target_start_incongruent)
            collated_batch["target_end_incongruent"].append(target_end_incongruent)
            collated_batch["target_start_baseline"].append(target_start_baseline)
            collated_batch["target_end_baseline"].append(target_end_baseline)
            collated_batch["target_structure"].append(item['target_structure'])
            collated_batch["source_csv"].append(item['source_csv'])
            collated_batch["csv_row"].append(item['csv_row'])
        else:
            logger.warning(
                f"Invalid item {idx} (CSV: {item.get('source_csv', 'N/A')}, Row: {item.get('csv_row', 'N/A')}). Skipping. "
                f"OrigTargetLen={original_num_target_tokens}, ItemEffectiveTargetLen={item_effective_target_len}. "
                f"Con: valid={valid_congruent}, s={target_start_congruent}, e={target_end_congruent}, len_ids={len_con_ids}. "
                f"Incon: valid={valid_incongruent}, s={target_start_incongruent}, e={target_end_incongruent}, len_ids={len_incon_ids}. "
                f"Base: valid={valid_baseline}, s={target_start_baseline}, e={target_end_baseline}, len_ids={len_base_ids}."
            )

    if items_kept_count == 0:
        logger.warning("Collate function resulted in an empty batch after index validation.")
        return {}

    # --- Padding ---
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collated_batch['congruent_input_ids'] = pad_batch_encodings(collated_batch["_congruent_encoding"], pad_token_id,
                                                                key='input_ids')
    collated_batch['congruent_attention_mask'] = pad_batch_encodings(collated_batch["_congruent_encoding"], 0,
                                                                     key='attention_mask')
    collated_batch['incongruent_input_ids'] = pad_batch_encodings(collated_batch["_incongruent_encoding"], pad_token_id,
                                                                  key='input_ids')
    collated_batch['incongruent_attention_mask'] = pad_batch_encodings(collated_batch["_incongruent_encoding"], 0,
                                                                       key='attention_mask')
    collated_batch['baseline_input_ids'] = pad_batch_encodings(collated_batch["_baseline_encoding"], pad_token_id,
                                                               key='input_ids')
    collated_batch['baseline_attention_mask'] = pad_batch_encodings(collated_batch["_baseline_encoding"], 0,
                                                                    key='attention_mask')

    # --- Create Labels Tensor (based on congruent condition and item_effective_target_len) ---
    labels_list = []
    for i in range(items_kept_count):  # Iterate over kept items
        congruent_item_input_ids = collated_batch["_congruent_encoding"][i]['input_ids']
        s_con = collated_batch["target_start_congruent"][
            i]  # This is a tensor now, use .item() if needed, but it's fine for slicing here
        e_con = collated_batch["target_end_congruent"][i]  # This uses item_effective_target_len

        label_tensor_for_item = torch.full((len(congruent_item_input_ids),), -100, dtype=torch.long)
        # s_con and e_con from the collated_batch are already calculated to be valid if the item was kept
        label_tensor_for_item[s_con:e_con] = torch.tensor(congruent_item_input_ids[s_con:e_con], dtype=torch.long)
        labels_list.append(label_tensor_for_item)

    collated_batch['labels'] = pad_sequence(labels_list, batch_first=True, padding_value=-100)

    index_keys = [
        "target_start_congruent", "target_end_congruent",
        "target_start_incongruent", "target_end_incongruent",
        "target_start_baseline", "target_end_baseline"
    ]
    for key in index_keys:
        if key in collated_batch:
            collated_batch[key] = torch.tensor(collated_batch[key], dtype=torch.long)

    del collated_batch["_congruent_encoding"]
    del collated_batch["_incongruent_encoding"]
    del collated_batch["_baseline_encoding"]

    return dict(collated_batch)


def pad_batch_encodings(encodings: List[Union[BatchEncoding, Dict[str, list]]], pad_value: int,
                        key: str = 'input_ids') -> torch.Tensor:
    """ Pads a list of BatchEncoding outputs or dicts {'input_ids': list, 'attention_mask': list}. """
    sequences = []
    for enc in encodings:
        if isinstance(enc, BatchEncoding):
            sequences.append(torch.tensor(enc[key]))
        elif isinstance(enc, dict) and key in enc:  # For our manually constructed baseline_encoding_dict
            sequences.append(torch.tensor(enc[key]))
        else:
            raise TypeError(f"Unsupported encoding type in pad_batch_encodings: {type(enc)}")
    return pad_sequence(sequences, batch_first=True, padding_value=pad_value)


def create_priming_dataloader(
        csv_path: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        delimiter: str = ".",
        num_workers: int = 0,
        max_length: Optional[int] = None,
        max_samples: int = -1,
        seed: int = 42,
        **kwargs
) -> Optional[DataLoader]:
    csv_path_obj = Path(csv_path)
    logger.info(f"Creating priming dataloader for: {csv_path_obj.name}")
    logger.info(f"Params: batch_size={batch_size}, max_samples={max_samples}, seed={seed}, max_length={max_length}")

    processed_data = load_and_process_priming_data(csv_path=csv_path_obj, tokenizer=tokenizer, delimiter=delimiter)

    if not processed_data:
        logger.warning(f"No data processed from {csv_path_obj.name}. Returning None for DataLoader.")
        return None

    original_size = len(processed_data)
    logger.info(f"Initially processed {original_size:,} items from {csv_path_obj.name}.")

    final_processed_data = processed_data
    if max_samples > 0 and max_samples < original_size:
        logger.info(f"Sampling {max_samples:,} items from the processed data (seed: {seed}).")
        random.seed(seed)
        try:
            final_processed_data = random.sample(processed_data, k=max_samples)
            logger.info(f"Using subset: {len(final_processed_data):,} items.")
        except ValueError as e:  # k > n
            logger.warning(
                f"Sampling error (k={max_samples}, n={original_size}): {e}. Using all {original_size} items instead.")
            # final_processed_data remains processed_data
    elif max_samples > 0 and original_size > 0:  # max_samples >= original_size
        logger.info(f"Max_samples ({max_samples:,}) >= processed items ({original_size:,}). Using all processed items.")
    elif original_size > 0:  # max_samples <= 0, use all
        logger.info(f"Max_samples <= 0. Using all {original_size:,} processed items.")
    else:  # original_size is 0
        pass

    if not final_processed_data:
        logger.warning(
            f"Data list is empty after sampling for {csv_path_obj.name} (or was initially empty). Returning None.")
        return None

    dataset = PrimingEvaluationDataset(final_processed_data)
    # Note: join_string in partial is item['congruent_prime'] + join_string + item['target_sentence']
    # The join_string here is what separates prime and target.
    collate_fn_partial = partial(collate_priming_eval_batch, tokenizer=tokenizer, join_string=delimiter + " ",
                                 max_length=max_length)

    sampler = SequentialSampler(dataset)  # Evaluation should be sequential
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collate_fn_partial,
        num_workers=num_workers,
        shuffle=False,  # Essential for SequentialSampler
        pin_memory=kwargs.get('pin_memory', False),  # Allow pin_memory to be passed
        drop_last=kwargs.get('drop_last', False)  # Allow drop_last to be passed
    )

    logger.info(f"Priming Effect DataLoader created for {csv_path_obj.name} with {len(dataset)} items.")
    return dataloader