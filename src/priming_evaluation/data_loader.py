# src/priming_evaluation/data_loader.py (Revised Collate for Baseline Indexing)

import logging
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple
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
    """
    Collates a batch for Priming Effect evaluation, including baseline inputs.
    Ensures target_start_baseline >= 1 for compatibility with evaluator.
    """
    batch = [item for item in batch if item is not None]
    if not batch: return {}

    max_length = max_length or getattr(tokenizer, 'model_max_length', None)
    collated_batch = defaultdict(list)
    items_to_keep_indices = []

    # --- Determine Tokenizer Properties (Robust check) ---
    test_text = "hello"  # Using a common word
    tokens_with_special = tokenizer.encode(test_text, add_special_tokens=True)
    tokens_without_special = tokenizer.encode(test_text, add_special_tokens=False)

    # Check for BOS addition
    adds_bos = False
    if tokenizer.bos_token_id is not None and \
            len(tokens_with_special) > 0 and \
            tokens_with_special[0] == tokenizer.bos_token_id:
        if len(tokens_without_special) == 0 or tokens_with_special[0] != tokens_without_special[0] or len(
                tokens_with_special) > len(tokens_without_special):
            adds_bos = True
    bos_offset = 1 if adds_bos else 0

    # Check for EOS addition (for consistent target length calculation if needed, though not directly for start indices)
    adds_eos = False
    if tokenizer.eos_token_id is not None and \
            len(tokens_with_special) > 0 and \
            tokens_with_special[-1] == tokenizer.eos_token_id:
        if len(tokens_without_special) == 0 or tokens_with_special[-1] != tokens_without_special[-1] or len(
                tokens_with_special) > len(tokens_without_special):
            adds_eos = True
    # eos_offset not strictly needed for start indices, but good to know
    # eos_len_effect = 1 if adds_eos else 0

    # --- Process each item ---
    for idx, item in enumerate(batch):
        target_sentence = item['target_sentence']
        # Tokenize target sentence without special tokens to get its actual length
        tokens_target_only_no_special = \
        tokenizer(target_sentence, add_special_tokens=False, return_attention_mask=False)['input_ids']
        num_actual_target_tokens = len(tokens_target_only_no_special)

        if num_actual_target_tokens == 0:
            logger.warning(
                f"Item {idx} (CSV: {item.get('source_csv', 'N/A')}, Row: {item.get('csv_row', 'N/A')}) has empty target sentence. Skipping.")
            continue

        # Congruent
        prime_congruent_context_str = item['congruent_prime'] + join_string
        tokens_prime_congruent_no_special = \
        tokenizer(prime_congruent_context_str, add_special_tokens=False, return_attention_mask=False)['input_ids']
        len_prime_congruent_tokens = len(tokens_prime_congruent_no_special)
        full_congruent_text = prime_congruent_context_str + target_sentence
        congruent_encoding = tokenizer(full_congruent_text, add_special_tokens=True,
                                       truncation=True if max_length else False, max_length=max_length,
                                       return_attention_mask=True)
        target_start_congruent = bos_offset + len_prime_congruent_tokens
        target_end_congruent = target_start_congruent + num_actual_target_tokens

        # Incongruent
        prime_incongruent_context_str = item['incongruent_prime'] + join_string
        tokens_prime_incongruent_no_special = \
        tokenizer(prime_incongruent_context_str, add_special_tokens=False, return_attention_mask=False)['input_ids']
        len_prime_incongruent_tokens = len(tokens_prime_incongruent_no_special)
        full_incongruent_text = prime_incongruent_context_str + target_sentence
        incongruent_encoding = tokenizer(full_incongruent_text, add_special_tokens=True,
                                         truncation=True if max_length else False, max_length=max_length,
                                         return_attention_mask=True)
        target_start_incongruent = bos_offset + len_prime_incongruent_tokens
        target_end_incongruent = target_start_incongruent + num_actual_target_tokens

        # Baseline
        # Ensure target_start_baseline >= 1 for evaluator compatibility (start-1 for logits)
        baseline_encoding_dict = None  # Will hold {'input_ids': list, 'attention_mask': list}

        if bos_offset == 1:  # Tokenizer adds BOS, so target will start at index 1
            _baseline_encoding_obj = tokenizer(target_sentence, add_special_tokens=True,
                                               truncation=True if max_length else False, max_length=max_length,
                                               return_attention_mask=True)
            baseline_encoding_dict = {'input_ids': _baseline_encoding_obj['input_ids'],
                                      'attention_mask': _baseline_encoding_obj['attention_mask']}
            target_start_baseline = bos_offset  # Should be 1
        else:  # bos_offset is 0, tokenizer does not automatically prepend BOS effectively
            if tokenizer.bos_token_id is not None:
                logger.debug(f"Manually prepending BOS for baseline (item {idx}) as tokenizer does not (bos_offset=0).")
                # Construct input_ids manually: [BOS, T1, T2, ..., Tn, (EOS if adds_eos)]
                manual_baseline_ids = [tokenizer.bos_token_id] + tokens_target_only_no_special
                if adds_eos and tokenizer.eos_token_id is not None:  # Check if tokenizer generally adds EOS
                    manual_baseline_ids.append(tokenizer.eos_token_id)

                # Truncate if necessary (manual truncation)
                if max_length and len(manual_baseline_ids) > max_length:
                    manual_baseline_ids = manual_baseline_ids[:max_length]
                    # Ensure target is not completely truncated
                    if len(manual_baseline_ids) <= 1:  # Only BOS left or empty
                        logger.warning(
                            f"Baseline for item {idx} truncated to BOS or empty after manual BOS. Skipping baseline part.")
                        # Mark baseline as invalid by setting target_start_baseline to 0, which evaluator will skip.
                        # Or, handle this more gracefully by not adding baseline at all.
                        # For now, let's make it so it *would* fail validation if this path is taken without valid target.
                        target_start_baseline = 0  # This will be caught by validation later
                    else:
                        target_start_baseline = 1
                else:
                    target_start_baseline = 1  # Target starts after manually added BOS

                baseline_encoding_dict = {
                    'input_ids': manual_baseline_ids,
                    'attention_mask': [1] * len(manual_baseline_ids)
                }
            else:  # No BOS token defined in tokenizer, and tokenizer doesn't add one.
                logger.warning(
                    f"Cannot ensure baseline target starts after BOS for item {idx} (bos_offset=0, tokenizer.bos_token_id is None). Baseline PPL might be incorrect or fail.")
                # Proceed with normal tokenization, target_start_baseline will be 0
                _baseline_encoding_obj = tokenizer(target_sentence, add_special_tokens=True,
                                                   truncation=True if max_length else False, max_length=max_length,
                                                   return_attention_mask=True)
                baseline_encoding_dict = {'input_ids': _baseline_encoding_obj['input_ids'],
                                          'attention_mask': _baseline_encoding_obj['attention_mask']}
                target_start_baseline = 0  # This will likely cause skipping in evaluator

        # Ensure num_actual_target_tokens is not zero if target_start_baseline is valid
        if target_start_baseline > 0 and len(
                baseline_encoding_dict['input_ids']) < target_start_baseline + num_actual_target_tokens:
            # This can happen if manual BOS + target + EOS got truncated too much
            logger.warning(
                f"Baseline target for item {idx} possibly truncated after manual BOS addition. Original target len: {num_actual_target_tokens}, available: {len(baseline_encoding_dict['input_ids']) - target_start_baseline}")
            # Adjust num_actual_target_tokens for this specific baseline instance if it got shorter
            # However, evaluator expects consistent target length based on 'congruent'.
            # This situation means this baseline item is problematic. We'll rely on later validation.
            pass

        target_end_baseline = target_start_baseline + num_actual_target_tokens

        # --- Validation of indices for each item ---
        # For an item to be valid, all its conditions' indices must be valid AND target_start_X >= 1
        len_con_ids = len(congruent_encoding['input_ids'])
        len_incon_ids = len(incongruent_encoding['input_ids'])
        len_base_ids = len(baseline_encoding_dict['input_ids'])

        valid_congruent = (target_start_congruent >= 1 and  # Crucial for evaluator: start-1
                           0 <= target_start_congruent < target_end_congruent <= len_con_ids and
                           (target_end_congruent - target_start_congruent) == num_actual_target_tokens)
        valid_incongruent = (target_start_incongruent >= 1 and
                             0 <= target_start_incongruent < target_end_incongruent <= len_incon_ids and
                             (target_end_incongruent - target_start_incongruent) == num_actual_target_tokens)
        valid_baseline = (target_start_baseline >= 1 and  # Crucial for evaluator: start-1
                          0 <= target_start_baseline < target_end_baseline <= len_base_ids and
                          (target_end_baseline - target_start_baseline) == num_actual_target_tokens)

        # If manual BOS prepending led to a baseline too short for the original num_actual_target_tokens
        if target_start_baseline >= 1 and (target_start_baseline + num_actual_target_tokens > len_base_ids):
            valid_baseline = False  # Target segment would go out of bounds

        if valid_congruent and valid_incongruent and valid_baseline:
            items_to_keep_indices.append(idx)  # Not strictly used later, but good for count
            collated_batch["_congruent_encoding"].append(congruent_encoding)  # Storing full BatchEncoding
            collated_batch["_incongruent_encoding"].append(incongruent_encoding)
            collated_batch["_baseline_encoding"].append(baseline_encoding_dict)  # Storing our dict

            collated_batch["target_start_congruent"].append(target_start_congruent)
            collated_batch["target_end_congruent"].append(target_end_congruent)
            collated_batch["target_start_incongruent"].append(target_start_incongruent)
            collated_batch["target_end_incongruent"].append(target_end_incongruent)
            collated_batch["target_start_baseline"].append(target_start_baseline)
            collated_batch["target_end_baseline"].append(target_end_baseline)

            collated_batch["target_structure"].append(item['target_structure'])
            collated_batch["source_csv"].append(item['source_csv'])  # For traceability
            collated_batch["csv_row"].append(item['csv_row'])  # For traceability
        else:
            logger.warning(
                f"Invalid item {idx} (CSV: {item.get('source_csv', 'N/A')}, Row: {item.get('csv_row', 'N/A')}). Skipping. "
                f"TargetLen={num_actual_target_tokens}. "
                f"Con: valid={valid_congruent}, s={target_start_congruent}, e={target_end_congruent}, len_ids={len_con_ids}. "
                f"Incon: valid={valid_incongruent}, s={target_start_incongruent}, e={target_end_incongruent}, len_ids={len_incon_ids}. "
                f"Base: valid={valid_baseline}, s={target_start_baseline}, e={target_end_baseline}, len_ids={len_base_ids}."
            )

    if not collated_batch["_congruent_encoding"]:  # Check if any items were kept
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

    # --- Create Labels Tensor (based on congruent condition) ---
    # This 'labels' tensor is what the evaluator uses.
    # The start_X and end_X indices point into this structure.
    labels_list = []
    for i in range(len(collated_batch["_congruent_encoding"])):  # Iterate over kept items
        # Use unpadded congruent input_ids to define the structure of the labels tensor
        # (where target tokens are, and -100 elsewhere)
        congruent_item_input_ids = collated_batch["_congruent_encoding"][i]['input_ids']
        s_con = collated_batch["target_start_congruent"][i]
        e_con = collated_batch["target_end_congruent"][i]

        # label tensor has same length as the unpadded congruent_input_ids for this item
        label_tensor_for_item = torch.full((len(congruent_item_input_ids),), -100, dtype=torch.long)
        if 0 <= s_con < e_con <= len(congruent_item_input_ids):
            label_tensor_for_item[s_con:e_con] = torch.tensor(congruent_item_input_ids[s_con:e_con], dtype=torch.long)
        else:  # Should not happen if validation above worked
            logger.error(
                f"Label construction error for item {i} despite passing earlier validation. s_con={s_con}, e_con={e_con}, len={len(congruent_item_input_ids)}")
        labels_list.append(label_tensor_for_item)

    collated_batch['labels'] = pad_sequence(labels_list, batch_first=True, padding_value=-100)

    # Convert index lists to tensors
    index_keys = [
        "target_start_congruent", "target_end_congruent",
        "target_start_incongruent", "target_end_incongruent",
        "target_start_baseline", "target_end_baseline"
    ]
    for key in index_keys:
        if key in collated_batch:
            collated_batch[key] = torch.tensor(collated_batch[key], dtype=torch.long)

    # Clean up temporary encoding lists
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
    collate_fn_partial = partial(collate_priming_eval_batch, tokenizer=tokenizer, join_string=join_string + " ",
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