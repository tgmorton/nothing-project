# src/priming_evaluation/data_loader.py (Revised with Sampling and Baseline in Collate)

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

# --- PrimingEvaluationDataset (No changes needed) ---
class PrimingEvaluationDataset(Dataset):
    """ PyTorch Dataset for priming evaluation based on the Sinclair et al. (2021) paper. """
    def __init__(self, processed_data: List[Dict[str, Any]]):
        if not isinstance(processed_data, list): raise TypeError(f"Expected list, got {type(processed_data)}")
        self.data = processed_data
        if not self.data: logger.warning("PrimingEvaluationDataset initialized with no data.")
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        if not isinstance(idx, int): raise TypeError(f"Index must be int, got {type(idx)}")
        if idx >= len(self.data): raise IndexError(f"Index {idx} out of range for len {len(self.data)}")
        return self.data[idx]

# --- get_structure_alternations (No changes needed) ---
def get_structure_alternations(columns: List[str]) -> Optional[Tuple[str, str]]:
    """Identifies the two structure types (e.g., 'po', 'do') from column names."""
    structures = set()
    for col in columns:
        if col.startswith(PRIME_PREFIX) or col.startswith(TARGET_PREFIX):
            structure_name = col[1:] # Remove prefix 'p' or 't'
            structures.add(structure_name)
    if len(structures) == 2: return tuple(sorted(list(structures)))
    elif len(structures) > 2: logger.warning(f"Found >2 structures: {structures}. Cannot determine pair."); return None
    else: logger.warning(f"Could not find 2 structures from: {structures}"); return None

# --- load_and_process_priming_data (No changes needed) ---
def load_and_process_priming_data(
    csv_path: Path,
    tokenizer: PreTrainedTokenizer,
    delimiter: str = ".",
) -> List[Dict[str, Any]]:
    """ Loads priming data, identifies prime/target pairs per row based on column prefixes. """
    processed_data = []; csv_filename = csv_path.name
    try:
        try: df = pd.read_csv(csv_path)
        except Exception: logger.warning(f"Pandas C engine failed for {csv_filename}. Trying Python."); df = pd.read_csv(csv_path, engine='python')
        df.columns = df.columns.str.strip(); logger.debug(f"Cleaned columns for {csv_filename}: {list(df.columns)}")
    except FileNotFoundError: logger.error(f"CSV not found: {csv_path}"); return []
    except Exception as e: logger.error(f"Error loading CSV {csv_filename}: {e}"); return []

    alternation = get_structure_alternations(list(df.columns))
    if alternation is None: logger.error(f"Cannot determine alternation for {csv_filename}. Cols: {list(df.columns)}"); return []
    struct_x, struct_y = alternation; logger.info(f"Alternation for {csv_filename}: {struct_x} / {struct_y}")

    prime_col_x, prime_col_y = f"{PRIME_PREFIX}{struct_x}", f"{PRIME_PREFIX}{struct_y}"
    target_col_x, target_col_y = f"{TARGET_PREFIX}{struct_x}", f"{TARGET_PREFIX}{struct_y}"
    required_cols = [prime_col_x, prime_col_y, target_col_x, target_col_y]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]; logger.error(f"CSV {csv_filename} missing required columns: {missing}"); return []

    logger.info(f"Processing {len(df)} rows from {csv_filename} for {struct_x}/{struct_y}...")
    items_created_count = 0
    for index, row in df.iterrows():
        try:
            prime_x_sent, prime_y_sent = str(row[prime_col_x]), str(row[prime_col_y])
            target_x_sent, target_y_sent = str(row[target_col_x]), str(row[target_col_y])
            if not all(s and isinstance(s, str) and s.strip() and s.lower() != 'nan' for s in [prime_x_sent, prime_y_sent, target_x_sent, target_y_sent]):
                 continue

            processed_data.append({"target_sentence": target_x_sent, "congruent_prime": prime_x_sent, "incongruent_prime": prime_y_sent, "target_structure": target_col_x, "congruent_prime_structure": prime_col_x, "incongruent_prime_structure": prime_col_y, "source_csv": csv_filename, "csv_row": index})
            processed_data.append({"target_sentence": target_y_sent, "congruent_prime": prime_y_sent, "incongruent_prime": prime_x_sent, "target_structure": target_col_y, "congruent_prime_structure": prime_col_y, "incongruent_prime_structure": prime_col_x, "source_csv": csv_filename, "csv_row": index})
            items_created_count += 2
        except Exception as e: logger.warning(f"Error processing row {index} in {csv_filename}: {e}. Skip."); continue
    logger.info(f"Finished processing {csv_filename}. Created {items_created_count} valid items from {len(df)} rows.");
    if items_created_count == 0: logger.warning(f"No valid items processed from {csv_filename}.")
    return processed_data


# --- REVISED collate_priming_eval_batch (for Baseline probability) ---
def collate_priming_eval_batch(
    batch: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    join_string: str = ". ",
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Collates a batch for Priming Effect evaluation, including baseline inputs.
    """
    batch = [item for item in batch if item is not None]
    if not batch: return {}

    max_length = max_length or getattr(tokenizer, 'model_max_length', None)
    collated_batch = defaultdict(list)
    items_to_keep_indices = []

    # --- Determine Tokenizer Properties ---
    # Tokenize a dummy string to check for BOS/EOS addition
    # It's important that add_special_tokens=True is used for dummy_tokens
    # and add_special_tokens=False for dummy_tokens_no_special to correctly infer.
    dummy_text = "test"
    # Ensure dummy_tokens uses add_special_tokens=True for accurate BOS/EOS detection.
    dummy_tokens_obj = tokenizer(dummy_text, add_special_tokens=True)
    dummy_tokens = dummy_tokens_obj['input_ids']
    # Ensure dummy_tokens_no_special uses add_special_tokens=False.
    dummy_tokens_no_special = tokenizer(dummy_text, add_special_tokens=False)['input_ids']

    adds_bos = (
        tokenizer.bos_token_id is not None and
        len(dummy_tokens) > 0 and # Ensure dummy_tokens is not empty
        dummy_tokens[0] == tokenizer.bos_token_id and
        (len(dummy_tokens_no_special) == 0 or dummy_tokens[0] != dummy_tokens_no_special[0])
    )
    adds_eos = (
        tokenizer.eos_token_id is not None and
        len(dummy_tokens) > 0 and # Ensure dummy_tokens is not empty
        dummy_tokens[-1] == tokenizer.eos_token_id and
        (len(dummy_tokens_no_special) == 0 or dummy_tokens[-1] != dummy_tokens_no_special[-1])
    )
    # Correction: if no special tokens are added, but BOS/EOS is part of normal tokenization,
    # this logic might misinterpret. The key is if `add_special_tokens=True` *results* in BOS/EOS.
    # A simpler check:
    test_tokens_with_special = tokenizer.encode("a", add_special_tokens=True)
    test_tokens_without_special = tokenizer.encode("a", add_special_tokens=False)

    adds_bos = (tokenizer.bos_token_id is not None and
                len(test_tokens_with_special) > len(test_tokens_without_special) and
                test_tokens_with_special[0] == tokenizer.bos_token_id)
    adds_eos = (tokenizer.eos_token_id is not None and
                len(test_tokens_with_special) > len(test_tokens_without_special) and
                test_tokens_with_special[-1] == tokenizer.eos_token_id)

    bos_offset = 1 if adds_bos else 0
    eos_offset = 1 if adds_eos else 0
    # logger.debug(f"Tokenizer properties: adds_bos={adds_bos} (offset={bos_offset}), adds_eos={adds_eos} (offset={eos_offset})")


    # --- Process each item ---
    for idx, item in enumerate(batch):
        target_sentence = item['target_sentence']

        # Congruent
        prime_congruent_context = item['congruent_prime'] + join_string
        tokens_prime_congruent_context = tokenizer(prime_congruent_context, add_special_tokens=False, return_attention_mask=False)['input_ids']
        len_prime_congruent_context = len(tokens_prime_congruent_context)
        full_congruent_text = prime_congruent_context + target_sentence
        congruent_encoding = tokenizer(full_congruent_text, add_special_tokens=True, truncation=True if max_length else False, max_length=max_length, return_attention_mask=True)
        target_start_congruent = bos_offset + len_prime_congruent_context
        target_end_congruent = len(congruent_encoding['input_ids']) - eos_offset

        # Incongruent
        prime_incongruent_context = item['incongruent_prime'] + join_string
        tokens_prime_incongruent_context = tokenizer(prime_incongruent_context, add_special_tokens=False, return_attention_mask=False)['input_ids']
        len_prime_incongruent_context = len(tokens_prime_incongruent_context)
        full_incongruent_text = prime_incongruent_context + target_sentence
        incongruent_encoding = tokenizer(full_incongruent_text, add_special_tokens=True, truncation=True if max_length else False, max_length=max_length, return_attention_mask=True)
        target_start_incongruent = bos_offset + len_prime_incongruent_context
        target_end_incongruent = len(incongruent_encoding['input_ids']) - eos_offset

        # Baseline (Target sentence only, with special tokens)
        baseline_encoding = tokenizer(target_sentence, add_special_tokens=True, truncation=True if max_length else False, max_length=max_length, return_attention_mask=True)
        target_start_baseline = bos_offset # Target starts after BOS token (if any)
        target_end_baseline = len(baseline_encoding['input_ids']) - eos_offset # Target ends before EOS token (if any)


        # Validate all indices
        # The end index is exclusive for slicing, so it should be > start index
        valid_congruent = (0 <= target_start_congruent < target_end_congruent <= len(congruent_encoding['input_ids']))
        valid_incongruent = (0 <= target_start_incongruent < target_end_incongruent <= len(incongruent_encoding['input_ids']))
        valid_baseline = (0 <= target_start_baseline < target_end_baseline <= len(baseline_encoding['input_ids']))

        # Ensure target length is positive for all
        # (target_end > target_start already implies length > 0 if they are token indices)

        if valid_congruent and valid_incongruent and valid_baseline:
            items_to_keep_indices.append(idx)
            collated_batch["_congruent_encoding"].append(congruent_encoding)
            collated_batch["_incongruent_encoding"].append(incongruent_encoding)
            collated_batch["_baseline_encoding"].append(baseline_encoding) # Add baseline encoding

            collated_batch["target_start_congruent"].append(target_start_congruent)
            collated_batch["target_end_congruent"].append(target_end_congruent)
            collated_batch["target_start_incongruent"].append(target_start_incongruent)
            collated_batch["target_end_incongruent"].append(target_end_incongruent)
            collated_batch["target_start_baseline"].append(target_start_baseline) # Add baseline start
            collated_batch["target_end_baseline"].append(target_end_baseline)   # Add baseline end

            collated_batch["target_structure"].append(item['target_structure'])
            collated_batch["source_csv"].append(item['source_csv'])
            collated_batch["csv_row"].append(item['csv_row'])
        else:
             logger.warning(
                 f"Invalid target indices for row {item.get('csv_row', 'N/A')} from {item.get('source_csv', 'N/A')}. "
                 f"Con: valid={valid_congruent}, s={target_start_congruent}, e={target_end_congruent}, len={len(congruent_encoding['input_ids'])}. "
                 f"Incon: valid={valid_incongruent}, s={target_start_incongruent}, e={target_end_incongruent}, len={len(incongruent_encoding['input_ids'])}. "
                 f"Base: valid={valid_baseline}, s={target_start_baseline}, e={target_end_baseline}, len={len(baseline_encoding['input_ids'])}. "
                 "Skipping item."
            )


    if not items_to_keep_indices: # or not collated_batch["_congruent_encoding"]:
        logger.warning("Collate function resulted in an empty batch after index validation.")
        return {}

    # --- Padding ---
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0 # Default to 0 if not set
    collated_batch['congruent_input_ids'] = pad_batch_encodings(collated_batch["_congruent_encoding"], pad_token_id, key='input_ids')
    collated_batch['congruent_attention_mask'] = pad_batch_encodings(collated_batch["_congruent_encoding"], 0, key='attention_mask')
    collated_batch['incongruent_input_ids'] = pad_batch_encodings(collated_batch["_incongruent_encoding"], pad_token_id, key='input_ids')
    collated_batch['incongruent_attention_mask'] = pad_batch_encodings(collated_batch["_incongruent_encoding"], 0, key='attention_mask')
    collated_batch['baseline_input_ids'] = pad_batch_encodings(collated_batch["_baseline_encoding"], pad_token_id, key='input_ids') # Pad baseline
    collated_batch['baseline_attention_mask'] = pad_batch_encodings(collated_batch["_baseline_encoding"], 0, key='attention_mask') # Pad baseline


    # --- Create Labels Tensor ---
    # The labels tensor represents the target tokens. It should be consistent.
    # We derive it from one of the encodings (e.g., congruent) and its target indices.
    # The evaluator will use the respective target_start_* and target_end_* for each condition
    # to slice the logits and compare against the relevant part of this common labels tensor.

    # Determine max length for labels based on actual tokenized lengths before padding
    # This is tricky because labels should ideally match the target segment from *any* of the inputs.
    # Let's assume the target sequence tokens are the same, and use congruent for label construction.
    # The evaluator expects labels[i, start_X : end_X] to be the target tokens.
    # The `start_X` and `end_X` indices are already adjusted for BOS/EOS within each input's tokenization.

    labels_list = []
    for i in range(len(collated_batch["_congruent_encoding"])): # Iterate over kept items
         # Use congruent_input_ids (unpadded, from _congruent_encoding) to extract target tokens for labels
         # These are the actual token IDs of the target sentence as it appeared in the congruent context
         input_ids_for_label_extraction = collated_batch["_congruent_encoding"][i]['input_ids']
         start_idx_for_label = collated_batch["target_start_congruent"][i]
         end_idx_for_label = collated_batch["target_end_congruent"][i]

         # Create a full label tensor for this item, initially all -100
         # The length of this label tensor should correspond to the *padded* length of the input_ids
         # for the congruent condition (or any, if they are padded to the same max_len_batch).
         # The evaluator slices logits using start-1:end-1 and labels using start:end.
         # So, labels need to be structured such that labels[start:end] are the target tokens.

         # Use the *unpadded* length for label creation before padding the list of labels.
         current_item_label = torch.full((len(input_ids_for_label_extraction),), -100, dtype=torch.long)

         if 0 <= start_idx_for_label < end_idx_for_label <= len(input_ids_for_label_extraction):
             target_tokens = torch.tensor(input_ids_for_label_extraction[start_idx_for_label:end_idx_for_label], dtype=torch.long)
             current_item_label[start_idx_for_label:end_idx_for_label] = target_tokens
         else:
             logger.error(f"Label index mismatch for item {i} (using congruent source for labels). "
                          f"Start={start_idx_for_label}, End={end_idx_for_label}, InputLen={len(input_ids_for_label_extraction)}. Setting empty label.")
             # current_item_label remains all -100
         labels_list.append(current_item_label)

    collated_batch['labels'] = pad_sequence(labels_list, batch_first=True, padding_value=-100)


    # Convert index lists to tensors
    index_keys = [
        "target_start_congruent", "target_end_congruent",
        "target_start_incongruent", "target_end_incongruent",
        "target_start_baseline", "target_end_baseline"
    ]
    for key in index_keys:
        if key in collated_batch: # Check if key exists (it should if items were kept)
             collated_batch[key] = torch.tensor(collated_batch[key], dtype=torch.long)

    # Clean up temporary encoding lists
    del collated_batch["_congruent_encoding"]
    del collated_batch["_incongruent_encoding"]
    del collated_batch["_baseline_encoding"]

    return dict(collated_batch)


# --- pad_batch_encodings (No changes needed) ---
def pad_batch_encodings(encodings: List[BatchEncoding], pad_value: int, key: str = 'input_ids') -> torch.Tensor:
    """ Pads a list of BatchEncoding outputs. """
    sequences = [torch.tensor(enc[key]) for enc in encodings]; return pad_sequence(sequences, batch_first=True, padding_value=pad_value)


# --- create_priming_dataloader (No changes needed from your provided version) ---
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
    """
    Creates DataLoader for Priming Effect evaluation, with optional sampling.
    """
    csv_path_obj = Path(csv_path)
    logger.info(f"Creating priming dataloader for: {csv_path_obj.name}")
    logger.info(f"Params: batch_size={batch_size}, max_samples={max_samples}, seed={seed}")

    # Pass tokenizer as it might be used by load_and_process_priming_data,
    # though in this version it's not strictly necessary for its direct operations.
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
        except ValueError as e:
            logger.error(f"Error during sampling (k={max_samples}, n={original_size}): {e}. Using full set.")
            # final_processed_data remains processed_data
    elif max_samples > 0:
         logger.info(f"Max_samples ({max_samples:,}) >= processed items ({original_size:,}). Using all processed items.")
    else:
         logger.info(f"Max_samples <= 0. Using all {original_size:,} processed items.")

    if not final_processed_data:
         logger.warning(f"Data list is empty after sampling for {csv_path_obj.name}. Returning None.")
         return None

    dataset = PrimingEvaluationDataset(final_processed_data)
    collate_fn_partial = partial(collate_priming_eval_batch, tokenizer=tokenizer, join_string=delimiter + " ", max_length=max_length)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collate_fn_partial,
        num_workers=num_workers,
        shuffle=False,
        **kwargs
    )

    logger.info(f"Priming Effect DataLoader created for {csv_path_obj.name} with {len(dataset)} items.")
    return dataloader