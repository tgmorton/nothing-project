# src/priming_evaluation/data_loader.py (Revised with Sampling)

import logging
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import random # Import the random module for sampling

import pandas as pd
import torch
# Import numpy if you prefer np.random.choice, otherwise random.sample is fine
# import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, SequentialSampler # Add SequentialSampler import
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
    tokenizer: PreTrainedTokenizer, # Keep tokenizer if needed for validation during processing
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
            # Basic validation for empty/NaN strings
            if not all(s and isinstance(s, str) and s.strip() and s.lower() != 'nan' for s in [prime_x_sent, prime_y_sent, target_x_sent, target_y_sent]):
                 # logger.debug(f"Skipping row {index} due to empty/NaN field in {csv_filename}")
                 continue

            # Add both alternation directions
            processed_data.append({"target_sentence": target_x_sent, "congruent_prime": prime_x_sent, "incongruent_prime": prime_y_sent, "target_structure": target_col_x, "congruent_prime_structure": prime_col_x, "incongruent_prime_structure": prime_col_y, "source_csv": csv_filename, "csv_row": index})
            processed_data.append({"target_sentence": target_y_sent, "congruent_prime": prime_y_sent, "incongruent_prime": prime_x_sent, "target_structure": target_col_y, "congruent_prime_structure": prime_col_y, "incongruent_prime_structure": prime_col_x, "source_csv": csv_filename, "csv_row": index})
            items_created_count += 2
        except Exception as e: logger.warning(f"Error processing row {index} in {csv_filename}: {e}. Skip."); continue
    logger.info(f"Finished processing {csv_filename}. Created {items_created_count} valid items from {len(df)} rows.");
    if items_created_count == 0: logger.warning(f"No valid items processed from {csv_filename}.")
    return processed_data


# --- REVISED collate_priming_eval_batch (No changes needed from provided version) ---
# (Keep the collate function exactly as you provided it in the previous turn)
def collate_priming_eval_batch(
    batch: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    join_string: str = ". ", # String used to join prime and target
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Collates a batch for Priming Effect evaluation. Uses revised index logic.
    """
    batch = [item for item in batch if item is not None]
    if not batch: return {}

    max_length = max_length or getattr(tokenizer, 'model_max_length', None)
    collated_batch = defaultdict(list)
    items_to_keep_indices = [] # Track indices of items that pass validation

    # --- Determine Tokenizer Properties ---
    dummy_tokens = tokenizer("test", add_special_tokens=True)['input_ids']
    dummy_tokens_no_special = tokenizer("test", add_special_tokens=False)['input_ids']
    adds_bos = tokenizer.bos_token_id is not None and len(dummy_tokens) > len(dummy_tokens_no_special) and dummy_tokens[0] == tokenizer.bos_token_id
    adds_eos = tokenizer.eos_token_id is not None and len(dummy_tokens) > len(dummy_tokens_no_special) and dummy_tokens[-1] == tokenizer.eos_token_id
    bos_offset = 1 if adds_bos else 0
    eos_offset = 1 if adds_eos else 0

    # --- Process each item ---
    for idx, item in enumerate(batch):
        prime_congruent_context = item['congruent_prime'] + join_string
        prime_incongruent_context = item['incongruent_prime'] + join_string
        tokens_prime_congruent_context = tokenizer(prime_congruent_context, add_special_tokens=False, return_attention_mask=False)['input_ids']
        tokens_prime_incongruent_context = tokenizer(prime_incongruent_context, add_special_tokens=False, return_attention_mask=False)['input_ids']
        len_prime_congruent_context = len(tokens_prime_congruent_context)
        len_prime_incongruent_context = len(tokens_prime_incongruent_context)

        full_congruent_text = prime_congruent_context + item['target_sentence']
        full_incongruent_text = prime_incongruent_context + item['target_sentence']
        congruent_encoding = tokenizer(full_congruent_text, add_special_tokens=True, truncation=True if max_length else False, max_length=max_length, return_attention_mask=True)
        incongruent_encoding = tokenizer(full_incongruent_text, add_special_tokens=True, truncation=True if max_length else False, max_length=max_length, return_attention_mask=True)

        target_start_congruent = bos_offset + len_prime_congruent_context
        target_start_incongruent = bos_offset + len_prime_incongruent_context
        target_end_congruent = len(congruent_encoding['input_ids']) - eos_offset
        target_end_incongruent = len(incongruent_encoding['input_ids']) - eos_offset

        valid_congruent = (0 <= target_start_congruent < target_end_congruent <= len(congruent_encoding['input_ids']))
        valid_incongruent = (0 <= target_start_incongruent < target_end_incongruent <= len(incongruent_encoding['input_ids']))

        if valid_congruent and valid_incongruent:
            items_to_keep_indices.append(idx)
            collated_batch["_congruent_encoding"].append(congruent_encoding)
            collated_batch["_incongruent_encoding"].append(incongruent_encoding)
            collated_batch["target_start_congruent"].append(target_start_congruent)
            collated_batch["target_end_congruent"].append(target_end_congruent)
            collated_batch["target_start_incongruent"].append(target_start_incongruent)
            collated_batch["target_end_incongruent"].append(target_end_incongruent)
            collated_batch["target_structure"].append(item['target_structure'])
            collated_batch["source_csv"].append(item['source_csv'])
            collated_batch["csv_row"].append(item['csv_row'])
        else:
             logger.warning(f"Invalid target indices calculated for row {item.get('csv_row', 'N/A')} from {item.get('source_csv', 'N/A')}. Skipping item.")

    if not items_to_keep_indices:
        logger.warning("Collate function resulted in an empty batch after index validation.")
        return {}

    # --- Padding ---
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collated_batch['congruent_input_ids'] = pad_batch_encodings(collated_batch["_congruent_encoding"], pad_token_id, key='input_ids')
    collated_batch['congruent_attention_mask'] = pad_batch_encodings(collated_batch["_congruent_encoding"], 0, key='attention_mask')
    collated_batch['incongruent_input_ids'] = pad_batch_encodings(collated_batch["_incongruent_encoding"], pad_token_id, key='input_ids')
    collated_batch['incongruent_attention_mask'] = pad_batch_encodings(collated_batch["_incongruent_encoding"], 0, key='attention_mask')

    # --- Create Labels Tensor ---
    max_len_batch = max(collated_batch['congruent_input_ids'].shape[1], collated_batch['incongruent_input_ids'].shape[1])
    labels_list = []
    for i in range(len(items_to_keep_indices)):
         input_ids = collated_batch["_congruent_encoding"][i]['input_ids']
         start_idx = collated_batch["target_start_congruent"][i]
         end_idx = collated_batch["target_end_congruent"][i]
         label = torch.full((len(input_ids),), -100, dtype=torch.long)
         # Ensure indices are valid before assignment
         if 0 <= start_idx < end_idx <= len(label):
             label[start_idx:end_idx] = torch.tensor(input_ids[start_idx:end_idx], dtype=torch.long)
         else:
             # This case should ideally not happen if validation above worked, but as a safeguard:
             logger.error(f"Label index mismatch AFTER validation. Item {i}, "
                          f"Start={start_idx}, End={end_idx}, LabelLen={len(label)}. Setting empty label.")
             # Keep label as all -100
         labels_list.append(label)

    collated_batch['labels'] = pad_sequence(labels_list, batch_first=True, padding_value=-100)

    # Convert index lists to tensors
    for key in ["target_start_congruent", "target_end_congruent", "target_start_incongruent", "target_end_incongruent"]:
        if key in collated_batch:
             collated_batch[key] = torch.tensor(collated_batch[key], dtype=torch.long)

    del collated_batch["_congruent_encoding"]
    del collated_batch["_incongruent_encoding"]

    return dict(collated_batch)


# --- pad_batch_encodings (No changes needed) ---
def pad_batch_encodings(encodings: List[BatchEncoding], pad_value: int, key: str = 'input_ids') -> torch.Tensor:
    """ Pads a list of BatchEncoding outputs. """
    sequences = [torch.tensor(enc[key]) for enc in encodings]; return pad_sequence(sequences, batch_first=True, padding_value=pad_value)


# --- REVISED create_priming_dataloader (with Sampling) ---
def create_priming_dataloader(
    csv_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    delimiter: str = ".",
    num_workers: int = 0,
    max_length: Optional[int] = None,
    # --- ADD SAMPLING PARAMETERS ---
    max_samples: int = -1, # Max samples per file, <=0 means no limit
    seed: int = 42,        # Seed for reproducible sampling
    # --- END SAMPLING PARAMETERS ---
    **kwargs # Keep kwargs for DataLoader flexibility
) -> Optional[DataLoader]:
    """
    Creates DataLoader for Priming Effect evaluation, with optional sampling.
    """
    csv_path_obj = Path(csv_path)
    logger.info(f"Creating priming dataloader for: {csv_path_obj.name}")
    logger.info(f"Params: batch_size={batch_size}, max_samples={max_samples}, seed={seed}")

    # 1. Load and process ALL data from the CSV first
    # Pass tokenizer only if load_and_process needs it for validation (seems it doesn't currently)
    processed_data = load_and_process_priming_data(csv_path=csv_path_obj, tokenizer=tokenizer, delimiter=delimiter)

    # Check if processing yielded any data
    if not processed_data:
        logger.warning(f"No data processed from {csv_path_obj.name}. Returning None for DataLoader.")
        return None

    original_size = len(processed_data)
    logger.info(f"Initially processed {original_size:,} items from {csv_path_obj.name}.")

    # --- START SAMPLING LOGIC (Applied AFTER processing) ---
    final_processed_data = processed_data # Start with the full list

    if max_samples > 0 and max_samples < original_size:
        logger.info(f"Sampling {max_samples:,} items from the processed data (seed: {seed}).")
        # Use random.sample for efficient sampling from the list
        random.seed(seed) # Seed the random module
        try:
            final_processed_data = random.sample(processed_data, k=max_samples)
            logger.info(f"Using subset: {len(final_processed_data):,} items.")
        except ValueError as e:
            logger.error(f"Error during sampling (k={max_samples}, n={original_size}): {e}. Using full set.")
            final_processed_data = processed_data # Fallback to full data on error
    elif max_samples > 0:
         logger.info(f"Max_samples ({max_samples:,}) >= processed items ({original_size:,}). Using all processed items.")
         # final_processed_data is already set to processed_data
    else:
         logger.info(f"Max_samples <= 0. Using all {original_size:,} processed items.")
         # final_processed_data is already set to processed_data
    # --- END SAMPLING LOGIC ---

    # Check again if we have data after potential sampling
    if not final_processed_data:
         logger.warning(f"Data list is empty after sampling for {csv_path_obj.name}. Returning None.")
         return None

    # 2. Create Dataset from the (potentially sampled) processed data
    dataset = PrimingEvaluationDataset(final_processed_data)

    # 3. Prepare collate function
    collate_fn_partial = partial(collate_priming_eval_batch, tokenizer=tokenizer, join_string=delimiter + " ", max_length=max_length)

    # 4. Create DataLoader
    # Use SequentialSampler for evaluation to ensure order is consistent
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler, # Use SequentialSampler
        batch_size=batch_size,
        collate_fn=collate_fn_partial,
        num_workers=num_workers,
        shuffle=False, # Shuffle should be False for evaluation sampler
        **kwargs # Pass any remaining DataLoader kwargs
    )

    logger.info(f"Priming Effect DataLoader created for {csv_path_obj.name} with {len(dataset)} items.")
    return dataloader