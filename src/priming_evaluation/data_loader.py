# src/priming_evaluation/data_loader.py (Revised Collate Function)

import logging
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
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

# --- load_and_process_priming_data (No changes needed from previous version) ---
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
            if not all(s and s.lower() != 'nan' for s in [prime_x_sent, prime_y_sent, target_x_sent, target_y_sent]): continue

            processed_data.append({"target_sentence": target_x_sent, "congruent_prime": prime_x_sent, "incongruent_prime": prime_y_sent, "target_structure": target_col_x, "congruent_prime_structure": prime_col_x, "incongruent_prime_structure": prime_col_y, "source_csv": csv_filename, "csv_row": index})
            processed_data.append({"target_sentence": target_y_sent, "congruent_prime": prime_y_sent, "incongruent_prime": prime_x_sent, "target_structure": target_col_y, "congruent_prime_structure": prime_col_y, "incongruent_prime_structure": prime_col_x, "source_csv": csv_filename, "csv_row": index})
            items_created_count += 2
        except Exception as e: logger.warning(f"Error processing row {index} in {csv_filename}: {e}. Skip."); continue
    logger.info(f"Finished processing {csv_filename}. Created {items_created_count} items.");
    if items_created_count == 0: logger.warning(f"No valid items processed from {csv_filename}.")
    return processed_data


# --- REVISED collate_priming_eval_batch ---
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
    # Check if tokenizer adds BOS/EOS by default. This affects index calculation.
    # Tokenize a dummy string to check for added special tokens.
    dummy_tokens = tokenizer("test", add_special_tokens=True)['input_ids']
    dummy_tokens_no_special = tokenizer("test", add_special_tokens=False)['input_ids']
    adds_bos = tokenizer.bos_token_id is not None and len(dummy_tokens) > len(dummy_tokens_no_special) and dummy_tokens[0] == tokenizer.bos_token_id
    adds_eos = tokenizer.eos_token_id is not None and len(dummy_tokens) > len(dummy_tokens_no_special) and dummy_tokens[-1] == tokenizer.eos_token_id
    bos_offset = 1 if adds_bos else 0
    eos_offset = 1 if adds_eos else 0
    # logger.debug(f"Tokenizer properties: adds_bos={adds_bos}, adds_eos={adds_eos}")

    # --- Process each item ---
    for idx, item in enumerate(batch):
        # --- Tokenize components to find prime context length ---
        prime_congruent_context = item['congruent_prime'] + join_string
        prime_incongruent_context = item['incongruent_prime'] + join_string

        tokens_prime_congruent_context = tokenizer(prime_congruent_context, add_special_tokens=False, return_attention_mask=False)['input_ids']
        tokens_prime_incongruent_context = tokenizer(prime_incongruent_context, add_special_tokens=False, return_attention_mask=False)['input_ids']

        len_prime_congruent_context = len(tokens_prime_congruent_context)
        len_prime_incongruent_context = len(tokens_prime_incongruent_context)

        # --- Tokenize full sequences with truncation & special tokens ---
        full_congruent_text = prime_congruent_context + item['target_sentence']
        full_incongruent_text = prime_incongruent_context + item['target_sentence']

        congruent_encoding = tokenizer(full_congruent_text, add_special_tokens=True, truncation=True if max_length else False, max_length=max_length, return_attention_mask=True)
        incongruent_encoding = tokenizer(full_incongruent_text, add_special_tokens=True, truncation=True if max_length else False, max_length=max_length, return_attention_mask=True)

        # --- Calculate target indices based on prime context length ---
        # Target starts after the prime context tokens (and potential BOS)
        target_start_congruent = bos_offset + len_prime_congruent_context
        target_start_incongruent = bos_offset + len_prime_incongruent_context

        # Target ends at the end of the sequence (before potential EOS)
        target_end_congruent = len(congruent_encoding['input_ids']) - eos_offset
        target_end_incongruent = len(incongruent_encoding['input_ids']) - eos_offset

        # --- Validate indices ---
        # Check if start < end and indices are within the bounds of the tokenized sequence length
        valid_congruent = (0 <= target_start_congruent < target_end_congruent <= len(congruent_encoding['input_ids']))
        valid_incongruent = (0 <= target_start_incongruent < target_end_incongruent <= len(incongruent_encoding['input_ids']))

        if valid_congruent and valid_incongruent:
            items_to_keep_indices.append(idx) # Mark this item as valid
            # Store encodings and indices for valid items
            collated_batch["_congruent_encoding"].append(congruent_encoding)
            collated_batch["_incongruent_encoding"].append(incongruent_encoding)
            collated_batch["target_start_congruent"].append(target_start_congruent)
            collated_batch["target_end_congruent"].append(target_end_congruent)
            collated_batch["target_start_incongruent"].append(target_start_incongruent)
            collated_batch["target_end_incongruent"].append(target_end_incongruent)
            # Store metadata only for items we keep
            collated_batch["target_structure"].append(item['target_structure'])
            collated_batch["source_csv"].append(item['source_csv'])
            collated_batch["csv_row"].append(item['csv_row'])
        else:
            # Log skipping
            logger.warning(f"Invalid target indices calculated for row {item.get('csv_row', 'N/A')} from {item.get('source_csv', 'N/A')}. Skipping item.")
            # Optionally log more details for debugging
            # logger.debug(f" Invalid Item details: Target '{item['target_structure']}'. "
            #              f"Cong: Start={target_start_congruent}, End={target_end_congruent}, ValidEnd={len(congruent_encoding['input_ids'])}, Valid={valid_congruent}. "
            #              f"Incong: Start={target_start_incongruent}, End={target_end_incongruent}, ValidEnd={len(incongruent_encoding['input_ids'])}, Valid={valid_incongruent}.")


    # If all items in the batch were skipped
    if not items_to_keep_indices:
        logger.warning("Collate function resulted in an empty batch after index validation.")
        return {}

    # --- Padding ---
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Pad the main input sequences using only the valid items
    collated_batch['congruent_input_ids'] = pad_batch_encodings(collated_batch["_congruent_encoding"], pad_token_id, key='input_ids')
    collated_batch['congruent_attention_mask'] = pad_batch_encodings(collated_batch["_congruent_encoding"], 0, key='attention_mask')
    collated_batch['incongruent_input_ids'] = pad_batch_encodings(collated_batch["_incongruent_encoding"], pad_token_id, key='input_ids')
    collated_batch['incongruent_attention_mask'] = pad_batch_encodings(collated_batch["_incongruent_encoding"], 0, key='attention_mask')

    # --- Create Labels Tensor ---
    # Labels tensor should have the same shape as the *longest* input sequence in the batch
    # Max length determined by the padding of congruent/incongruent inputs
    max_len_batch = max(collated_batch['congruent_input_ids'].shape[1], collated_batch['incongruent_input_ids'].shape[1])
    labels_list = []
    for i in range(len(items_to_keep_indices)): # Iterate through valid items
         # Use congruent input IDs as base (could use incongruent too)
         input_ids = collated_batch["_congruent_encoding"][i]['input_ids']
         start_idx = collated_batch["target_start_congruent"][i]
         end_idx = collated_batch["target_end_congruent"][i]

         # Create label sequence: -100 everywhere except target region
         label = torch.full((len(input_ids),), -100, dtype=torch.long)
         label[start_idx:end_idx] = torch.tensor(input_ids[start_idx:end_idx], dtype=torch.long)
         labels_list.append(label)

    # Pad the labels list
    collated_batch['labels'] = pad_sequence(labels_list, batch_first=True, padding_value=-100)

    # Convert index lists to tensors
    for key in ["target_start_congruent", "target_end_congruent", "target_start_incongruent", "target_end_incongruent"]:
        if key in collated_batch:
             collated_batch[key] = torch.tensor(collated_batch[key], dtype=torch.long)

    # Clean up temporary keys
    del collated_batch["_congruent_encoding"]
    del collated_batch["_incongruent_encoding"]

    return dict(collated_batch)


# --- pad_batch_encodings (No changes needed) ---
def pad_batch_encodings(encodings: List[BatchEncoding], pad_value: int, key: str = 'input_ids') -> torch.Tensor:
    """ Pads a list of BatchEncoding outputs. """
    sequences = [torch.tensor(enc[key]) for enc in encodings]; return pad_sequence(sequences, batch_first=True, padding_value=pad_value)

# --- create_priming_dataloader (No changes needed) ---
def create_priming_dataloader(
    csv_path: str, tokenizer: PreTrainedTokenizer, batch_size: int,
    delimiter: str = ".", num_workers: int = 0, max_length: Optional[int] = None, **kwargs
) -> Optional[DataLoader]:
    """ Creates DataLoader for Priming Effect evaluation. """
    csv_path_obj = Path(csv_path)
    processed_data = load_and_process_priming_data(csv_path=csv_path_obj, tokenizer=tokenizer, delimiter=delimiter)
    if not processed_data: return None
    dataset = PrimingEvaluationDataset(processed_data)
    collate_fn_partial = partial(collate_priming_eval_batch, tokenizer=tokenizer, join_string=delimiter + " ", max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_partial, num_workers=num_workers, shuffle=False, **kwargs)
    logger.info(f"Priming Effect DataLoader created for {csv_path_obj.name} with {len(dataset)} items.")
    return dataloader