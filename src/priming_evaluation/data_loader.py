# src/priming_evaluation/data_loader.py (Revised for the new evaluator)

import logging
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional
from pathlib import Path
import random

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


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


def load_and_process_priming_data(
        csv_path: Path,
) -> List[Dict[str, Any]]:
    """
    Loads priming data from a CSV.
    Assumes each row contains the four necessary sentence components.
    e.g., columns 'congruent_prime', 'incongruent_prime', 'congruent_target', 'incongruent_target'
    """
    processed_data = []
    csv_filename = csv_path.name
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
    except Exception as e:
        logger.error(f"Error loading or processing CSV {csv_filename}: {e}");
        return []

    # These are the required columns for the new evaluator
    required_cols = ['congruent_prime', 'incongruent_prime', 'congruent_target', 'incongruent_target']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error(f"CSV {csv_filename} missing required columns for the new evaluator: {missing}")
        return []

    logger.info(f"Processing {len(df)} rows from {csv_filename}...")
    for index, row in df.iterrows():
        item = {
            'congruent_prime': str(row['congruent_prime']),
            'incongruent_prime': str(row['incongruent_prime']),
            'congruent_target': str(row['congruent_target']),
            'incongruent_target': str(row['incongruent_target']),
            'target_structure': str(row.get('target_structure', 'default')),  # Optional: for grouping results
            'source_csv': csv_filename,
            'csv_row': index
        }
        # Basic validation to skip empty/NaN rows
        if all(item[key] and item[key].lower() != 'nan' for key in required_cols):
            processed_data.append(item)
        else:
            logger.warning(f"Skipping row {index} in {csv_filename} due to empty or 'nan' value.")

    logger.info(f"Finished processing {csv_filename}. Created {len(processed_data)} valid items.")
    return processed_data


def collate_for_new_evaluator(
        batch: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
) -> Dict[str, Any]:
    """
    NEW collate function.
    Tokenizes each of the 4 sentence parts separately and prepares the batch
    with the keys expected by the new evaluator script.
    """
    batch = [item for item in batch if item is not None]
    if not batch: return {}

    collated_batch = defaultdict(list)

    # Keys for the four sentence parts
    sentence_keys = ['congruent_prime', 'incongruent_prime', 'congruent_target', 'incongruent_target']

    # --- 1. Tokenize all sentence parts for each item in the batch ---
    for key in sentence_keys:
        # Tokenize sentences without special tokens; the evaluator will handle them
        sentences = [item[key] for item in batch]
        tokenized_output = tokenizer(sentences, add_special_tokens=False)
        # Store the raw token lists for length calculations
        collated_batch[f'_{key}_tokens'] = tokenized_output['input_ids']

    # --- 2. Calculate target start indices ---
    bos_offset = 1  # The new evaluator assumes a BOS token will be added
    collated_batch['con_target_start_in_con_prime_context'] = [len(tokens) + bos_offset for tokens in
                                                               collated_batch['_congruent_prime_tokens']]
    collated_batch['incon_target_start_in_con_prime_context'] = [len(tokens) + bos_offset for tokens in
                                                                 collated_batch['_congruent_prime_tokens']]
    collated_batch['con_target_start_in_incon_prime_context'] = [len(tokens) + bos_offset for tokens in
                                                                 collated_batch['_incongruent_prime_tokens']]
    collated_batch['incon_target_start_in_incon_prime_context'] = [len(tokens) + bos_offset for tokens in
                                                                   collated_batch['_incongruent_prime_tokens']]

    # --- 3. Pad the token lists to create final tensors ---
    pad_token_id = tokenizer.pad_token_id
    for key in sentence_keys:
        sequences = [torch.tensor(tokens) for tokens in collated_batch[f'_{key}_tokens']]
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_token_id)
        # Rename to the keys the evaluator expects
        final_key = key.replace('_', 'T_') if 'target' in key else key.replace('_', 'P_')
        collated_batch[f'{final_key}_input_ids'] = padded_sequences
        # Clean up temporary token lists
        del collated_batch[f'_{key}_tokens']

    # --- 4. Add other necessary metadata ---
    collated_batch['target_structure'] = [item['target_structure'] for item in batch]

    # Convert index lists to tensors
    index_keys = [
        'con_target_start_in_con_prime_context', 'incon_target_start_in_con_prime_context',
        'con_target_start_in_incon_prime_context', 'incon_target_start_in_incon_prime_context'
    ]
    for key in index_keys:
        collated_batch[key] = torch.tensor(collated_batch[key], dtype=torch.long)

    return dict(collated_batch)


def create_priming_dataloader(
        csv_path: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        num_workers: int = 0,
        max_samples: int = -1,
        seed: int = 42,
        **kwargs
) -> Optional[DataLoader]:
    csv_path_obj = Path(csv_path)
    logger.info(f"Creating priming dataloader for: {csv_path_obj.name}")
    logger.info(f"Params: batch_size={batch_size}, max_samples={max_samples}, seed={seed}")

    processed_data = load_and_process_priming_data(csv_path=csv_path_obj)

    if not processed_data:
        logger.warning(f"No data processed from {csv_path_obj.name}. Returning None for DataLoader.")
        return None

    # Handle sampling
    if max_samples > 0 and len(processed_data) > max_samples:
        logger.info(f"Sampling {max_samples:,} items from {len(processed_data):,} (seed: {seed}).")
        random.seed(seed)
        processed_data = random.sample(processed_data, k=max_samples)

    dataset = PrimingEvaluationDataset(processed_data)
    collate_fn_partial = partial(collate_for_new_evaluator, tokenizer=tokenizer)

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collate_fn_partial,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=kwargs.get('pin_memory', True)
    )

    logger.info(f"Priming DataLoader created for {csv_path_obj.name} with {len(dataset)} items.")
    return dataloader