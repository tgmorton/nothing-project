# src/priming_evaluation/data_loader.py (Corrected for key name consistency)

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
from transformers import PreTrainedTokenizer

# (The file is mostly the same as the last version, with one key change in collate_for_new_evaluator)
logger = logging.getLogger(__name__)


def get_structure_alternations(columns: List[str]) -> Optional[Tuple[str, str]]:
    structures = set()
    for col in columns:
        if col.startswith('p') or col.startswith('t'):
            structure_name = col[1:]
            if structure_name:
                structures.add(structure_name)
    if len(structures) == 2:
        return tuple(sorted(list(structures)))
    else:
        logger.warning(f"Could not determine exactly two structures. Found: {structures} from columns: {columns}")
        return None


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


def load_and_process_priming_data(csv_path: Path) -> List[Dict[str, Any]]:
    processed_data = []
    csv_filename = csv_path.name
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
    except Exception as e:
        logger.error(f"Error loading or processing CSV {csv_filename}: {e}");
        return []

    alternation = get_structure_alternations(list(df.columns))
    if alternation is None:
        logger.error(f"Could not process {csv_filename}.")
        return []

    struct1, struct2 = alternation
    logger.info(f"Detected alternation for {csv_filename}: '{struct1}' vs '{struct2}'")

    p_col1, p_col2 = f'p{struct1}', f'p{struct2}'
    t_col1, t_col2 = f't{struct1}', f't{struct2}'
    required_cols = [p_col1, p_col2, t_col1, t_col2]

    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error(f"CSV {csv_filename} is missing constructed columns: {missing}")
        return []

    logger.info(f"Processing {len(df)} rows from {csv_filename}...")
    for index, row in df.iterrows():
        try:
            prime1, prime2 = str(row[p_col1]), str(row[p_col2])
            target1, target2 = str(row[t_col1]), str(row[t_col2])

            if not all(s and s.lower() != 'nan' for s in [prime1, prime2, target1, target2]):
                logger.warning(f"Skipping row {index} in {csv_filename} due to empty or 'nan' value.")
                continue

            processed_data.append({
                'congruent_prime': prime1, 'incongruent_prime': prime2,
                'congruent_target': target1, 'incongruent_target': target2,
                'target_structure': struct1, 'source_csv': csv_filename, 'csv_row': index
            })
            processed_data.append({
                'congruent_prime': prime2, 'incongruent_prime': prime1,
                'congruent_target': target2, 'incongruent_target': target1,
                'target_structure': struct2, 'source_csv': csv_filename, 'csv_row': index
            })
        except (KeyError, TypeError) as e:
            logger.warning(f"Skipping row {index} in {csv_filename} due to error: {e}")
            continue

    logger.info(f"Finished processing {csv_filename}. Created {len(processed_data)} valid items.")
    return processed_data


def collate_for_new_evaluator(
        batch: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
) -> Dict[str, Any]:
    batch = [item for item in batch if item is not None]
    if not batch: return {}

    collated_batch = defaultdict(list)

    sentence_keys = ['congruent_prime', 'incongruent_prime', 'congruent_target', 'incongruent_target']

    for key in sentence_keys:
        sentences = [item[key] for item in batch]
        tokenized_output = tokenizer(sentences, add_special_tokens=False)
        collated_batch[f'_{key}_tokens'] = tokenized_output['input_ids']

    bos_offset = 1
    collated_batch['con_target_start_in_con_prime_context'] = [len(tokens) + bos_offset for tokens in
                                                               collated_batch['_congruent_prime_tokens']]
    collated_batch['incon_target_start_in_con_prime_context'] = [len(tokens) + bos_offset for tokens in
                                                                 collated_batch['_congruent_prime_tokens']]
    collated_batch['con_target_start_in_incon_prime_context'] = [len(tokens) + bos_offset for tokens in
                                                                 collated_batch['_incongruent_prime_tokens']]
    collated_batch['incon_target_start_in_incon_prime_context'] = [len(tokens) + bos_offset for tokens in
                                                                   collated_batch['_incongruent_prime_tokens']]

    pad_token_id = tokenizer.pad_token_id
    for key in sentence_keys:
        sequences = [torch.tensor(tokens) for tokens in collated_batch[f'_{key}_tokens']]
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_token_id)
        # --- KEY CHANGE: Use full, consistent names ---
        final_key = key
        # --- End of Change ---
        collated_batch[f'{final_key}_input_ids'] = padded_sequences
        del collated_batch[f'_{key}_tokens']

    collated_batch['target_structure'] = [item['target_structure'] for item in batch]

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