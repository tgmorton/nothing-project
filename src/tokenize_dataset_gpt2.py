import argparse
import logging
import os
from pathlib import Path
import time # Optional: for timing steps

# Use try-except for optional dependencies if you want graceful failure
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError:
    print("Please install datasets and transformers: pip install datasets transformers")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Argument Parsing ---
#   input_data_files: str - Path to raw input files (default. '../data/raw/text_data/train_10M/*.train')

def parse_args():
    """Parses command-line arguments for data tokenization."""
    parser = argparse.ArgumentParser(description="Tokenize raw text data for language model training.")
    parser.add_argument(
        "--input_data_files", # Changed name slightly for clarity with load_dataset
        type=str,
        required=True,
        help="Path or glob pattern to the raw input text file(s) (e.g., 'data/*.txt', 'data/corpus.txt')."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Identifier for the tokenizer from Hugging Face Hub (e.g., 'gpt2') or path to local tokenizer files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the processed dataset will be saved."
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=1024, # Common sequence length for GPT-2 style models
        help="The target sequence length for chunking the tokenized text."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None, # Use None to let datasets library use os.cpu_count()
        help="Number of CPU cores to use for parallel processing. Defaults to all available cores."
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory if it already exists."
    )
    # Optional: Add arguments for dataset subset (e.g., --dataset_split train) if needed

    args = parser.parse_args()
    return args

# --- Core Processing Functions ---

def tokenize_function(examples, tokenizer):
    """Applies the tokenizer to a batch of text examples."""
    # Tokenize without padding or truncation initially, as we'll group texts later
    # Adjust tokenizer arguments if needed based on specific tokenizer behavior
    return tokenizer(examples["text"], return_attention_mask=False)

def group_texts(examples, sequence_length):
    """
    Groups tokenized texts into fixed-length chunks (packing).
    This is a standard function often used in LM data preparation.
    """
    # Concatenate all texts from the batch
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # Drop the small remainder to ensure chunks are of equal size
    # Alternative: handle remainder separately if needed
    if total_length >= sequence_length:
        total_length = (total_length // sequence_length) * sequence_length

    # Split by chunks of sequence_length
    result = {
        k: [t[i : i + sequence_length] for i in range(0, total_length, sequence_length)]
        for k, t in concatenated_examples.items()
    }
    # Note: This simple version creates 'input_ids' key. If tokenizer_function
    # returned other keys ('attention_mask', 'token_type_ids'), they'd be chunked here too.
    # For basic LM pretraining, only 'input_ids' are typically needed after packing.
    # Labels are usually created dynamically during training (input_ids shifted).
    # result["labels"] = result["input_ids"].copy() # Uncomment if labels needed in dataset
    return result

# --- Main Execution ---
def main():
    """Orchestrates the data tokenization and processing pipeline."""
    args = parse_args()
    logger.info(f"Starting tokenization process with args: {args}")

    # --- Validate Paths and Setup ---
    output_path = Path(args.output_dir)
    if output_path.exists() and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    output_path.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Tokenizer ---
    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True) # Use fast tokenizer if available
    # TODO: Check if tokenizer needs specific setup (e.g., pad token) although often not needed for packing

    # --- 2. Load Raw Dataset ---
    logger.info(f"Loading raw text data from: {args.input_data_files}")
    # Using 'text' type assumes one document per line in the input file(s)
    # Adjust loader if using json, csv, etc.
    try:
        raw_datasets = load_dataset('text', data_files=args.input_data_files, split='train') # Assuming 'train' split exists or is inferred
        logger.info(f"Raw dataset loaded: {raw_datasets}")
    except Exception as e:
        logger.error(f"Failed to load dataset from {args.input_data_files}: {e}")
        raise

    # --- 3. Tokenize Data ---
    logger.info("Applying tokenizer to the dataset...")
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        fn_kwargs={'tokenizer': tokenizer}, # Pass tokenizer to the function
        batched=True,
        num_proc=args.num_workers,
        remove_columns=raw_datasets.column_names, # Remove original 'text' column
        desc="Running tokenizer on dataset",
    )
    logger.info(f"Tokenization complete. Dataset info: {tokenized_datasets}")
    logger.info(f"Example tokenized sample: {tokenized_datasets[0]}") # Log first sample

    # --- 4. Group/Chunk Data ---
    logger.info(f"Grouping texts into chunks of length {args.sequence_length}...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        fn_kwargs={'sequence_length': args.sequence_length}, # Pass sequence length
        batched=True,
        num_proc=args.num_workers,
        desc=f"Grouping texts into chunks of {args.sequence_length}",
    )
    logger.info(f"Grouping complete. Final dataset info: {lm_datasets}")
    logger.info(f"Example final sample: {lm_datasets[0]}") # Log first sample

    # --- 5. Save Processed Dataset ---
    logger.info(f"Saving processed dataset to {args.output_dir}...")
    try:
        lm_datasets.save_to_disk(args.output_dir)
        logger.info("Dataset saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save dataset to {args.output_dir}: {e}")
        raise

    logger.info("Tokenization process finished successfully!")

if __name__ == "__main__":
    main()