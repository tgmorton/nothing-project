# train_tokenizer.py (Version 2 - Flexible Glob Input)
import os
import glob
import argparse
from pathlib import Path

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="Train a ByteLevel BPE tokenizer using a glob pattern for input files.")
parser.add_argument(
    "--input_glob",  # Changed from --input_dir
    type=str,
    required=True,
    # Updated help text to clarify it's a glob pattern
    help="Glob pattern for finding input *.train files (e.g., 'data/**/*.train', '../data/*.train'). "
         "IMPORTANT: Quote the pattern on the command line if it contains wildcards (*, ?, **).",
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory where the trained tokenizer files will be saved.",
)
parser.add_argument(
    "--vocab_size",
    type=int,
    default=30000,
    help="Target vocabulary size for the tokenizer.",
)
parser.add_argument(
    "--min_frequency",
    type=int,
    default=2,
    help="Minimum frequency for pairs to be merged.",
)
args = parser.parse_args()

# --- 2. Configuration & Setup ---
INPUT_GLOB_PATTERN = args.input_glob # Use the glob pattern directly
TOKENIZER_OUTPUT_DIR = args.output_dir
VOCAB_SIZE = args.vocab_size
MIN_FREQUENCY = args.min_frequency
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|padding|>",
    "[0]"
    # Add any other custom special tokens here if needed, e.g., "[0]"
]

# Create output directory if it doesn't exist
Path(TOKENIZER_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# --- 3. Find Training Files ---
# Use the provided glob pattern directly
print(f"Searching for files matching glob pattern: {INPUT_GLOB_PATTERN}")
# Use recursive=True to support the '**' wildcard if present in the user's pattern
train_files = glob.glob(INPUT_GLOB_PATTERN, recursive=True)

if not train_files:
    # Updated error message
    raise FileNotFoundError(f"No files found matching glob pattern: '{INPUT_GLOB_PATTERN}'. Please check the pattern and ensure it's quoted correctly if using wildcards.")

print(f"Found {len(train_files)} training files.")
# print("Files found:", train_files) # Uncomment to list files if needed for debugging

# --- 4. Train BPE Tokenizer ---
try:
    from tokenizers import ByteLevelBPETokenizer
except ImportError:
    print("Error: `tokenizers` library not found. Please install it: pip install tokenizers")
    exit(1)

print("\n--- Training Tokenizer ---")
tokenizer = ByteLevelBPETokenizer()

# Pass the list of files found by glob to the trainer
tokenizer.train(
    files=train_files,
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQUENCY,
    special_tokens=SPECIAL_TOKENS
)

print("Tokenizer training complete.")

# --- 5. Save Tokenizer ---

# Save basic vocab/merges files
tokenizer.save_model(TOKENIZER_OUTPUT_DIR)
print(f"Tokenizer vocab.json and merges.txt saved to: {TOKENIZER_OUTPUT_DIR}")

# Save in transformers format for easier loading (recommended)
try:
    from transformers import PreTrainedTokenizerFast

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|padding|>",
    )
    fast_tokenizer.padding_side = 'right'
    fast_tokenizer.save_pretrained(TOKENIZER_OUTPUT_DIR)
    print(f"Transformers compatible tokenizer files (tokenizer.json etc.) also saved to: {TOKENIZER_OUTPUT_DIR}")

except ImportError:
    print("\nNote: `transformers` library not found. Skipping saving in transformers format.")
    print("You can still use the vocab.json and merges.txt files, but loading might require manual instantiation later.")
except Exception as e:
    print(f"\nWarning: Could not save tokenizer in transformers format. Error: {e}")


print("\n--- Tokenizer Training Script Finished ---")