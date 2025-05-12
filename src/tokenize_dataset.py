# tokenize_dataset.py
import os
import glob
import json
import argparse
from pathlib import Path

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="Tokenize text data using a pre-trained tokenizer and save as Arrow.")
parser.add_argument(
    "--input_dir",
    type=str,
    required=True,
    help="Directory containing input *.train files (must be the same files used for tokenizer training).",
)
parser.add_argument(
    "--tokenizer_dir",
    type=str,
    required=True,
    help="Directory containing the pre-trained tokenizer files (output of train_tokenizer.py).",
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory where the tokenized Arrow dataset will be saved.",
)
# Optional: Add arguments for num_proc, etc. if needed
# parser.add_argument("--num_proc", type=int, default=None, help="Number of processes for dataset mapping.")

args = parser.parse_args()

# --- 2. Configuration & Setup ---
INPUT_DATA_DIR = args.input_dir
TOKENIZER_PATH = args.tokenizer_dir
ARROW_OUTPUT_DIR = args.output_dir
# NUM_PROC = args.num_proc

# Create output directory if it doesn't exist
Path(ARROW_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# --- 3. Find Input Data Files ---
# It's important to use the same files that the tokenizer was trained on
print(f"Searching for *.train files in: {INPUT_DATA_DIR}")
train_files = glob.glob(os.path.join(INPUT_DATA_DIR, "**/*.train"), recursive=True)

if not train_files:
    raise FileNotFoundError(f"No *.train files found in {INPUT_DATA_DIR}. These should be the same files used to train the tokenizer.")

print(f"Found {len(train_files)} data files to tokenize.")

# --- 4. Load Pre-trained Tokenizer ---
print(f"\n--- Loading Tokenizer from: {TOKENIZER_PATH} ---")
try:
    from transformers import AutoTokenizer
    tokenizer_for_processing = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    print("Loaded tokenizer using AutoTokenizer.")
except ImportError:
    print("Error: `transformers` library not found. Please install it: pip install transformers")
    exit(1)
except Exception as e:
    print(f"Error loading tokenizer with AutoTokenizer: {e}")
    print("Attempting manual load (requires tokenizers library and vocab/merges files).")
    try:
        from tokenizers import ByteLevelBPETokenizer
        vocab_file = os.path.join(TOKENIZER_PATH, "vocab.json")
        merges_file = os.path.join(TOKENIZER_PATH, "merges.txt")
        if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
             raise FileNotFoundError("vocab.json or merges.txt not found for manual loading.")
        tokenizer_for_processing = ByteLevelBPETokenizer(
             vocab=vocab_file,
             merges=merges_file,
        )
        print("Loaded tokenizer manually using ByteLevelBPETokenizer.")
        # Note: Manual loading might require different handling in tokenize_function if not wrapped
    except ImportError:
         print("Error: `tokenizers` library not found for manual fallback. Please install it.")
         exit(1)
    except Exception as manual_e:
         print(f"Error loading tokenizer manually: {manual_e}")
         exit(1)


# --- 5. Load Data using Hugging Face datasets ---
try:
    from datasets import load_dataset
except ImportError:
    print("Error: `datasets` library not found. Please install it: pip install datasets pyarrow")
    exit(1)

print("\n--- Loading Text Data ---")
# 'text' dataset builder reads lines from text files
raw_datasets = load_dataset('text', data_files={'train': train_files})
print("Raw dataset loaded:")
print(raw_datasets)


# --- 6. Tokenize Data ---
print("\n--- Tokenizing Data ---")

# Define the function to tokenize batches of text
def tokenize_function(examples):
    # Assumes tokenizer_for_processing is a loaded HF Tokenizer object
    # (either from AutoTokenizer or wrapped if loaded manually)
    # No padding/truncation applied here, saving raw token sequences.
    if hasattr(tokenizer_for_processing, '__call__'):
        tokenized_output = tokenizer_for_processing(examples['text'])
        return {'input_ids': tokenized_output['input_ids']}
    else:
        # Handle case where tokenizer was loaded manually and isn't callable directly
        # This part might need adjustment based on how manual loading was done
        output_ids = []
        for text_line in examples['text']:
             output_ids.append(tokenizer_for_processing.encode(text_line).ids)
        return {'input_ids': output_ids}


# Apply the tokenization function
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    # num_proc=NUM_PROC, # Optional parallel processing
    remove_columns=raw_datasets["train"].column_names, # Keep only input_ids
    desc="Running tokenizer on dataset",
)

print("Tokenization complete.")
print("Tokenized dataset structure:")
print(tokenized_datasets)
print("Example entry (first 50 token IDs):")
if len(tokenized_datasets['train']) > 0:
    print(tokenized_datasets['train'][0]['input_ids'][:50])
else:
    print("Dataset is empty.")

# --- 7. Save Tokenized Data as Arrow with Metadata ---
print(f"\n--- Saving Tokenized Dataset to Arrow Format in: {ARROW_OUTPUT_DIR} ---")
# save_to_disk handles Arrow conversion AND metadata generation
tokenized_datasets['train'].save_to_disk(ARROW_OUTPUT_DIR)

print("Dataset saved successfully.")
print(f"Check {ARROW_OUTPUT_DIR} for 'data-*.arrow', 'dataset_info.json', and 'state.json'.")

# --- 8. Verify Metadata (Optional Preview) ---
print("\n--- Verifying Metadata (Preview) ---")
info_path = os.path.join(ARROW_OUTPUT_DIR, "dataset_info.json")
state_path = os.path.join(ARROW_OUTPUT_DIR, "state.json")

if os.path.exists(info_path):
    with open(info_path, 'r') as f:
        info_data = json.load(f)
    print("\nContents of dataset_info.json (preview):")
    print(f"  Builder Name: {info_data.get('builder_name')}")
    print(f"  Features: {info_data.get('features')}")
    print(f"  Num examples (train): {info_data.get('splits', {}).get('train', {}).get('num_examples')}")
else:
    print(f"Could not find {info_path}")

if os.path.exists(state_path):
    with open(state_path, 'r') as f:
        state_data = json.load(f)
    print("\nContents of state.json (preview):")
    print(f"  Data Files: {state_data.get('_data_files')}")
    print(f"  Fingerprint: {state_data.get('_fingerprint')}")
else:
    print(f"Could not find {state_path}")

print("\n--- Dataset Tokenization Script Finished ---")