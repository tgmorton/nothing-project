# tokenize_dataset.py (Version 2 - with optional max_length)
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
parser.add_argument(
    "--max_length",  # New optional argument
    type=int,
    default=None,
    help="Optional: Maximum sequence length. If provided, sequences will be truncated to this length."
)
# Optional: Add arguments for num_proc, etc. if needed
# parser.add_argument("--num_proc", type=int, default=None, help="Number of processes for dataset mapping.")

args = parser.parse_args()

# --- 2. Configuration & Setup ---
INPUT_DATA_DIR = args.input_dir
TOKENIZER_PATH = args.tokenizer_dir
ARROW_OUTPUT_DIR = args.output_dir
MAX_LENGTH = args.max_length  # Store the max_length argument
# NUM_PROC = args.num_proc

# Create output directory if it doesn't exist
Path(ARROW_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# --- 3. Find Input Data Files ---
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
    print(f"Tokenizer's default model_max_length: {tokenizer_for_processing.model_max_length}")
    if MAX_LENGTH:
        print(f"Applying user-defined max_length: {MAX_LENGTH}")
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
        # For manual ByteLevelBPETokenizer, truncation is handled differently (see tokenize_function)
        print("Loaded tokenizer manually using ByteLevelBPETokenizer.")
        if MAX_LENGTH:
            print(f"Applying user-defined max_length: {MAX_LENGTH} (will be handled during encoding)")
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
raw_datasets = load_dataset('text', data_files={'train': train_files})
print("Raw dataset loaded:")
print(raw_datasets)


# --- 6. Tokenize Data ---
print("\n--- Tokenizing Data ---")

# Define the function to tokenize batches of text
def tokenize_function(examples):
    tokenizer_options = {}
    if MAX_LENGTH is not None:
        tokenizer_options['max_length'] = MAX_LENGTH
        tokenizer_options['truncation'] = True
        # If you also want padding to max_length, uncomment below:
        # tokenizer_options['padding'] = "max_length"
        # tokenizer_options['return_attention_mask'] = True # Usually needed if padding

    if hasattr(tokenizer_for_processing, '__call__'): # For transformers FastTokenizers
        tokenized_output = tokenizer_for_processing(
            examples['text'],
            **tokenizer_options # Pass options here
        )
        # Prepare dict for what .map expects. If padding/attention_mask enabled, add them.
        output_dict = {'input_ids': tokenized_output['input_ids']}
        if 'attention_mask' in tokenized_output:
            output_dict['attention_mask'] = tokenized_output['attention_mask']
        return output_dict
    else: # For manually loaded ByteLevelBPETokenizer from 'tokenizers' library
        # This path assumes tokenizer_for_processing is an instance from 'tokenizers' library
        # and needs manual handling for truncation if MAX_LENGTH is set.
        all_input_ids = []
        for text_item in examples['text']:
            # The .encode() method of tokenizers.ByteLevelBPETokenizer doesn't take max_length directly.
            # Truncation can be enabled on the tokenizer instance itself or handled post-encoding.
            # For simplicity here, we'll truncate post-encoding if MAX_LENGTH is set.
            # A more robust way for tokenizers.Tokenizer would be to configure its truncation property.
            encoding = tokenizer_for_processing.encode(text_item)
            ids = encoding.ids
            if MAX_LENGTH is not None and len(ids) > MAX_LENGTH:
                ids = ids[:MAX_LENGTH]
            all_input_ids.append(ids)
        return {'input_ids': all_input_ids}


# Apply the tokenization function
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    # num_proc=NUM_PROC, # Optional parallel processing
    remove_columns=raw_datasets["train"].column_names, # Keep only specified columns
    desc="Running tokenizer on dataset",
)

print("Tokenization complete.")
print("Tokenized dataset structure:")
print(tokenized_datasets)
print("Example entry (first 50 token IDs if available, else full sequence):")
if len(tokenized_datasets['train']) > 0:
    example_ids = tokenized_datasets['train'][0]['input_ids']
    print(example_ids[:min(50, len(example_ids))])
    if MAX_LENGTH:
        print(f"Length of this example after potential truncation: {len(example_ids)} (max_length was {MAX_LENGTH})")
else:
    print("Dataset is empty.")

# --- 7. Save Tokenized Data as Arrow with Metadata ---
print(f"\n--- Saving Tokenized Dataset to Arrow Format in: {ARROW_OUTPUT_DIR} ---")
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
    print(f"  Features: {info_data.get('features')}") # This will show 'input_ids' and 'attention_mask' if generated
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