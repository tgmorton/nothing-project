import os
import glob
import torch
from transformers import BertTokenizer, BertForMaskedLM
import argparse
import spacy
from tqdm import tqdm

# Import autocast for mixed precision if CUDA is available
if torch.cuda.is_available():
    from torch.cuda.amp import autocast
else:
    # Define a dummy autocast context manager if CUDA is not available
    # so the 'with autocast():' line doesn't break on CPU-only setups.
    class autocast:
        def __init__(self, enabled=True): pass

        def __enter__(self): pass

        def __exit__(self, *args): pass

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch._jit_internal")

# --- Configuration ---
BERT_MODEL_NAME = 'bert-base-uncased'
SPACY_MODEL_NAME = 'en_core_web_sm'
K_TOP = 10
SPECIAL_MARKER = "[0]"
DEFAULT_CHUNK_READ_SIZE_CHARS = 500_000


def get_device():
    if torch.cuda.is_available():
        print("Using GPU.")
        return torch.device("cuda")
    else:
        print("Using CPU.")
        return torch.device("cpu")


def load_bert_model_and_tokenizer(model_name, device):
    print(f"Loading BERT model: {model_name}...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    print("BERT Model and tokenizer loaded.")
    return tokenizer, model


def load_spacy_model(model_name):
    print(f"Loading spaCy model: {model_name}...")
    try:
        disabled_pipes = ["parser", "ner", "tagger", "lemmatizer", "attribute_ruler"]
        nlp = spacy.load(model_name, disable=disabled_pipes)

        active_sentence_segmenters = any(pipe_name in ["senter", "sentencizer"] for pipe_name in nlp.pipe_names)
        if not active_sentence_segmenters:
            print(f"Pipeline {nlp.pipe_names} lacks an active sentence segmenter. Adding 'sentencizer'.")
            try:
                nlp.add_pipe('sentencizer', first=True)
                print("Successfully added 'sentencizer' to the spaCy pipeline.")
            except Exception as add_pipe_e:
                print(f"Failed to add 'sentencizer': {add_pipe_e}. Sentence segmentation may fail.")
        elif 'senter' in nlp.pipe_names:
            print(f"'senter' component is active in pipeline: {nlp.pipe_names}.")
        elif 'sentencizer' in nlp.pipe_names:
            print(f"'sentencizer' component is active in pipeline: {nlp.pipe_names}.")

        new_max_length = 12 * 1024 * 1024
        if nlp.max_length < new_max_length:
            nlp.max_length = new_max_length
            print(f"Increased spaCy nlp.max_length to: {nlp.max_length}")
        else:
            print(f"Current spaCy nlp.max_length ({nlp.max_length}) is sufficient and was not changed.")
    except OSError:
        print(f"spaCy model '{model_name}' not found. Please download it by running:")
        print(f"python -m spacy download {model_name}")
        raise
    except Exception as e:
        print(f"An error occurred while loading or configuring the spaCy model: {e}")
        raise
    print(f"Final spaCy pipeline components: {nlp.pipe_names}")
    print("spaCy model loaded and configured.")
    return nlp


# Modified to include AMP (autocast)
def annotate_sentence(sentence_text, tokenizer, model, device, that_token_id, k_top_val):
    original_words = sentence_text.split()
    if len(original_words) < 2:
        return " ".join(original_words)

    output_word_list = list(original_words)
    bert_context_word_list = list(original_words)
    insertions_made_count = 0

    # Determine if autocast should be enabled
    enable_amp = (device.type == 'cuda')

    for i in range(len(original_words) - 1):
        current_insertion_idx_for_lists = i + 1 + insertions_made_count
        temp_bert_input_words = list(bert_context_word_list)
        temp_bert_input_words.insert(current_insertion_idx_for_lists, tokenizer.mask_token)
        masked_input_string = " ".join(temp_bert_input_words)

        inputs = tokenizer.encode_plus(
            masked_input_string,
            return_tensors='pt',
            add_special_tokens=True,
            truncation=True,
            max_length=tokenizer.model_max_length
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        try:
            mask_token_indices = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_token_indices) == 0:
                continue
            mask_token_index = mask_token_indices[0]
        except IndexError:
            continue

        with torch.no_grad():
            # Use autocast for mixed precision if on CUDA and enabled
            with autocast(enabled=enable_amp):
                outputs = model(input_ids, attention_mask=attention_mask)
            predictions = outputs.logits  # Logits are usually float32 by default after autocast

        if mask_token_index >= predictions.shape[1]:
            continue

        mask_logits = predictions[0, mask_token_index, :]
        top_k_indices = torch.topk(mask_logits, k_top_val, dim=-1).indices.tolist()

        if that_token_id in top_k_indices:
            output_word_list.insert(current_insertion_idx_for_lists, SPECIAL_MARKER)
            bert_context_word_list.insert(current_insertion_idx_for_lists, 'that')
            insertions_made_count += 1
    return " ".join(output_word_list)


def process_file(filepath, output_filepath, bert_tokenizer, bert_model, spacy_nlp, device, that_token_id, k_top_val,
                 current_chunk_read_size_chars):
    first_sentence_written_to_file = True
    try:
        file_size = os.path.getsize(filepath)
        with open(output_filepath, 'w', encoding='utf-8') as f_out, \
                open(filepath, 'r', encoding='utf-8') as f_in, \
                tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Reading {os.path.basename(filepath)}",
                     leave=False) as pbar_file_read:

            while True:
                text_chunk = f_in.read(current_chunk_read_size_chars)
                if not text_chunk:
                    break
                pbar_file_read.update(len(text_chunk.encode('utf-8', errors='ignore')))
                if not text_chunk.strip():
                    continue
                doc = spacy_nlp(text_chunk)
                for sent in doc.sents:
                    sentence_text = sent.text.strip()
                    if not sentence_text:
                        continue
                    annotated_sentence = annotate_sentence(sentence_text, bert_tokenizer, bert_model, device,
                                                           that_token_id, k_top_val)
                    if not first_sentence_written_to_file:
                        f_out.write(" ")
                    f_out.write(annotated_sentence)
                    first_sentence_written_to_file = False
    except FileNotFoundError:
        tqdm.write(f"Error: Input file not found {filepath}")
    except Exception as e:
        tqdm.write(f"Error processing file {filepath}: {e} (output may be incomplete or missing)")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate text files with a special marker '[0]' where 'that' is a plausible complementizer, using spaCy for sentence tokenization.")
    parser.add_argument("input_folder", type=str, help="Folder containing .train text files to annotate.")
    parser.add_argument("output_folder", type=str, help="Folder where annotated files will be saved.")
    parser.add_argument("--bert_model_name", type=str, default=BERT_MODEL_NAME,
                        help=f"Name of the BERT model to use (default: {BERT_MODEL_NAME}).")
    parser.add_argument("--spacy_model_name", type=str, default=SPACY_MODEL_NAME,
                        help=f"Name of the spaCy model for sentence tokenization (default: {SPACY_MODEL_NAME}).")
    parser.add_argument("--k_top", type=int, default=K_TOP,
                        help=f"Consider 'that' if it's in the top K predictions (default: {K_TOP}).")
    parser.add_argument("--chunk_size_chars", type=int, default=DEFAULT_CHUNK_READ_SIZE_CHARS,
                        help=f"Number of characters to read into memory at a time for processing (default: {DEFAULT_CHUNK_READ_SIZE_CHARS}).")

    args = parser.parse_args()

    device = get_device()
    # --- Enable CuDNN benchmarking if on CUDA ---
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("torch.backends.cudnn.benchmark set to True")
    # --- End of CuDNN benchmarking ---

    bert_tokenizer, bert_model = load_bert_model_and_tokenizer(args.bert_model_name, device)
    spacy_nlp = load_spacy_model(args.spacy_model_name)

    if args.chunk_size_chars >= spacy_nlp.max_length:
        tqdm.write(
            f"Warning: CHUNK_READ_SIZE_CHARS ({args.chunk_size_chars}) is >= spaCy's nlp.max_length ({spacy_nlp.max_length}). This might lead to errors.")

    that_token_id = bert_tokenizer.convert_tokens_to_ids('that')
    if isinstance(that_token_id, list):
        print(f"Warning: 'that' tokenized into multiple IDs: {that_token_id}. Using the first one.")
        that_token_id = that_token_id[0]
    if that_token_id == bert_tokenizer.unk_token_id:
        print(f"Warning: 'that' is an unknown token ({bert_tokenizer.unk_token}) for this BERT tokenizer.")

    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' not found.")
        return
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        print(f"Created output folder: {args.output_folder}")

    input_file_pattern = os.path.join(args.input_folder, "*.train")
    train_files = glob.glob(input_file_pattern)

    if not train_files:
        print(f"No .train files found in '{args.input_folder}' matching pattern '{input_file_pattern}'.")
        return
    print(f"Found {len(train_files)} .train files to process.")

    for filepath in tqdm(train_files, desc="Overall File Progress", unit="file"):
        base_filename = os.path.basename(filepath)
        output_filename = base_filename + ".annotated"
        output_filepath = os.path.join(args.output_folder, output_filename)
        process_file(filepath, output_filepath, bert_tokenizer, bert_model, spacy_nlp, device, that_token_id,
                     args.k_top, args.chunk_size_chars)

    print("Annotation process complete.")


if __name__ == "__main__":
    main()