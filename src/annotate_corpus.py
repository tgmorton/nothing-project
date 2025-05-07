import os
import glob
import torch
from transformers import BertTokenizer, BertForMaskedLM
import argparse
import spacy
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch._jit_internal")

if torch.cuda.is_available():
    from torch.cuda.amp import autocast
else:
    class autocast:  # Dummy for CPU
        def __init__(self, enabled=True): self.enabled = enabled

        def __enter__(self): pass

        def __exit__(self, *args): pass

BERT_MODEL_NAME = 'bert-base-uncased'
SPACY_MODEL_NAME = 'en_core_web_sm'
K_TOP = 10
SPECIAL_MARKER = "[0]"
DEFAULT_CHUNK_READ_SIZE_CHARS = 500_000


def get_device():
    # ... (function remains the same) ...
    if torch.cuda.is_available():
        print("Using GPU.")
        return torch.device("cuda")
    else:
        print("Using CPU.")
        return torch.device("cpu")


def load_bert_model_and_tokenizer(model_name, device):
    # ... (function remains the same) ...
    print(f"Loading BERT model: {model_name}...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    print("BERT Model and tokenizer loaded.")
    return tokenizer, model


def load_spacy_model(model_name):
    # ... (function remains the same, including nlp.max_length adjustment and sentencizer add) ...
    print(f"Loading spaCy model: {model_name}...")
    try:
        disabled_pipes = ["parser", "ner", "tagger", "lemmatizer", "attribute_ruler"]
        nlp = spacy.load(model_name, disable=disabled_pipes)

        active_sentence_segmenters = any(pipe_name in ["senter", "sentencizer"] for pipe_name in nlp.pipe_names)
        if not active_sentence_segmenters:
            tqdm.write(
                f"Pipeline {nlp.pipe_names} lacks an active sentence segmenter. Adding 'sentencizer'.")  # Use tqdm.write
            try:
                nlp.add_pipe('sentencizer', first=True)
                tqdm.write("Successfully added 'sentencizer' to the spaCy pipeline.")  # Use tqdm.write
            except Exception as add_pipe_e:
                tqdm.write(
                    f"Failed to add 'sentencizer': {add_pipe_e}. Sentence segmentation may fail.")  # Use tqdm.write
        elif 'senter' in nlp.pipe_names:
            tqdm.write(f"'senter' component is active in pipeline: {nlp.pipe_names}.")  # Use tqdm.write
        elif 'sentencizer' in nlp.pipe_names:
            tqdm.write(f"'sentencizer' component is active in pipeline: {nlp.pipe_names}.")  # Use tqdm.write

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


def annotate_sentence(sentence_text, tokenizer, model, device, that_token_id, k_top_val):
    # ... (function remains the same) ...
    original_words = sentence_text.split()
    if len(original_words) < 2:
        return " ".join(original_words)

    output_word_list = list(original_words)
    bert_context_word_list = list(original_words)
    insertions_made_count = 0
    enable_amp = (device.type == 'cuda')

    for i in range(len(original_words) - 1):
        current_insertion_idx_for_lists = i + 1 + insertions_made_count
        temp_bert_input_words = list(bert_context_word_list)
        temp_bert_input_words.insert(current_insertion_idx_for_lists, tokenizer.mask_token)
        masked_input_string = " ".join(temp_bert_input_words)

        inputs = tokenizer.encode_plus(
            masked_input_string, return_tensors='pt', add_special_tokens=True,
            truncation=True, max_length=tokenizer.model_max_length
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        try:
            mask_token_indices = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if not mask_token_indices.numel(): continue  # Check if empty
            mask_token_index = mask_token_indices[0]
        except IndexError:
            continue

        with torch.no_grad():
            with autocast(enabled=enable_amp):
                outputs = model(input_ids, attention_mask=attention_mask)
            predictions = outputs.logits

        if mask_token_index >= predictions.shape[1]: continue

        mask_logits = predictions[0, mask_token_index, :]
        top_k_indices = torch.topk(mask_logits, k_top_val, dim=-1).indices.tolist()

        if that_token_id in top_k_indices:
            output_word_list.insert(current_insertion_idx_for_lists, SPECIAL_MARKER)
            bert_context_word_list.insert(current_insertion_idx_for_lists, 'that')
            insertions_made_count += 1
    return " ".join(output_word_list)


# Modified process_file with more granular progress bar
def process_file(filepath, output_filepath, bert_tokenizer, bert_model, spacy_nlp, device, that_token_id, k_top_val,
                 current_chunk_read_size_chars):
    tqdm.write(f"--- Starting process_file for: {os.path.basename(filepath)} ---")
    first_sentence_written_to_file = True

    try:
        file_size = os.path.getsize(filepath)
        tqdm.write(f"[{os.path.basename(filepath)}] File size: {file_size} bytes.")

        # Calculate approximate number of chunks for better descriptions
        num_total_chunks_approx = (
                                              file_size + current_chunk_read_size_chars - 1) // current_chunk_read_size_chars if current_chunk_read_size_chars > 0 else 1
        if num_total_chunks_approx == 0 and file_size > 0: num_total_chunks_approx = 1

        # Progress bar for reading the current file (position 1, as main file loop is 0)
        with open(output_filepath, 'w', encoding='utf-8') as f_out, \
                open(filepath, 'r', encoding='utf-8') as f_in, \
                tqdm(total=file_size, unit='B', unit_scale=True,
                     desc=f"Reading {os.path.basename(filepath)}",
                     leave=False, position=1, dynamic_ncols=True) as pbar_file_read:

            chunk_num = 0
            while True:
                chunk_num += 1
                # tqdm.write(f"[{os.path.basename(filepath)}] Attempting to read chunk {chunk_num}/{num_total_chunks_approx} (up to {current_chunk_read_size_chars} chars)...")
                text_chunk = f_in.read(current_chunk_read_size_chars)
                # tqdm.write(f"[{os.path.basename(filepath)}] Read {len(text_chunk)} characters for chunk {chunk_num}.")

                if not text_chunk:
                    tqdm.write(f"[{os.path.basename(filepath)}] End of file reached.")
                    break

                pbar_file_read.update(len(text_chunk.encode('utf-8', errors='ignore')))

                if not text_chunk.strip():
                    tqdm.write(f"[{os.path.basename(filepath)}] Chunk {chunk_num} is whitespace, skipping.")
                    continue

                # tqdm.write(f"[{os.path.basename(filepath)}] Processing chunk {chunk_num} with spaCy...")
                doc = spacy_nlp(text_chunk)
                sentences_in_chunk = list(doc.sents)  # Convert to list to get total for tqdm
                # tqdm.write(f"[{os.path.basename(filepath)}] spaCy found {len(sentences_in_chunk)} sentences in chunk {chunk_num}.")

                # Progress bar for annotating sentences within the current chunk (position 2)
                for sent in tqdm(sentences_in_chunk,
                                 desc=f"Annotating chunk {chunk_num}/{num_total_chunks_approx} of {os.path.basename(filepath)}",
                                 leave=False, unit="sent", position=2, dynamic_ncols=True):
                    sentence_text = sent.text.strip()
                    if not sentence_text:
                        continue

                    annotated_sentence = annotate_sentence(sentence_text, bert_tokenizer, bert_model, device,
                                                           that_token_id, k_top_val)

                    if not first_sentence_written_to_file:
                        f_out.write(" ")
                    f_out.write(annotated_sentence)
                    first_sentence_written_to_file = False
                    # tqdm.write(f"[{os.path.basename(filepath)}] Finished annotating chunk {chunk_num}.")

        tqdm.write(f"--- Finished processing and writing to: {output_filepath} ---")

    except FileNotFoundError:
        tqdm.write(f"Error: Input file not found {filepath}")
    except Exception as e:
        import traceback
        tqdm.write(f"Error processing file {filepath}: {e}\n{traceback.format_exc()}")


# main() function needs to set position=0 for its tqdm loop
def main():
    parser = argparse.ArgumentParser(description="Annotate text files...")  # Truncated for brevity
    # ... (all argparse setup remains the same) ...
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
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("torch.backends.cudnn.benchmark set to True")

    bert_tokenizer, bert_model = load_bert_model_and_tokenizer(args.bert_model_name, device)
    spacy_nlp = load_spacy_model(args.spacy_model_name)

    if args.chunk_size_chars <= 0:  # Prevent zero or negative chunk size
        tqdm.write(
            f"Error: chunk_size_chars ({args.chunk_size_chars}) must be positive. Using default {DEFAULT_CHUNK_READ_SIZE_CHARS}.")
        args.chunk_size_chars = DEFAULT_CHUNK_READ_SIZE_CHARS

    if args.chunk_size_chars >= spacy_nlp.max_length:
        tqdm.write(
            f"Warning: CHUNK_READ_SIZE_CHARS ({args.chunk_size_chars}) is >= spaCy's nlp.max_length ({spacy_nlp.max_length}). This might lead to errors.")

    that_token_id = bert_tokenizer.convert_tokens_to_ids('that')
    if isinstance(that_token_id, list):
        print(f"Warning: 'that' tokenized into multiple IDs: {that_token_id}. Using the first one.")
        that_token_id = that_token_id[0]
    if that_token_id == bert_tokenizer.unk_token_id:
        print(f"Warning: 'that' is an unknown token ({bert_tokenizer.unk_token}) for this BERT tokenizer.")

    # ... (folder checks remain the same) ...
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

    # Outer progress bar for files (position 0)
    for filepath in tqdm(train_files, desc="Overall File Progress", unit="file", position=0, dynamic_ncols=True):
        base_filename = os.path.basename(filepath)
        output_filename = base_filename + ".annotated"
        output_filepath = os.path.join(args.output_folder, output_filename)
        process_file(filepath, output_filepath, bert_tokenizer, bert_model, spacy_nlp, device, that_token_id,
                     args.k_top, args.chunk_size_chars)

    print("Annotation process complete.")


if __name__ == "__main__":
    main()