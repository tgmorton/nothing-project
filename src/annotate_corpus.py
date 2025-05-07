import os
import glob
import torch
from transformers import BertTokenizer, BertForMaskedLM
import argparse
import spacy
from tqdm import tqdm
import time  # For timing operations
import traceback  # For detailed error tracebacks

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch._jit_internal")

if torch.cuda.is_available():
    from torch.cuda.amp import autocast
else:
    # Define a dummy autocast context manager if CUDA is not available
    # so the 'with autocast():' line doesn't break on CPU-only setups.
    class autocast:
        def __init__(self, enabled=True):
            self.enabled = enabled

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

# --- Configuration ---
BERT_MODEL_NAME = 'bert-base-uncased'
SPACY_MODEL_NAME = 'en_core_web_sm'
K_TOP = 10
SPECIAL_MARKER = "[0]"
DEFAULT_CHUNK_READ_SIZE_CHARS = 500_000


def get_device():
    if torch.cuda.is_available():
        # tqdm.write("Using GPU.") # Using print for initial setup messages
        print("Using GPU.")
        return torch.device("cuda")
    else:
        # tqdm.write("Using CPU.")
        print("Using CPU.")
        return torch.device("cpu")


def load_bert_model_and_tokenizer(model_name, device):
    load_start_time = time.perf_counter()
    # tqdm.write(f"Loading BERT model: {model_name}...")
    print(f"Loading BERT model: {model_name}...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    load_end_time = time.perf_counter()
    # tqdm.write(f"BERT Model and tokenizer loaded in {load_end_time - load_start_time:.2f} seconds.")
    print(f"BERT Model and tokenizer loaded in {load_end_time - load_start_time:.2f} seconds.")
    return tokenizer, model


def load_spacy_model(model_name):
    load_start_time = time.perf_counter()
    # tqdm.write(f"Loading spaCy model: {model_name}...")
    print(f"Loading spaCy model: {model_name}...")
    try:
        disabled_pipes = ["parser", "ner", "tagger", "lemmatizer", "attribute_ruler"]
        nlp = spacy.load(model_name, disable=disabled_pipes)

        active_sentence_segmenters = any(pipe_name in ["senter", "sentencizer"] for pipe_name in nlp.pipe_names)
        if not active_sentence_segmenters:
            tqdm.write(
                f"Pipeline {nlp.pipe_names} lacks an active sentence segmenter. Adding 'sentencizer'.")
            try:
                nlp.add_pipe('sentencizer', first=True)
                tqdm.write("Successfully added 'sentencizer' to the spaCy pipeline.")
            except Exception as add_pipe_e:
                tqdm.write(
                    f"Failed to add 'sentencizer': {add_pipe_e}. Sentence segmentation may fail.")
        elif 'senter' in nlp.pipe_names:
            tqdm.write(f"'senter' component is active in pipeline: {nlp.pipe_names}.")
        elif 'sentencizer' in nlp.pipe_names:
            tqdm.write(f"'sentencizer' component is active in pipeline: {nlp.pipe_names}.")

        new_max_length = 12 * 1024 * 1024
        if nlp.max_length < new_max_length:
            nlp.max_length = new_max_length
            # tqdm.write(f"Increased spaCy nlp.max_length to: {nlp.max_length}")
            print(f"Increased spaCy nlp.max_length to: {nlp.max_length}")
        else:
            # tqdm.write(f"Current spaCy nlp.max_length ({nlp.max_length}) is sufficient and was not changed.")
            print(f"Current spaCy nlp.max_length ({nlp.max_length}) is sufficient and was not changed.")
    except OSError:
        print(f"spaCy model '{model_name}' not found. Please download it by running:")
        print(f"python -m spacy download {model_name}")
        raise
    except Exception as e:
        print(f"An error occurred while loading or configuring the spaCy model: {e}")
        raise

    load_end_time = time.perf_counter()
    # tqdm.write(f"Final spaCy pipeline components: {nlp.pipe_names}")
    # tqdm.write(f"spaCy model loaded and configured in {load_end_time - load_start_time:.2f} seconds.")
    print(f"Final spaCy pipeline components: {nlp.pipe_names}")
    print(f"spaCy model loaded and configured in {load_end_time - load_start_time:.2f} seconds.")
    return nlp


def annotate_sentence(sentence_text, tokenizer, model, device, that_token_id, k_top_val):
    func_start_time = time.perf_counter()

    timings = {
        "encode_plus_duration": 0.0,
        "model_inference_duration": 0.0,
        "topk_duration": 0.0,
        "bert_calls": 0,
        "num_slots_processed": 0
    }

    original_words = sentence_text.split()
    if len(original_words) < 2:
        timings["total_function_time"] = time.perf_counter() - func_start_time
        return " ".join(original_words), timings

    output_word_list = list(original_words)
    bert_context_word_list = list(original_words)
    insertions_made_count = 0
    enable_amp = (device.type == 'cuda')

    for i in range(len(original_words) - 1):
        timings["num_slots_processed"] += 1
        current_insertion_idx_for_lists = i + 1 + insertions_made_count
        temp_bert_input_words = list(bert_context_word_list)
        temp_bert_input_words.insert(current_insertion_idx_for_lists, tokenizer.mask_token)
        masked_input_string = " ".join(temp_bert_input_words)

        t_start_encode = time.perf_counter()
        inputs = tokenizer.encode_plus(
            masked_input_string, return_tensors='pt', add_special_tokens=True,
            truncation=True, max_length=tokenizer.model_max_length
        )
        timings["encode_plus_duration"] += (time.perf_counter() - t_start_encode)

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        try:
            mask_token_indices = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if not mask_token_indices.numel(): continue
            mask_token_index = mask_token_indices[0]
        except IndexError:
            continue

        timings["bert_calls"] += 1
        with torch.no_grad():
            with autocast(enabled=enable_amp):
                t_start_model = time.perf_counter()
                outputs = model(input_ids, attention_mask=attention_mask)
                timings["model_inference_duration"] += (time.perf_counter() - t_start_model)
            predictions = outputs.logits

        if mask_token_index >= predictions.shape[1]: continue

        mask_logits = predictions[0, mask_token_index, :]

        t_start_topk = time.perf_counter()
        top_k_indices = torch.topk(mask_logits, k_top_val, dim=-1).indices.tolist()
        timings["topk_duration"] += (time.perf_counter() - t_start_topk)

        if that_token_id in top_k_indices:
            output_word_list.insert(current_insertion_idx_for_lists, SPECIAL_MARKER)
            bert_context_word_list.insert(current_insertion_idx_for_lists, 'that')
            insertions_made_count += 1

    timings["total_function_time"] = time.perf_counter() - func_start_time
    return " ".join(output_word_list), timings


def process_file(filepath, output_filepath, bert_tokenizer, bert_model, spacy_nlp, device, that_token_id, k_top_val,
                 current_chunk_read_size_chars):
    tqdm.write(f"--- Starting process_file for: {os.path.basename(filepath)} ---")
    file_process_start_time = time.perf_counter()
    first_sentence_written_to_file = True

    # File-level timing aggregators
    file_total_read_time = 0.0
    file_total_spacy_time = 0.0
    file_total_annotation_loop_time = 0.0
    file_total_bert_calls = 0
    file_total_encode_plus_duration = 0.0
    file_total_model_inference_duration = 0.0
    file_total_topk_duration = 0.0
    file_total_annotate_function_time = 0.0
    file_total_sentences_processed = 0
    file_total_slots_processed = 0

    try:
        t_getsize_start = time.perf_counter()
        file_size = os.path.getsize(filepath)
        t_getsize_end = time.perf_counter()
        tqdm.write(
            f"[{os.path.basename(filepath)}] Got file size: {file_size} bytes in {t_getsize_end - t_getsize_start:.4f}s.")

        num_total_chunks_approx = (
                                          file_size + current_chunk_read_size_chars - 1) // current_chunk_read_size_chars if current_chunk_read_size_chars > 0 else 1
        if num_total_chunks_approx == 0 and file_size > 0: num_total_chunks_approx = 1
        tqdm.write(f"[{os.path.basename(filepath)}] Approx. chunks: {num_total_chunks_approx}")

        with open(output_filepath, 'w', encoding='utf-8') as f_out, \
                open(filepath, 'r', encoding='utf-8') as f_in, \
                tqdm(total=file_size, unit='B', unit_scale=True,
                     desc=f"Reading {os.path.basename(filepath)}",
                     leave=False, position=1, dynamic_ncols=True) as pbar_file_read:

            chunk_num = 0
            while True:
                chunk_num += 1

                t_read_start = time.perf_counter()
                text_chunk = f_in.read(current_chunk_read_size_chars)
                t_read_end = time.perf_counter()
                chunk_read_time = t_read_end - t_read_start
                file_total_read_time += chunk_read_time

                if not text_chunk:
                    tqdm.write(
                        f"[{os.path.basename(filepath)}] End of file reached after {chunk_read_time:.4f}s for final read attempt.")
                    break

                pbar_file_read.update(len(text_chunk.encode('utf-8', errors='ignore')))
                tqdm.write(
                    f"[{os.path.basename(filepath)}] Read {len(text_chunk)} chars for chunk {chunk_num} in {chunk_read_time:.4f}s.")

                if not text_chunk.strip():
                    tqdm.write(f"[{os.path.basename(filepath)}] Chunk {chunk_num} is whitespace, skipping.")
                    continue

                t_spacy_start = time.perf_counter()
                doc = spacy_nlp(text_chunk)
                t_spacy_end = time.perf_counter()
                spacy_proc_time = t_spacy_end - t_spacy_start
                file_total_spacy_time += spacy_proc_time

                sentences_in_chunk = list(doc.sents)
                tqdm.write(
                    f"[{os.path.basename(filepath)}] SpaCy processed chunk {chunk_num} in {spacy_proc_time:.4f}s. Found {len(sentences_in_chunk)} sentences.")

                chunk_sents_processed = 0
                chunk_total_bert_calls = 0
                chunk_total_encode_plus_duration = 0.0
                chunk_total_model_inference_duration = 0.0
                chunk_total_topk_duration = 0.0
                chunk_total_annotate_function_time = 0.0
                chunk_total_slots_processed = 0

                t_annotation_loop_start = time.perf_counter()
                for sent in tqdm(sentences_in_chunk,
                                 desc=f"Annotating chunk {chunk_num}/{num_total_chunks_approx} of {os.path.basename(filepath)}",
                                 leave=False, unit="sent", position=2, dynamic_ncols=True):
                    sentence_text = sent.text.strip()
                    if not sentence_text:
                        continue

                    annotated_sentence, sentence_timings = annotate_sentence(sentence_text, bert_tokenizer, bert_model,
                                                                             device,
                                                                             that_token_id, k_top_val)

                    chunk_sents_processed += 1
                    chunk_total_bert_calls += sentence_timings["bert_calls"]
                    chunk_total_encode_plus_duration += sentence_timings["encode_plus_duration"]
                    chunk_total_model_inference_duration += sentence_timings["model_inference_duration"]
                    chunk_total_topk_duration += sentence_timings["topk_duration"]
                    chunk_total_annotate_function_time += sentence_timings["total_function_time"]
                    chunk_total_slots_processed += sentence_timings["num_slots_processed"]

                    if not first_sentence_written_to_file:
                        f_out.write(" ")
                    f_out.write(annotated_sentence)
                    first_sentence_written_to_file = False

                t_annotation_loop_end = time.perf_counter()
                annotation_loop_time = t_annotation_loop_end - t_annotation_loop_start
                file_total_annotation_loop_time += annotation_loop_time

                # Accumulate file totals
                file_total_sentences_processed += chunk_sents_processed
                file_total_bert_calls += chunk_total_bert_calls
                file_total_encode_plus_duration += chunk_total_encode_plus_duration
                file_total_model_inference_duration += chunk_total_model_inference_duration
                file_total_topk_duration += chunk_total_topk_duration
                file_total_annotate_function_time += chunk_total_annotate_function_time
                file_total_slots_processed += chunk_total_slots_processed

                tqdm.write(
                    f"--- Chunk {chunk_num}/{num_total_chunks_approx} ({os.path.basename(filepath)}) Timings ---")
                tqdm.write(f"  Read time: {chunk_read_time:.4f}s")
                tqdm.write(f"  SpaCy proc time: {spacy_proc_time:.4f}s")
                tqdm.write(f"  Annotation loop time (for {chunk_sents_processed} sents): {annotation_loop_time:.4f}s")
                if chunk_sents_processed > 0:
                    tqdm.write(f"    Avg time per sent in loop: {annotation_loop_time / chunk_sents_processed:.4f}s")
                tqdm.write(f"  Total 'annotate_sentence' func time: {chunk_total_annotate_function_time:.4f}s")
                tqdm.write(f"    Breakdown (within annotate_sentence calls for this chunk):")
                tqdm.write(
                    f"      Total BERT calls: {chunk_total_bert_calls} (across {chunk_total_slots_processed} slots)")
                tqdm.write(f"      Total Encode+: {chunk_total_encode_plus_duration:.4f}s")
                tqdm.write(f"      Total Model Infer: {chunk_total_model_inference_duration:.4f}s")
                if chunk_total_bert_calls > 0:
                    tqdm.write(
                        f"        Avg Model Infer per call: {chunk_total_model_inference_duration / chunk_total_bert_calls:.4f}s")
                tqdm.write(f"      Total TopK: {chunk_total_topk_duration:.4f}s")
                tqdm.write(f"--- End Chunk {chunk_num} Timings ---")

        file_process_end_time = time.perf_counter()
        total_file_time = file_process_end_time - file_process_start_time
        tqdm.write(f"--- FINISHED FILE: {os.path.basename(filepath)} in {total_file_time:.2f}s ---")
        tqdm.write(f"  Total sentences processed: {file_total_sentences_processed}")
        tqdm.write(f"  Total read time: {file_total_read_time:.2f}s")
        tqdm.write(f"  Total spaCy time: {file_total_spacy_time:.2f}s")
        tqdm.write(f"  Total annotation loop time: {file_total_annotation_loop_time:.2f}s")
        tqdm.write(f"  Total accumulated 'annotate_sentence' func time: {file_total_annotate_function_time:.2f}s")
        tqdm.write(f"    Overall Breakdown (within all annotate_sentence calls for this file):")
        tqdm.write(f"      Total BERT calls: {file_total_bert_calls} (across {file_total_slots_processed} slots)")
        tqdm.write(f"      Total Encode+: {file_total_encode_plus_duration:.2f}s")
        tqdm.write(f"      Total Model Infer: {file_total_model_inference_duration:.2f}s")
        if file_total_bert_calls > 0:
            tqdm.write(
                f"        Avg Model Infer per call: {file_total_model_inference_duration / file_total_bert_calls:.4f}s")
            tqdm.write(
                f"        Avg Slots per BERT call: {file_total_slots_processed / file_total_bert_calls if file_total_bert_calls > 0 else 0 :.2f}")  # Should be 1
            tqdm.write(
                f"        Avg Bert calls per sentence: {file_total_bert_calls / file_total_sentences_processed if file_total_sentences_processed > 0 else 0 :.2f}")

        tqdm.write(f"      Total TopK: {file_total_topk_duration:.2f}s")
        tqdm.write(f"--- End File Summary for {os.path.basename(filepath)} ---")


    except FileNotFoundError:
        tqdm.write(f"Error: Input file not found {filepath}")
    except Exception as e:
        tqdm.write(f"Error processing file {filepath}: {e}\n{traceback.format_exc()}")


def main():
    parser = argparse.ArgumentParser(description="Annotate text files with extensive timing.")
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
    script_start_time = time.perf_counter()

    # Use print for initial setup messages that appear before tqdm loops dominate
    device = get_device()
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("torch.backends.cudnn.benchmark set to True")

    bert_tokenizer, bert_model = load_bert_model_and_tokenizer(args.bert_model_name, device)
    spacy_nlp = load_spacy_model(args.spacy_model_name)

    if args.chunk_size_chars <= 0:
        print(
            f"Error: chunk_size_chars ({args.chunk_size_chars}) must be positive. Using default {DEFAULT_CHUNK_READ_SIZE_CHARS}.")
        args.chunk_size_chars = DEFAULT_CHUNK_READ_SIZE_CHARS

    if args.chunk_size_chars >= spacy_nlp.max_length:
        print(  # Changed to print as it's before main loop
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

    overall_start_time = time.perf_counter()
    for filepath in tqdm(train_files, desc="Overall File Progress", unit="file", position=0, dynamic_ncols=True):
        process_file(filepath, output_filepath, bert_tokenizer, bert_model, spacy_nlp, device, that_token_id,
                     args.k_top, args.chunk_size_chars)

    overall_end_time = time.perf_counter()
    print(f"Annotation process complete. Total script time: {overall_end_time - overall_start_time:.2f} seconds.")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(time.time()))} (System local time)")
    print(
        f"Current date from script perspective: {time.strftime('%A, %B %d, %Y at %I:%M:%S %p %Z', time.gmtime())} (UTC)")


if __name__ == "__main__":
    main()