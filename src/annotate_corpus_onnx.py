import os
import glob
import torch  # Still needed for topk, device handling
from transformers import AutoTokenizer  # Keep AutoTokenizer
import onnxruntime as ort  # Import ONNX Runtime
import numpy as np  # For numpy arrays needed by ONNX Runtime
import argparse
import spacy
from tqdm import tqdm
import time
import traceback
import string  # Import string for punctuation handling

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch._jit_internal")
warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)  # Ignore specific warning

# NOTE: torch.cuda.amp.autocast is NOT used with ONNX Runtime

# --- Configuration ---
SPACY_MODEL_NAME = 'en_core_web_sm'
K_TOP = 10
SPECIAL_MARKER = "[0]"
DEFAULT_CHUNK_READ_SIZE_CHARS = 500_000
# Define 'that' variants to check against, including common contractions
THAT_VARIANTS = {'that', "that's"}


def get_device():
    # Determines if CUDA is available for provider selection
    if torch.cuda.is_available():
        print("CUDA available via PyTorch. Will attempt to use ONNX Runtime CUDA Execution Provider.")
        device_idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(',')[0]
        return f"cuda:{device_idx}"  # Return string like 'cuda:0'
    else:
        print("Using CPU.")
        return "cpu"


def load_onnx_session_and_tokenizer(model_onnx_path, tokenizer_dir_path, device_str):
    """Loads ONNX Runtime session and Hugging Face Tokenizer."""
    load_start_time = time.perf_counter()
    print(f"Loading ONNX model from: {model_onnx_path}")
    print(f"Loading Tokenizer from: {tokenizer_dir_path}")
    session = None
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir_path)
        sess_options = ort.SessionOptions()
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': device_str.split(':')[-1],
            }),
            'CPUExecutionProvider'  # Fallback
        ] if device_str.startswith('cuda') else ['CPUExecutionProvider']

        if device_str.startswith('cuda'):
            print(
                f"Attempting to configure ONNX Runtime with CUDAExecutionProvider on device {device_str.split(':')[-1]}.")
        else:
            print(f"Configuring ONNX Runtime with CPUExecutionProvider.")

        session = ort.InferenceSession(model_onnx_path, sess_options=sess_options, providers=providers)
        current_providers = session.get_providers()
        print(f"ONNX Runtime session created using providers: {current_providers}")
        if device_str.startswith('cuda') and 'CUDAExecutionProvider' not in current_providers:
            print(
                "Warning: Requested CUDAExecutionProvider, but it's not in the active providers list. Check ONNX Runtime GPU build and CUDA setup.")

    except Exception as e:
        print(f"Error loading ONNX model/tokenizer: {e}")
        raise
    load_end_time = time.perf_counter()
    print(f"ONNX session and tokenizer loaded in {load_end_time - load_start_time:.2f} seconds.")
    return session, tokenizer


def load_spacy_model(model_name):
    load_start_time = time.perf_counter()
    print(f"Loading spaCy model: {model_name}...")
    try:
        disabled_pipes = ["parser", "ner", "tagger", "lemmatizer", "attribute_ruler"]
        nlp = spacy.load(model_name, disable=disabled_pipes)
        active_sentence_segmenters = any(pipe_name in ["senter", "sentencizer"] for pipe_name in nlp.pipe_names)
        if not active_sentence_segmenters:
            tqdm.write(f"Pipeline {nlp.pipe_names} lacks an active sentence segmenter. Adding 'sentencizer'.")
            try:
                nlp.add_pipe('sentencizer', first=True); tqdm.write("Successfully added 'sentencizer'.")
            except Exception as e:
                tqdm.write(f"Failed to add 'sentencizer': {e}.")
        elif 'senter' in nlp.pipe_names:
            tqdm.write(f"'senter' component is active in pipeline: {nlp.pipe_names}.")
        elif 'sentencizer' in nlp.pipe_names:
            tqdm.write(f"'sentencizer' component is active in pipeline: {nlp.pipe_names}.")
        new_max_length = 12 * 1024 * 1024
        if nlp.max_length < new_max_length:
            nlp.max_length = new_max_length; print(f"Increased spaCy nlp.max_length to: {nlp.max_length}")
        else:
            print(f"Current spaCy nlp.max_length ({nlp.max_length}) is sufficient.")
    except OSError:
        print(f"spaCy model '{model_name}' not found. DL: python -m spacy download {model_name}"); raise
    except Exception as e:
        print(f"An error occurred loading spaCy model: {e}"); raise
    load_end_time = time.perf_counter()
    print(f"Final spaCy pipeline components: {nlp.pipe_names}")
    print(f"spaCy model loaded and configured in {load_end_time - load_start_time:.2f} seconds.")
    return nlp


# --- MODIFIED annotate_sentence with adjacency check ---
def annotate_sentence(sentence_text, tokenizer, ort_session, that_token_id, k_top_val):
    """Uses ONNX Runtime session for inference and checks adjacent words."""
    func_start_time = time.perf_counter()

    timings = {
        "encode_plus_duration": 0.0,
        "model_inference_duration": 0.0,
        "topk_duration": 0.0,
        "bert_calls": 0,  # Or ONNX calls
        "num_slots_processed": 0
    }

    # Define punctuation to strip for adjacency check
    punctuation_to_strip = string.punctuation

    original_words = sentence_text.split()
    if len(original_words) < 2:
        timings["total_function_time"] = time.perf_counter() - func_start_time
        return " ".join(original_words), timings

    output_word_list = list(original_words)
    bert_context_word_list = list(original_words)
    insertions_made_count = 0

    for i in range(len(original_words) - 1):  # Slot is between original_words[i] and original_words[i+1]
        timings["num_slots_processed"] += 1
        current_insertion_idx_for_lists = i + 1 + insertions_made_count

        # --- Check adjacent words before doing expensive work ---
        word_before_raw = original_words[i]
        word_after_raw = original_words[i + 1]

        # Normalize for checking: lowercase and strip punctuation
        word_before_norm = word_before_raw.lower().strip(punctuation_to_strip)
        word_after_norm = word_after_raw.lower().strip(punctuation_to_strip)

        # If either adjacent word is 'that' or 'that's', skip this slot entirely
        if word_before_norm in THAT_VARIANTS or word_after_norm in THAT_VARIANTS:
            # Optional: Add debug logging if needed
            # tqdm.write(f"DEBUG: Skipped slot between '{word_before_raw}' and '{word_after_raw}' due to adjacent 'that'.")
            continue  # Move to the next potential insertion slot (next i)
        # --- End adjacency check ---

        # If adjacency check passed, proceed with masking and prediction
        temp_bert_input_words = list(bert_context_word_list)
        temp_bert_input_words.insert(current_insertion_idx_for_lists, tokenizer.mask_token)
        masked_input_string = " ".join(temp_bert_input_words)

        t_start_encode = time.perf_counter()
        inputs = tokenizer.encode_plus(
            masked_input_string, return_tensors='pt', add_special_tokens=True,
            truncation=True, max_length=tokenizer.model_max_length
        )
        timings["encode_plus_duration"] += (time.perf_counter() - t_start_encode)

        input_ids_pt = inputs['input_ids']
        attention_mask_pt = inputs['attention_mask']

        try:
            mask_token_indices = (input_ids_pt[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if not mask_token_indices.numel(): continue
            mask_token_index = mask_token_indices[0].item()
        except IndexError:
            continue

        ort_inputs = {
            'input_ids': input_ids_pt.cpu().numpy(),
            'attention_mask': attention_mask_pt.cpu().numpy()
        }
        if 'token_type_ids' in [inp.name for inp in ort_session.get_inputs()]:
            if 'token_type_ids' in inputs:
                ort_inputs['token_type_ids'] = inputs['token_type_ids'].cpu().numpy()
            else:
                ort_inputs['token_type_ids'] = np.zeros_like(ort_inputs['input_ids'])

        timings["bert_calls"] += 1
        t_start_model = time.perf_counter()
        ort_outputs = ort_session.run(None, ort_inputs)
        predictions_np = ort_outputs[0]
        timings["model_inference_duration"] += (time.perf_counter() - t_start_model)

        if mask_token_index >= predictions_np.shape[1]: continue

        mask_logits_np = predictions_np[0, mask_token_index, :]

        t_start_topk = time.perf_counter()
        mask_logits_pt = torch.from_numpy(mask_logits_np)
        top_k_indices = torch.topk(mask_logits_pt, k_top_val, dim=-1).indices.tolist()
        timings["topk_duration"] += (time.perf_counter() - t_start_topk)

        # Check if 'that' is predicted (and adjacency check already passed)
        if that_token_id in top_k_indices:
            # Insert marker and update context list (for potential future use, though batching removes this need)
            output_word_list.insert(current_insertion_idx_for_lists, SPECIAL_MARKER)
            bert_context_word_list.insert(current_insertion_idx_for_lists,
                                          'that')  # Still update context for consistency with original logic
            insertions_made_count += 1

    timings["total_function_time"] = time.perf_counter() - func_start_time
    return " ".join(output_word_list), timings


def process_file(filepath, output_filepath, bert_tokenizer, ort_session, spacy_nlp, device_str, that_token_id,
                 k_top_val,
                 current_chunk_read_size_chars):
    # ... (This function remains the same as the last version, including timing and flushing) ...
    tqdm.write(f"--- Starting process_file for: {os.path.basename(filepath)} ---")
    file_process_start_time = time.perf_counter()
    first_sentence_written_to_file = True

    file_total_read_time = 0.0;
    file_total_spacy_time = 0.0;
    file_total_annotation_loop_time = 0.0
    file_total_onnx_calls = 0;
    file_total_encode_plus_duration = 0.0;
    file_total_model_inference_duration = 0.0
    file_total_topk_duration = 0.0;
    file_total_annotate_function_time = 0.0
    file_total_sentences_processed = 0;
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
                tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Reading {os.path.basename(filepath)}",
                     leave=False, position=1, dynamic_ncols=True) as pbar_file_read:
            chunk_num = 0
            while True:
                chunk_num += 1;
                t_read_start = time.perf_counter();
                text_chunk = f_in.read(current_chunk_read_size_chars);
                t_read_end = time.perf_counter();
                chunk_read_time = t_read_end - t_read_start;
                file_total_read_time += chunk_read_time
                if not text_chunk: tqdm.write(
                    f"[{os.path.basename(filepath)}] End of file reached after {chunk_read_time:.4f}s for final read attempt."); break
                pbar_file_read.update(len(text_chunk.encode('utf-8', errors='ignore')))
                if not text_chunk.strip(): tqdm.write(
                    f"[{os.path.basename(filepath)}] Chunk {chunk_num} is whitespace, skipping."); continue

                t_spacy_start = time.perf_counter();
                doc = spacy_nlp(text_chunk);
                t_spacy_end = time.perf_counter();
                spacy_proc_time = t_spacy_end - t_spacy_start;
                file_total_spacy_time += spacy_proc_time
                sentences_in_chunk = list(doc.sents)

                chunk_sents_processed = 0;
                chunk_total_onnx_calls = 0;
                chunk_total_encode_plus_duration = 0.0;
                chunk_total_model_inference_duration = 0.0
                chunk_total_topk_duration = 0.0;
                chunk_total_annotate_function_time = 0.0;
                chunk_total_slots_processed = 0
                t_annotation_loop_start = time.perf_counter()

                for sent in tqdm(sentences_in_chunk,
                                 desc=f"Annotating chunk {chunk_num}/{num_total_chunks_approx} of {os.path.basename(filepath)}",
                                 leave=False, unit="sent", position=2, dynamic_ncols=True):
                    sentence_text = sent.text.strip();
                    if not sentence_text: continue
                    annotated_sentence, sentence_timings = annotate_sentence(sentence_text, bert_tokenizer, ort_session,
                                                                             that_token_id, k_top_val)
                    chunk_sents_processed += 1;
                    chunk_total_onnx_calls += sentence_timings["bert_calls"];
                    chunk_total_encode_plus_duration += sentence_timings["encode_plus_duration"]
                    chunk_total_model_inference_duration += sentence_timings["model_inference_duration"];
                    chunk_total_topk_duration += sentence_timings["topk_duration"]
                    chunk_total_annotate_function_time += sentence_timings["total_function_time"];
                    chunk_total_slots_processed += sentence_timings["num_slots_processed"]
                    if not first_sentence_written_to_file: f_out.write(" "); f_out.write(
                        annotated_sentence); first_sentence_written_to_file = False

                t_annotation_loop_end = time.perf_counter();
                annotation_loop_time = t_annotation_loop_end - t_annotation_loop_start;
                file_total_annotation_loop_time += annotation_loop_time
                file_total_sentences_processed += chunk_sents_processed;
                file_total_onnx_calls += chunk_total_onnx_calls;
                file_total_encode_plus_duration += chunk_total_encode_plus_duration
                file_total_model_inference_duration += chunk_total_model_inference_duration;
                file_total_topk_duration += chunk_total_topk_duration
                file_total_annotate_function_time += chunk_total_annotate_function_time;
                file_total_slots_processed += chunk_total_slots_processed

                tqdm.write(
                    f"--- Chunk {chunk_num}/{num_total_chunks_approx} ({os.path.basename(filepath)}) Timings ---");
                tqdm.write(f"  Read time: {chunk_read_time:.4f}s ({len(text_chunk)} chars)");
                tqdm.write(f"  SpaCy proc time: {spacy_proc_time:.4f}s");
                tqdm.write(f"  Annotation loop time (for {chunk_sents_processed} sents): {annotation_loop_time:.4f}s")
                if chunk_sents_processed > 0: tqdm.write(
                    f"    Avg time per sent in loop: {annotation_loop_time / chunk_sents_processed:.4f}s")
                tqdm.write(
                    f"  Total accumulated 'annotate_sentence' func time: {chunk_total_annotate_function_time:.4f}s");
                tqdm.write(f"    Breakdown (within annotate_sentence calls for this chunk):");
                tqdm.write(
                    f"      Total ONNX calls: {chunk_total_onnx_calls} (across {chunk_total_slots_processed} slots)");
                tqdm.write(f"      Total Encode+: {chunk_total_encode_plus_duration:.4f}s")
                tqdm.write(f"      Total Model Infer: {chunk_total_model_inference_duration:.4f}s");
                if chunk_total_onnx_calls > 0: tqdm.write(
                    f"        Avg Model Infer per call: {chunk_total_model_inference_duration / chunk_total_onnx_calls:.4f}s")
                tqdm.write(f"      Total TopK: {chunk_total_topk_duration:.4f}s");
                tqdm.write(f"--- End Chunk {chunk_num} Timings ---");
                f_out.flush();
                tqdm.write(f"[{os.path.basename(filepath)}] Output flushed after chunk {chunk_num}.")

        file_process_end_time = time.perf_counter();
        total_file_time = file_process_end_time - file_process_start_time
        tqdm.write(f"--- FINISHED FILE: {os.path.basename(filepath)} in {total_file_time:.2f}s ---");
        tqdm.write(f"  Total sentences processed: {file_total_sentences_processed}");
        tqdm.write(f"  Total read time: {file_total_read_time:.2f}s");
        tqdm.write(f"  Total spaCy time: {file_total_spacy_time:.2f}s")
        tqdm.write(f"  Total annotation loop time: {file_total_annotation_loop_time:.2f}s");
        tqdm.write(f"  Total accumulated 'annotate_sentence' func time: {file_total_annotate_function_time:.2f}s")
        tqdm.write(f"    Overall Breakdown (within all annotate_sentence calls for this file):");
        tqdm.write(f"      Total ONNX calls: {file_total_onnx_calls} (across {file_total_slots_processed} slots)");
        tqdm.write(f"      Total Encode+: {file_total_encode_plus_duration:.2f}s")
        tqdm.write(f"      Total Model Infer: {file_total_model_inference_duration:.2f}s")
        if file_total_onnx_calls > 0: tqdm.write(
            f"        Avg Model Infer per call: {file_total_model_inference_duration / file_total_onnx_calls:.4f}s"); tqdm.write(
            f"        Avg Slots per ONNX call: {file_total_slots_processed / file_total_onnx_calls if file_total_onnx_calls > 0 else 0 :.2f}"); tqdm.write(
            f"        Avg ONNX calls per sentence: {file_total_onnx_calls / file_total_sentences_processed if file_total_sentences_processed > 0 else 0 :.2f}")
        tqdm.write(f"      Total TopK: {file_total_topk_duration:.2f}s");
        tqdm.write(f"--- End File Summary for {os.path.basename(filepath)} ---")

    except FileNotFoundError:
        tqdm.write(f"Error: Input file not found {filepath}")
    except Exception as e:
        tqdm.write(f"Error processing file {filepath}: {e}\n{traceback.format_exc()}")


def main():
    # ... (main function remains the same as the last ONNX version) ...
    parser = argparse.ArgumentParser(description="Annotate text files using ONNX Runtime with extensive timing.")
    parser.add_argument("input_folder", type=str, help="Folder containing .train text files to annotate.")
    parser.add_argument("output_folder", type=str, help="Folder where annotated files will be saved.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the ONNX model file (e.g., ./onnx_model/model.onnx).")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to the directory containing tokenizer files (e.g., ./onnx_model/).")
    parser.add_argument("--spacy_model_name", type=str, default=SPACY_MODEL_NAME,
                        help=f"Name of the spaCy model for sentence tokenization (default: {SPACY_MODEL_NAME}).")
    parser.add_argument("--k_top", type=int, default=K_TOP,
                        help=f"Consider 'that' if it's in the top K predictions (default: {K_TOP}).")
    parser.add_argument("--chunk_size_chars", type=int, default=DEFAULT_CHUNK_READ_SIZE_CHARS,
                        help=f"Number of characters to read into memory at a time for processing (default: {DEFAULT_CHUNK_READ_SIZE_CHARS}).")
    args = parser.parse_args()
    script_start_time = time.perf_counter()
    device_str = get_device()
    ort_session, bert_tokenizer = load_onnx_session_and_tokenizer(args.model_path, args.tokenizer_path, device_str)
    spacy_nlp = load_spacy_model(args.spacy_model_name)
    if args.chunk_size_chars <= 0: print(
        f"Error: chunk_size_chars must be positive. Using default {DEFAULT_CHUNK_READ_SIZE_CHARS}."); args.chunk_size_chars = DEFAULT_CHUNK_READ_SIZE_CHARS
    if args.chunk_size_chars >= spacy_nlp.max_length: print(
        f"Warning: CHUNK_READ_SIZE_CHARS ({args.chunk_size_chars}) is >= spaCy's nlp.max_length ({spacy_nlp.max_length}).")
    try:
        that_token_id = bert_tokenizer.convert_tokens_to_ids('that')
        if isinstance(that_token_id, list): print(
            f"Warning: 'that' tokenized into multiple IDs: {that_token_id}. Using first."); that_token_id = \
        that_token_id[0]
        if that_token_id == bert_tokenizer.unk_token_id: print(
            f"Warning: 'that' is an unknown token ({bert_tokenizer.unk_token_id}).")
        that_token_id = int(that_token_id)
    except Exception as e:
        print(f"Error getting token ID for 'that': {e}. Cannot proceed."); return
    if not os.path.isdir(args.input_folder): print(f"Error: Input folder '{args.input_folder}' not found."); return
    if not os.path.exists(args.output_folder): os.makedirs(args.output_folder); print(
        f"Created output folder: {args.output_folder}")
    input_file_pattern = os.path.join(args.input_folder, "*.train")
    train_files = glob.glob(input_file_pattern)
    if not train_files: print(f"No .train files found in '{args.input_folder}'."); return
    print(f"Found {len(train_files)} .train files to process.")
    overall_start_time = time.perf_counter()
    for filepath_idx, filepath in enumerate(
            tqdm(train_files, desc="Overall File Progress", unit="file", position=0, dynamic_ncols=True)):
        base_filename = os.path.basename(filepath);
        output_filename = base_filename + ".annotated";
        output_filepath = os.path.join(args.output_folder, output_filename)
        tqdm.write(f"\n--- Starting File {filepath_idx + 1}/{len(train_files)}: {base_filename} ---")
        process_file(filepath, output_filepath, bert_tokenizer, ort_session, spacy_nlp, device_str, that_token_id,
                     args.k_top, args.chunk_size_chars)
        tqdm.write(f"--- Finished File {filepath_idx + 1}/{len(train_files)}: {base_filename} ---")
    overall_end_time = time.perf_counter();
    total_script_time = overall_end_time - overall_start_time
    print(
        f"Annotation process complete. Total script time: {total_script_time:.2f} seconds ({total_script_time / 3600:.2f} hours).")
    current_time_utc = time.gmtime();
    print(f"Finished at (UTC): {time.strftime('%Y-%m-%d %H:%M:%S %Z', current_time_utc)}")


if __name__ == "__main__":
    main()