# convert_to_onnx.py
from optimum.onnxruntime import ORTModelForMaskedLM
from transformers import AutoTokenizer
import argparse
import os
import time

def main():
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face Masked LM model to ONNX format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased", # Default to faster model
        help="Name/path of the Hugging Face model (e.g., distilbert-base-uncased, bert-base-uncased)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory path to save the exported ONNX model and associated tokenizer/config files."
    )
    # Optional: Add arguments for ONNX optimizations like quantization later if needed
    # parser.add_argument("--quantize", action="store_true", help="Apply INT8 quantization (requires further setup).")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    start_time = time.perf_counter()
    print(f"Loading and exporting '{args.model_name}' to ONNX format...")

    try:
        # Load from Hub, convert to ONNX, and save all components (model.onnx, config.json etc.)
        # The 'export=True' flag triggers the conversion during loading.
        ort_model = ORTModelForMaskedLM.from_pretrained(args.model_name, export=True)
        ort_model.save_pretrained(args.output_dir)
        print(f"ONNX model components saved to {args.output_dir}")

        print(f"Saving tokenizer for '{args.model_name}'...")
        # Also save the corresponding tokenizer to the same directory
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Tokenizer saved to {args.output_dir}")

    except Exception as e:
        print(f"\nError during conversion or saving:")
        print(e)
        import traceback
        traceback.print_exc()
        return

    end_time = time.perf_counter()
    print(f"\nSuccessfully converted model and saved components to: {args.output_dir}")
    print(f"Conversion took {end_time - start_time:.2f} seconds.")
    print("Files should include model.onnx (or model_optimized.onnx), config.json, tokenizer files, etc.")

if __name__ == "__main__":
    main()