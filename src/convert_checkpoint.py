import torch
import argparse
from pathlib import Path
import logging
from transformers import AutoConfig, GPT2LMHeadModel # Use the specific model class from your training

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def convert_checkpoint(checkpoint_dir: Path):
    """
    Loads model weights from 'training_state.pt' within a checkpoint directory
    and saves them in the standard 'pytorch_model.bin' (or 'model.safetensors')
    format within the same directory.

    Args:
        checkpoint_dir (Path): Path object pointing to the checkpoint directory.

    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    logger.info(f"Starting conversion for checkpoint directory: {checkpoint_dir}")

    # --- Validate Input Paths ---
    if not checkpoint_dir.is_dir():
        logger.error(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return False

    config_file = checkpoint_dir / "config.json"
    state_file = checkpoint_dir / "training_state.pt"
    # Define potential output filenames based on transformers standard
    output_model_file_bin = checkpoint_dir / "pytorch_model.bin"
    output_model_file_safetensors = checkpoint_dir / "model.safetensors"

    if not config_file.is_file():
        logger.error(f"Error: config.json not found in {checkpoint_dir}")
        return False

    if not state_file.is_file():
        logger.error(f"Error: {state_file.name} not found in {checkpoint_dir}")
        return False

    # Check if standard weights file already exists
    if output_model_file_bin.is_file() or output_model_file_safetensors.is_file():
        logger.warning(
            f"Warning: Standard model weight file ('{output_model_file_bin.name}' or "
            f"'{output_model_file_safetensors.name}') already exists in {checkpoint_dir}. "
            "Overwriting."
        )
        # You could add logic here to skip if you prefer not to overwrite:
        # logger.info("Skipping conversion as standard weights file already exists.")
        # return True

    # --- Load Config ---
    try:
        logger.info(f"Loading model configuration from {config_file}...")
        # Load the config associated with the checkpoint
        config = AutoConfig.from_pretrained(checkpoint_dir)
        logger.info("Configuration loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        return False

    # --- Load Training State ---
    try:
        logger.info(f"Loading training state from {state_file} (using map_location='cpu')...")
        # Load state dictionary to CPU first to avoid device mismatch issues
        # *** ADD weights_only=False due to PyTorch 2.6+ defaults and saved argparse.Namespace ***
        checkpoint_state = torch.load(state_file, map_location='cpu', weights_only=False)
        logger.info("Training state loaded.")
    except Exception as e:
        logger.error(f"Failed to load training state file {state_file}: {e}", exc_info=True)
        return False

    # --- Extract Model State Dict ---
    if 'model' not in checkpoint_state:
        logger.error(f"Error: 'model' key not found in the loaded state dictionary from {state_file}.")
        return False
    model_state_dict = checkpoint_state['model']
    logger.info("Extracted model state dictionary (key='model').")

    # --- Instantiate Model and Load Weights ---
    try:
        logger.info(f"Instantiating blank model structure ({config.model_type} - assuming GPT2LMHeadModel)...")
        # Instantiate the model structure using the loaded config.
        # **Important**: Use the same model class as your training script (GPT2LMHeadModel).
        model = GPT2LMHeadModel(config=config)
        logger.info("Model structure instantiated.")

        logger.info("Loading extracted state dict into model...")
        # Load the weights into the instantiated model structure
        model.load_state_dict(model_state_dict)
        logger.info("Model weights loaded successfully into the model instance.")
    except Exception as e:
        logger.error(f"Failed to instantiate model or load state dict: {e}", exc_info=True)
        return False

    # --- Save Model Weights in Standard Format ---
    try:
        logger.info(f"Saving model weights to {checkpoint_dir} using model.save_pretrained()...")
        # This crucial step saves the weights into pytorch_model.bin (or model.safetensors)
        # in the SAME directory.
        model.save_pretrained(checkpoint_dir)
        logger.info(f"Standard model weights saved successfully to {checkpoint_dir}.")
    except Exception as e:
        logger.error(f"Failed to save model weights using save_pretrained: {e}", exc_info=True)
        return False

    logger.info(f"Conversion complete for {checkpoint_dir}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a custom training checkpoint by extracting model weights "
                    "from 'training_state.pt' and saving them in the standard "
                    "Hugging Face format (e.g., 'pytorch_model.bin') within the same directory."
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to the checkpoint directory containing training_state.pt and config.json"
    )

    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint_dir).resolve() # Get absolute path

    if convert_checkpoint(checkpoint_path):
        logger.info("Script finished successfully.")
    else:
        logger.error("Script finished with errors.")
        exit(1) # Exit with a non-zero code to indicate failure