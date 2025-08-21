# copy and modified from https://github.com/BUTSpeechFIT/DiCoW/blob/main/inference.py
import torch
import argparse
import os
import glob
import logging
from pathlib import Path

from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForSpeechSeq2Seq
from dicow_pipeline import DiCoWPipeline
from diarizen_pipeline import DiariZenPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def create_lower_uppercase_mapping(tokenizer):
    tokenizer.upper_cased_tokens = {}
    vocab = tokenizer.get_vocab()
    for token, index in vocab.items():
        if len(token) < 1:
            continue
        if token[0] == "Ġ" and len(token) > 1:
            lower_cased_token = (
                token[0] + token[1].lower() + (token[2:] if len(token) > 2 else "")
            )
        else:
            lower_cased_token = token[0].lower() + token[1:]
        if lower_cased_token != token:
            lower_index = vocab.get(lower_cased_token, None)
            if lower_index is not None:
                tokenizer.upper_cased_tokens[lower_index] = index
            else:
                pass


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process WAV files using DiCoW and DiariZen models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--dicow-model",
        type=str,
        default="BUT-FIT/DiCoW_v3_2",
        help="DiCoW model name or path",
    )

    parser.add_argument(
        "--diarization-model",
        type=str,
        default="BUT-FIT/diarizen-wavlm-large-s80-md",
        help="Diarization model name or path",
    )

    # Input/Output arguments
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Path to folder containing WAV files to process",
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default="./output",
        help="Path to output folder for results",
    )

    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*.wav",
        help="File pattern to match WAV files (e.g., '*.wav', 'recording_*.wav')",
    )

    # Processing arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for processing",
    )

    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=16,
        help="Batch size for embedding processing",
    )

    parser.add_argument(
        "--segmentation-batch-size",
        type=int,
        default=16,
        help="Batch size for segmentation processing",
    )

    parser.add_argument(
        "--verbose", type=str2bool, default=True, help="Enable verbose output"
    )

    return parser.parse_args()


def get_device(device_arg):
    """Get the appropriate device based on argument and availability."""
    if device_arg == "auto":
        return (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
    else:
        return torch.device(device_arg)


def find_wav_files(input_folder, file_pattern):
    """Find all WAV files in the input folder matching the pattern."""
    search_pattern = os.path.join(input_folder, file_pattern)
    wav_files = glob.glob(search_pattern)

    if not wav_files:
        logging.info(
            f"No WAV files found matching pattern '{file_pattern}' in '{input_folder}'"
        )
        return []

    return sorted(wav_files)


def process_audio_file(pipeline, audio_path, output_folder, verbose=False):
    """Process a single audio file and save the results as txt."""
    if verbose:
        logging.info(f"Processing: {audio_path}")

    try:
        # Process the audio file
        result = pipeline(audio_path, return_timestamps=True)

        # Prepare output filename
        audio_filename = Path(audio_path).stem
        output_path = os.path.join(output_folder, f"{audio_filename}.txt")

        # Save as text file
        with open(output_path, "w", encoding="utf-8") as f:
            if isinstance(result, dict) and "text" in result:
                f.write(result["text"])
            else:
                f.write(str(result))

        if verbose:
            logging.info(f"Saved result to: {output_path}")

        return output_path

    except Exception as e:
        logging.info(f"Error processing {audio_path}: {str(e)}")
        return None


def main():
    """Main function to run the audio processing pipeline."""
    args = parse_arguments()

    # Validate input folder
    if not os.path.exists(args.input_folder):
        logging.info(f"Error: Input folder '{args.input_folder}' does not exist.")
        return

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Get device
    device = get_device(args.device)
    if args.verbose:
        logging.info(f"Using device: {device}")

    # Find WAV files
    wav_files = find_wav_files(args.input_folder, args.file_pattern)
    if not wav_files:
        return

    if args.verbose:
        logging.info(f"Found {len(wav_files)} WAV files to process")

    # Load models
    if args.verbose:
        logging.info("Loading DiCoW model...")

    dicow = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.dicow_model, trust_remote_code=True
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.dicow_model)
    tokenizer = AutoTokenizer.from_pretrained(args.dicow_model)
    create_lower_uppercase_mapping(tokenizer)
    dicow.set_tokenizer(tokenizer)

    if args.verbose:
        logging.info("Loading diarization model...")

    diar_pipeline = DiariZenPipeline.from_pretrained(args.diarization_model).to(device)
    diar_pipeline.embedding_batch_size = args.embedding_batch_size
    diar_pipeline.segmentation_batch_size = args.segmentation_batch_size

    # Create pipeline
    pipeline = DiCoWPipeline(
        dicow,
        diarization_pipeline=diar_pipeline,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        device=device,
    )

    if args.verbose:
        logging.info("Pipeline initialized successfully")

    # Process files
    successful_files = 0
    for audio_file in wav_files:
        result = process_audio_file(
            pipeline, audio_file, args.output_folder, args.verbose
        )
        if result:
            successful_files += 1

    logging.info(
        f"\nProcessing complete! Successfully processed {successful_files}/{len(wav_files)} files."
    )
    logging.info(f"Results saved to: {args.output_folder}")


if __name__ == "__main__":
    main()
