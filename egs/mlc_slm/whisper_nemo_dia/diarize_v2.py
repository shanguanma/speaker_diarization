# copy and modified from https://github.com/MahmoudAshraf97/whisper-diarization/blob/main/diarize.py
import argparse
import logging
import os
import re
import glob
from pathlib import Path

import faster_whisper
import torch

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel

from helpers import (
    cleanup,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)

mtypes = {"cpu": "int8", "cuda": "float16"}

pid = os.getpid()
temp_outputs_dir = f"temp_outputs_{pid}"
temp_path = os.path.join(os.getcwd(), "temp_outputs")
os.makedirs(temp_path, exist_ok=True)

def parse_arguments():
    """Parse command line arguments."""
    # Initialize parser
    parser = argparse.ArgumentParser()
    #parser.add_argument(
    #    "-a", "--audio", help="name of the target audio file", required=True
    #)
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
    parser.add_argument(
        "--no-stem",
        action="store_false",
        dest="stemming",
        default=True,
        help="Disables source separation."
        "This helps with long files that don't contain a lot of music.",
    )

    parser.add_argument(
        "--suppress_numerals",
        action="store_true",
        dest="suppress_numerals",
        default=False,
        help="Suppresses Numerical Digits."
        "This helps the diarization accuracy but converts all digits into written text.",
    )

    parser.add_argument(
        "--whisper-model",
        dest="model_name",
        default="medium.en",
        help="name of the Whisper model to use",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        default=8,
        help="Batch size for batched inference, reduce if you run out of memory, "
        "set to 0 for original whisper longform inference",
    )

    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=whisper_langs,
        help="Language spoken in the audio, specify None to perform language detection",
    )

    parser.add_argument(
        "--device",
        dest="device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="if you have a GPU use 'cuda', otherwise 'cpu'",
    )
    

    # Processing arguments
    #parser.add_argument(
    #    "--device",
    #    type=str,
    #    default="auto",
    #    choices=["auto", "cpu", "cuda"],
    #    help="Device to use for processing",
    #)

    parser.add_argument(
        "--diarizer",
        default="msdd",
        choices=["msdd"],
        help="Choose the diarization model to use",
    )
    parser.add_argument("--msdd-model-path", type=str, default="diar_msdd_telephonic", help="if it is name str , it will load nvidia nemo pretrained model, if it is model path, it will load a local model")
    parser.add_argument("--use-speaker-model-from-ckpt", type=str2bool, default=True, help="")
    parser.add_argument("--realign-with-punc", type=str2bool, default=True, help="Use the punctuation alignment model to realign speaker and txt") 
    parser.add_argument("--align-model", type=str, default="MahmoudAshraf/mms-300m-1130-forced-aligner",
            help="""choice ssl ctc model from huggingface,i.e.:https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py""")
    parser.add_argument("--local-files-only", type=str2bool, default=False, help="if local_files_only=True, args.align_model is local directory.")
    args = parser.parse_args()
    return args



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


def find_wav_files(input_folder, file_pattern):
    """Find all WAV files in the input folder matching the pattern."""
    search_pattern = os.path.join(input_folder, file_pattern)
    wav_files = glob.glob(search_pattern)
    # remove resample.wav
    wav_files = [wav for wav in wav_files if "resampled" not in wav]
    if not wav_files:
        logging.info(
            f"No WAV files found matching pattern '{file_pattern}' in '{input_folder}'"
        )
        return []

    return sorted(wav_files)



def gen_txt_with_ws(args,audio_path):
    # Get device
    #device = get_device(args.device)
    #logging.info(f"Using device: {args.device}")
    
    logging.info(f"Loading faster whisper args.model_name model from {args.device}...")
    whisper_model = faster_whisper.WhisperModel(
        args.model_name, device=args.device, compute_type=mtypes[args.device],local_files_only=args.local_files_only,
    )
    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
    
    vocal_target=audio_path
    audio_waveform = faster_whisper.decode_audio(vocal_target) # float32 audio sample
    suppress_tokens = (
        find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
        if args.suppress_numerals
        else [-1]
    )
    
    # get language
    language = process_language_arg(args.language, args.model_name)
    logging.info(f"specified {language}")
    

    if args.batch_size > 0:
        transcript_segments, info = whisper_pipeline.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            batch_size=args.batch_size,
        )
    else:
        transcript_segments, info = whisper_model.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            vad_filter=True,
        )

    full_transcript = "".join(segment.text for segment in transcript_segments)

    # clear gpu vram
    del whisper_model, whisper_pipeline
    torch.cuda.empty_cache()

    # Forced Alignment
    logging.info(f"Forced Alignment....")
    #alignment_model, alignment_tokenizer, align_dictionary = load_alignment_model(
    alignment_model,alignment_meta = load_alignment_model(
        args.device,
        model_path=args.align_model,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    if alignment_meta["tokenizer"] is None:
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
        emissions, stride = generate_emissions(
            alignment_model,
            alignment_meta,
            torch.from_numpy(audio_waveform)
            .to(torch.float32)
            .to(args.device),
            batch_size=args.batch_size,
        )

    else:
        emissions, stride = generate_emissions(
            alignment_model,
            alignment_meta,
            torch.from_numpy(audio_waveform)
            .to(alignment_model.dtype)
            .to(alignment_model.device),
            batch_size=args.batch_size,
        )

    del alignment_model
    torch.cuda.empty_cache()

    tokens_starred, text_starred = preprocess_text(
        full_transcript,
        romanize=True,
        language=langs_to_iso[info.language],
    )
    alignment_tokenizer = alignment_meta["tokenizer"]
    alignment_dictionary = alignment_meta["dictionary"]
    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
        alignment_dictionary,
    )

    spans = get_spans(tokens_starred, segments, blank_token)

    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    if args.diarizer == "msdd":
        from diarization import MSDDDiarizer
        logging.info("Loadding nemo msdd diarization model...")
        diarizer_model = MSDDDiarizer(args,device=args.device)
    

    logging.info("Diarization ....")
    speaker_ts = diarizer_model.diarize(torch.from_numpy(audio_waveform).unsqueeze(0))
    
    del diarizer_model
    torch.cuda.empty_cache()
    
    logging.info("words speaker mapping ....")
    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
    return wsm, info, speaker_ts


def realign_with_punc(info, wsm):
    if info.language in punct_model_langs:
        # restoring punctuation in the transcript to help realign the sentences
        punct_model = PunctuationModel(model="kredor/punctuate-all")

        words_list = list(map(lambda x: x["word"], wsm))

        labled_words = punct_model.predict(words_list, chunk_size=230)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # We don't want to punctuate U.S.A. with a period. Right?
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

    else:
        logging.warning(
            f"Punctuation restoration is not available for {info.language} language."
            " Using the original punctuation."
        )
    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    return wsm


def process_single_audio(args,audio_path,verbose=True):
    
    wsm, info, speaker_ts = gen_txt_with_ws(args, audio_path)
    if args.realign_with_punc:
        logging.info(f"using punc to realign....")
        wsm = realign_with_punc(info, wsm)

    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    # Prepare output filename
    audio_filename = Path(audio_path).stem
    output_path = os.path.join(args.output_folder, f"{audio_filename}.txt")
    output_path_srt = os.path.join(args.output_folder, f"{audio_filename}.srt")

    with open(f"{output_path}", "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    with open(f"{output_path_srt}", "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)

        if verbose:
            logging.info(f"Saved result to: {output_path}")
            logging.info(f"Saved result to: {output_path_srt}")
        return output_path
    cleanup(temp_path)

def main():
    args = parse_arguments()
    
    # Validate input folder
    if not os.path.exists(args.input_folder):
        logging.info(f"Error: Input folder '{args.input_folder}' does not exist.")
        return

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Find WAV files
    wav_files = find_wav_files(args.input_folder, args.file_pattern)
    # Process files
    successful_files = 0
    for audio_file in wav_files:
        result = process_single_audio(args,audio_file)
        if result:
            successful_files += 1

    logging.info(
        f"\nProcessing complete! Successfully processed {successful_files}/{len(wav_files)} files."
    )
    logging.info(f"Results saved to: {args.output_folder}")

if __name__ == "__main__":
    main()
