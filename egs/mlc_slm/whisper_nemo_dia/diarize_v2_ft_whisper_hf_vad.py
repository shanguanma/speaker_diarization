# copy and modified from https://github.com/MahmoudAshraf97/whisper-diarization/blob/main/diarize.py
# Added VAD support using silero-vad_v5 for better long audio transcription with finetuned Whisper models
import argparse
import logging
import os
import re
import glob
from pathlib import Path

import numpy as np
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

from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa

# Import VAD functions
from fireredasr.models.vad import (
    VadOptions,
    get_speech_timestamps,
    collect_chunks,
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
        help="Batch size for batched inference, reduce if you run out of memory",
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
    
    parser.add_argument(
        "--diarizer",
        default="msdd",
        choices=["msdd"],
        help="Choose the diarization model to use",
    )
    parser.add_argument("--msdd-model-path", type=str, default="diar_msdd_telephonic", help="if it is name str , it will load nvidia nemo pretrained model, if it is model path, it will load a local model")
    parser.add_argument("--use-speaker-model-from-ckpt", type=str2bool, default=True, help="")
    parser.add_argument("--realign-with-punc", type=str2bool, default=True, help="Use the punctuation alignment model to realign speaker and txt") 
    
    parser.add_argument("--punct-model", type=str,default="kredor/punctuate-all", help="it is str or local model path")

    parser.add_argument("--align-model", type=str, default="MahmoudAshraf/mms-300m-1130-forced-aligner",
            help="""choice ssl ctc model from huggingface,i.e.:https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py""")
    parser.add_argument("--local-files-only", type=str2bool, default=False, help="if local_files_only=True, args.model_name(asr model) is local directory.")
    parser.add_argument("--ctc-model-local-files-only", type=str2bool, default=False, help="if ctc_model_local_files_only=True, args.align_model is local directory.")

    # VAD related arguments
    parser.add_argument("--use-vad", type=str2bool, default=True, help="Enable VAD for long audio processing (default: True)")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="VAD speech threshold (default: 0.5)")
    parser.add_argument("--vad-min-speech-duration-ms", type=int, default=0, help="Minimum speech duration in milliseconds (default: 0)")
    parser.add_argument("--vad-max-speech-duration-s", type=float, default=30.0, help="Maximum speech duration in seconds for VAD chunks (default: 30.0)")
    parser.add_argument("--vad-min-silence-duration-ms", type=int, default=160, help="Minimum silence duration in milliseconds (default: 500)")
    parser.add_argument("--vad-speech-pad-ms", type=int, default=400, help="Speech padding in milliseconds (default: 400)")

    args = parser.parse_args()
    return args

def prepare_audio_for_whisper(audio_data, target_sr=16000):
    """
    准备音频数据以符合Whisper输入格式
    
    参数:
    - audio_data: 原始音频数据，可以是多种格式
    - target_sr: 目标采样率，Whisper要求16kHz
    
    返回:
    - 处理后的音频字典，包含array和sampling_rate
    """
    
    # 情况1: 如果audio_data已经是字典格式（包含array和sampling_rate）
    if isinstance(audio_data, dict) and 'array' in audio_data and 'sampling_rate' in audio_data:
        audio_array = audio_data['array']
        current_sr = audio_data['sampling_rate']
        
        # 确保采样率为16kHz
        if current_sr != target_sr:
            audio_array = librosa.resample(audio_array, orig_sr=current_sr, target_sr=target_sr)
        
        return {"array": audio_array, "sampling_rate": target_sr}
    
    # 情况2: 如果audio_data是numpy数组，但没有采样率信息
    elif isinstance(audio_data, np.ndarray):
        # 假设采样率已经是16kHz，如果不是，你需要知道原始采样率
        # 这里假设已经是正确格式，实际使用时需要根据你的数据调整
        return {"array": audio_data, "sampling_rate": target_sr}
    
    # 情况3: 如果audio_data是文件路径
    elif isinstance(audio_data, str):
        # 从文件加载音频
        try:
            audio_array, original_sr = librosa.load(audio_data, sr=target_sr)
            return {"array": audio_array, "sampling_rate": target_sr}
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None
    
    # 情况4: 如果是torch tensor
    elif isinstance(audio_data, torch.Tensor):
        audio_array = audio_data.numpy() if audio_data.device.type == 'cpu' else audio_data.cpu().numpy()
        return {"array": audio_array, "sampling_rate": target_sr}
    
    else:
        raise ValueError(f"Unsupported audio data type: {type(audio_data)}")


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


def format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS.mmm format"""
    milliseconds = int(seconds * 1000)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    secs = milliseconds // 1_000
    milliseconds -= secs * 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def transcribe_with_vad(args, audio_path, model, processor):
    """
    使用 VAD 对长音频进行分段转录
    
    Args:
        args: 命令行参数
        audio_path: 音频文件路径
        model: Whisper模型
        processor: Whisper处理器
    
    Returns:
        完整的转录文本
    """
    # 加载音频
    audio_waveform = faster_whisper.decode_audio(audio_path)
    duration = len(audio_waveform) / 16000
    logging.info(f"Processing audio with duration {format_timestamp(duration)}")
    
    # 应用 VAD 分段
    if duration < args.vad_max_speech_duration_s:
        # 短音频，不需要 VAD
        logging.info("Audio is short, processing without VAD")
        clip_timestamps = [{"start": 0, "end": len(audio_waveform)}]
    else:
        # 长音频，使用 VAD
        logging.info("Applying VAD for long audio...")
        vad_parameters = VadOptions(
            threshold=args.vad_threshold,
            min_speech_duration_ms=args.vad_min_speech_duration_ms,
            max_speech_duration_s=args.vad_max_speech_duration_s,
            min_silence_duration_ms=args.vad_min_silence_duration_ms,
            speech_pad_ms=args.vad_speech_pad_ms,
        )
        clip_timestamps = get_speech_timestamps(audio_waveform, vad_parameters, sampling_rate=16000)
    
    # 收集音频块
    audio_chunks, chunks_metadata = collect_chunks(
        audio_waveform, 
        clip_timestamps, 
        sampling_rate=16000,
        max_duration=args.vad_max_speech_duration_s
    )
    
    # 计算 VAD 后的总时长
    duration_after_vad = sum([metadata["duration"] for metadata in chunks_metadata])
    logging.info(f"VAD filter removed {format_timestamp(duration - duration_after_vad)} of audio")
    logging.info(f"Number of speech chunks: {len(audio_chunks)}")
    
    # 如果没有检测到语音
    if not audio_chunks or len(audio_chunks) == 0:
        logging.warning("No speech detected by VAD")
        return ""
    
    # 对每个音频块进行转录
    all_transcripts = []
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language, 
        task="transcribe"
    )
    
    for i, (chunk, metadata) in enumerate(zip(audio_chunks, chunks_metadata)):
        if len(chunk) == 0:
            logging.warning(f"Chunk {i+1}/{len(audio_chunks)} is empty, skipping")
            continue
        
        logging.info(f"Transcribing chunk {i+1}/{len(audio_chunks)} (duration: {metadata['duration']:.2f}s)")
        
        try:
            # 处理音频块
            input_features = processor(
                chunk,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(args.device)
            
            # 生成转录
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_length=448,
                    num_beams=1,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )
            
            # 解码
            transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            if transcript.strip():
                all_transcripts.append(transcript.strip())
                logging.debug(f"Chunk {i+1} transcript: {transcript[:100]}...")
            
        except Exception as e:
            logging.error(f"Error transcribing chunk {i+1}: {e}")
            continue
    
    # 合并所有转录
    full_transcript = " ".join(all_transcripts)
    logging.info(f"Total transcribed text length: {len(full_transcript)} characters")
    
    return full_transcript


def gen_txt_with_ws(args, audio_path):
    """
    生成带词级时间戳的转录
    """
    logging.info(f"Loading Whisper model {args.model_name} from {args.device}... (local_files_only={args.local_files_only})")
    
    vocal_target = audio_path
    audio_waveform = faster_whisper.decode_audio(vocal_target)  # float32 audio sample
    
    # 设置语言
    language = "en" if args.language is None else args.language
    logging.info(f"Specified language: {language}")
    
    # 加载微调后的模型和处理器
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name, local_files_only=args.local_files_only).to(args.device)
    processor = WhisperProcessor.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    
    # 使用 VAD 进行转录
    if args.use_vad:
        logging.info("Using VAD-based transcription for better quality on long audio")
        full_transcript = transcribe_with_vad(args, vocal_target, model, processor)
    else:
        # 不使用 VAD，直接处理（可能效果较差）
        logging.info("Processing without VAD (may include silence segments)")
        audio_array, sr = librosa.load(vocal_target, sr=16000)
        
        input_features = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(args.device)
        
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language,
            task="transcribe"
        )
        
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=448,
                num_beams=1,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        
        full_transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    logging.info(f"Audio path: {vocal_target}")
    logging.info(f"Whisper output transcript length: {len(full_transcript)} characters")
    logging.info(f"Transcript preview: {full_transcript[:200]}...")
    
    # 清理 GPU 显存
    del model
    torch.cuda.empty_cache()

    # Forced Alignment
    logging.info(f"Forced Alignment....")
    alignment_model, alignment_meta = load_alignment_model(
        args.device,
        model_path=args.align_model,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
        local_files_only=args.ctc_model_local_files_only,
    )
    if alignment_meta["tokenizer"] is None:
        dtype = torch.float16 if args.device == "cuda" else torch.float32
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
        language=langs_to_iso["en"],
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
        logging.info("Loading nemo msdd diarization model...")
        diarizer_model = MSDDDiarizer(args, device=args.device)
    
    logging.info("Diarization ....")
    speaker_ts = diarizer_model.diarize(torch.from_numpy(audio_waveform).unsqueeze(0))
    
    del diarizer_model
    torch.cuda.empty_cache()
    
    logging.info("Words speaker mapping ....")
    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
    return wsm, speaker_ts


def realign_with_punc(wsm, args):
    if "en" in punct_model_langs:
        # restoring punctuation in the transcript to help realign the sentences
        punct_model = PunctuationModel(model=args.punct_model)

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
            f"Punctuation restoration is not available for en language."
            " Using the original punctuation."
        )
    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    return wsm


def process_single_audio(args, audio_path, verbose=True):
    
    wsm, speaker_ts = gen_txt_with_ws(args, audio_path)
    if args.realign_with_punc:
        logging.info(f"Using punctuation to realign....")
        wsm = realign_with_punc(wsm, args)

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
    
    # Log VAD settings
    if args.use_vad:
        logging.info("=" * 80)
        logging.info("VAD Settings:")
        logging.info(f"  Threshold: {args.vad_threshold}")
        logging.info(f"  Min speech duration: {args.vad_min_speech_duration_ms} ms")
        logging.info(f"  Max speech duration: {args.vad_max_speech_duration_s} s")
        logging.info(f"  Min silence duration: {args.vad_min_silence_duration_ms} ms")
        logging.info(f"  Speech padding: {args.vad_speech_pad_ms} ms")
        logging.info("=" * 80)
    else:
        logging.info("VAD is disabled")
    
    # Process files
    successful_files = 0
    for audio_file in wav_files:
        result = process_single_audio(args, audio_file)
        if result:
            successful_files += 1

    logging.info(
        f"\nProcessing complete! Successfully processed {successful_files}/{len(wav_files)} files."
    )
    logging.info(f"Results saved to: {args.output_folder}")

if __name__ == "__main__":
    main()

