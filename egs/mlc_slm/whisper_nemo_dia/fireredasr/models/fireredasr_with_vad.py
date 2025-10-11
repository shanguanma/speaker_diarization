import os
import time
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, BinaryIO
import logging
import torch

from fireredasr.data.asr_feat import ASRFeatExtractor
from fireredasr.models.fireredasr_aed import FireRedAsrAed
from fireredasr.models.fireredasr_llm import FireRedAsrLlm
from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
from fireredasr.tokenizer.llm_tokenizer import LlmTokenizerWrapper
from fireredasr.models.vad import (
    SpeechTimestampsMap,
    VadOptions,
    collect_chunks,
    get_speech_timestamps,
)

        

def format_timestamp(
    seconds: float,
    always_include_hours: bool = False,
    decimal_marker: str = ".",
) -> str:
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )

def decode_audio(
    input_file: Union[str, BinaryIO],
    sampling_rate: int = 16000,
    split_stereo: bool = False,
):
    """Decodes the audio using faster-whisper's decode_audio function.

    Args:
      input_file: Path to the input file or a file-like object.
      sampling_rate: Resample the audio to this sample rate.
      split_stereo: Return separate left and right channels.

    Returns:
      A float32 Numpy array.

      If `split_stereo` is enabled, the function returns a 2-tuple with the
      separated left and right channels.
    """
    # Import faster-whisper's decode_audio function
    from faster_whisper.audio import decode_audio as faster_whisper_decode_audio
    return faster_whisper_decode_audio(input_file, sampling_rate, split_stereo)


#def vad_filter(batch_wav_path,sampling_rate=16000,chunk_length=30):
#    for audio in batch_wav_path:
#        if not isinstance(audio, np.ndarray):
#            audio = decode_audio(audio, sampling_rate=sampling_rate)
#        duration = audio.shape[0] / sampling_rate
#        print("Processing audio with duration %s", format_timestamp(duration))
#        chunk_length=chunk_length # i.e.30 its unit is second.
#        if duration < chunk_length:
#            clip_timestamps = [{"start": 0, "end": audio.shape[0]}]
#        else:
#            vad_parameters = VadOptions(max_speech_duration_s=chunk_length,min_silence_duration_ms=160,)
#            clip_timestamps = get_speech_timestamps(audio, vad_parameters)
#
#        audio_chunks, chunks_metadata = collect_chunks(audio, clip_timestamps, max_duration=chunk_length)
#        duration_after_vad = (
#            sum((segment["end"] - segment["start"]) for segment in clip_timestamps)
#            / sampling_rate
#        )
#        print("VAD filter removed %s of audio",format_timestamp(duration - duration_after_vad))
#        features = (
#            [self.model.feature_extractor(chunk)[..., :-1] for chunk in audio_chunks]
#            if duration_after_vad
#            else []
#        )

class FireRedAsr:
    @classmethod
    def from_pretrained(cls, asr_type, model_dir):
        assert asr_type in ["aed", "llm"]

        cmvn_path = os.path.join(model_dir, "cmvn.ark")
        feat_extractor = ASRFeatExtractor(cmvn_path)

        if asr_type == "aed":
            model_path = os.path.join(model_dir, "model.pth.tar")
            dict_path =os.path.join(model_dir, "dict.txt")
            spm_model = os.path.join(model_dir, "train_bpe1000.model")
            model = load_fireredasr_aed_model(model_path)
            tokenizer = ChineseCharEnglishSpmTokenizer(dict_path, spm_model)
        elif asr_type == "llm":
            model_path = os.path.join(model_dir, "model.pth.tar")
            encoder_path = os.path.join(model_dir, "asr_encoder.pth.tar")
            llm_dir = os.path.join(model_dir, "Qwen2-7B-Instruct")
            model, tokenizer = load_firered_llm_model_and_tokenizer(
                model_path, encoder_path, llm_dir)
        model.eval()
        return cls(asr_type, feat_extractor, model, tokenizer)

    def __init__(self, asr_type, feat_extractor, model, tokenizer):
        self.asr_type = asr_type
        self.feat_extractor = feat_extractor
        self.model = model
        self.tokenizer = tokenizer
        self.sampling_rate = 16000

    def apply_vad_filter(
        self, 
        audio: np.ndarray, 
        vad_parameters: Optional[VadOptions] = None,
        chunk_length: float = 30.0
    ) -> Tuple[List[np.ndarray], List[Dict[str, float]], List[Dict[str, int]]]:
        """
        Apply VAD filter to audio and return chunks with metadata.
        
        Args:
            audio: Input audio array
            vad_parameters: VAD parameters, if None uses default
            chunk_length: Maximum chunk length in seconds
            
        Returns:
            Tuple of (audio_chunks, chunks_metadata, clip_timestamps)
        """
        duration = audio.shape[0] / self.sampling_rate
        logging.info(f"Processing audio with duration {format_timestamp(duration)}")
        
        if duration < chunk_length:
            clip_timestamps = [{"start": 0, "end": audio.shape[0]}]
        else:
            if vad_parameters is None:
                vad_parameters = VadOptions(
                    max_speech_duration_s=chunk_length,
                    min_silence_duration_ms=160,
                )
            clip_timestamps = get_speech_timestamps(audio, vad_parameters)
        
        audio_chunks, chunks_metadata = collect_chunks(
            audio, clip_timestamps, max_duration=chunk_length
        )
        
        duration_after_vad = (
            sum((segment["end"] - segment["start"]) for segment in clip_timestamps)
            / self.sampling_rate
        )
        logging.info(f"VAD filter removed {format_timestamp(duration - duration_after_vad)} of audio")
        
        return audio_chunks, chunks_metadata, clip_timestamps

    def transcribe_with_vad(
        self, 
        batch_uttid: List[str], 
        batch_wav_path: List[str], 
        args: Dict = {},
        vad_parameters: Optional[VadOptions] = None,
        chunk_length: float = 30.0
    ) -> List[Dict]:
        """
        Transcribe audio with VAD filtering for long audio support.
        
        Args:
            batch_uttid: List of utterance IDs
            batch_wav_path: List of audio file paths
            args: Transcription arguments
            vad_parameters: VAD parameters
            chunk_length: Maximum chunk length in seconds
            
        Returns:
            List of transcription results
        """
        all_results = []
        
        for uttid, wav_path in zip(batch_uttid, batch_wav_path):
            # Load audio
            audio = decode_audio(wav_path, sampling_rate=self.sampling_rate)
            
            # Apply VAD filter
            audio_chunks, chunks_metadata, clip_timestamps = self.apply_vad_filter(
                audio, vad_parameters, chunk_length
            )
            
            if not audio_chunks:
                # No speech detected
                all_results.append({
                    "uttid": uttid, 
                    "text": "", 
                    "wav": wav_path,
                    "rtf": "0.0000"
                })
                continue
            
            # Process each chunk
            chunk_results = []
            total_duration = 0
            start_time = time.time()
            
            for i, (chunk, metadata) in enumerate(zip(audio_chunks, chunks_metadata)):
                if len(chunk) == 0:
                    continue
                    
                # Create temporary file for this chunk
                temp_wav_path = f"/tmp/temp_chunk_{uttid}_{i}.wav"
                import soundfile as sf
                sf.write(temp_wav_path, chunk, self.sampling_rate)
                
                # Transcribe this chunk
                chunk_result = self.transcribe([uttid], [temp_wav_path], args)
                if chunk_result:
                    chunk_results.extend(chunk_result)
                
                # Clean up temp file
                os.remove(temp_wav_path)
                
                total_duration += metadata["duration"]
            
            # Combine results
            if chunk_results:
                # Combine all text from chunks
                combined_text = " ".join([result["text"] for result in chunk_results])
                
                # Calculate RTF
                elapsed = time.time() - start_time
                rtf = elapsed / total_duration if total_duration > 0 else 0
                
                all_results.append({
                    "uttid": uttid,
                    "text": combined_text,
                    "wav": wav_path,
                    "rtf": f"{rtf:.4f}",
                    "chunks": len(audio_chunks),
                    "vad_timestamps": clip_timestamps
                })
            else:
                all_results.append({
                    "uttid": uttid, 
                    "text": "", 
                    "wav": wav_path,
                    "rtf": "0.0000"
                })
        
        return all_results

#    def vad_filter(self, batch_wav_path,sampling_rate=16000,chunk_length=30):
#        for audio in batch_wav_path:
#            if not isinstance(audio, np.ndarray):
#                audio = decode_audio(audio, sampling_rate=sampling_rate)
#            duration = audio.shape[0] / sampling_rate
#            print("Processing audio with duration %s", format_timestamp(duration))
#            chunk_length=chunk_length # i.e.30 its unit is second.
#            if duration < chunk_length:
#                clip_timestamps = [{"start": 0, "end": audio.shape[0]}]
#            else:
#                vad_parameters = VadOptions(max_speech_duration_s=chunk_length,min_silence_duration_ms=160,)
#                clip_timestamps = get_speech_timestamps(audio, vad_parameters)
#
#            audio_chunks, chunks_metadata = collect_chunks(audio, clip_timestamps, max_duration=chunk_length)
#            duration_after_vad = (
#                sum((segment["end"] - segment["start"]) for segment in clip_timestamps)
#                / sampling_rate
#            )
#            print("VAD filter removed %s of audio",format_timestamp(duration - duration_after_vad))
#            features = (
#                [self.model.feature_extractor(chunk)[..., :-1] for chunk in audio_chunks]
#                if duration_after_vad
#                else []
#            )
    @torch.no_grad()
    def transcribe(self, batch_uttid, batch_wav_path, args={}, use_vad=False, vad_parameters=None, chunk_length=30.0):
        """
        Transcribe audio with optional VAD support for long audio processing.
        
        Args:
            batch_uttid: List of utterance IDs
            batch_wav_path: List of audio file paths
            args: Transcription arguments
            use_vad: Whether to use VAD for long audio processing
            vad_parameters: VAD parameters
            chunk_length: Maximum chunk length in seconds for VAD
            
        Returns:
            List of transcription results
        """
        if use_vad:
            return self.transcribe_with_vad(batch_uttid, batch_wav_path, args, vad_parameters, chunk_length)
        
        # Original transcribe logic without VAD
        feats, lengths, durs = self.feat_extractor(batch_wav_path)
        total_dur = sum(durs)
        if args.get("use_gpu", False):
            feats, lengths = feats.cuda(), lengths.cuda()
            self.model.cuda()
        else:
            self.model.cpu()

        if self.asr_type == "aed":
            start_time = time.time()

            hyps = self.model.transcribe(
                feats, lengths,
                args.get("beam_size", 1),
                args.get("nbest", 1),
                args.get("decode_max_len", 0),
                args.get("softmax_smoothing", 1.0),
                args.get("aed_length_penalty", 0.0),
                args.get("eos_penalty", 1.0)
            )

            elapsed = time.time() - start_time
            rtf= elapsed / total_dur if total_dur > 0 else 0

            results = []
            for uttid, wav, hyp in zip(batch_uttid, batch_wav_path, hyps):
                hyp = hyp[0]  # only return 1-best
                hyp_ids = [int(id) for id in hyp["yseq"].cpu()]
                text = self.tokenizer.detokenize(hyp_ids)
                results.append({"uttid": uttid, "text": text, "wav": wav,
                    "rtf": f"{rtf:.4f}"})
            return results

        elif self.asr_type == "llm":
            input_ids, attention_mask, _, _ = \
                LlmTokenizerWrapper.preprocess_texts(
                    origin_texts=[""]*feats.size(0), tokenizer=self.tokenizer,
                    max_len=128, decode=True)
            if args.get("use_gpu", False):
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            start_time = time.time()

            generated_ids = self.model.transcribe(
                feats, lengths, input_ids, attention_mask,
                args.get("beam_size", 1),
                args.get("decode_max_len", 0),
                args.get("decode_min_len", 0),
                args.get("repetition_penalty", 1.0),
                args.get("llm_length_penalty", 0.0),
                args.get("temperature", 1.0)
            )

            elapsed = time.time() - start_time
            rtf= elapsed / total_dur if total_dur > 0 else 0
            texts = self.tokenizer.batch_decode(generated_ids,
                                                skip_special_tokens=True)
            results = []
            for uttid, wav, text in zip(batch_uttid, batch_wav_path, texts):
                results.append({"uttid": uttid, "text": text, "wav": wav,
                                "rtf": f"{rtf:.4f}"})
            return results



def load_fireredasr_aed_model(model_path):
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    print("model args:", package["args"])
    model = FireRedAsrAed.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=True)
    return model


def load_firered_llm_model_and_tokenizer(model_path, encoder_path, llm_dir):
    package = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    package["args"].encoder_path = encoder_path
    package["args"].llm_dir = llm_dir
    print("model args:", package["args"])
    model = FireRedAsrLlm.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=False)
    tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(llm_dir)
    return model, tokenizer
