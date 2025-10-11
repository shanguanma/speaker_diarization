"""
手动分块处理长音频的示例脚本
用于需要更精细控制转录过程的场景
"""

import torch
import librosa
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_overlapping_transcripts(
    transcripts: List[str],
    timestamps: List[float],
    overlap_seconds: float = 4.0
) -> str:
    """
    合并有重叠的转录片段
    
    Args:
        transcripts: 转录文本列表
        timestamps: 每个片段的起始时间戳
        overlap_seconds: 重叠时长（秒）
    
    Returns:
        合并后的完整转录文本
    """
    if not transcripts:
        return ""
    
    if len(transcripts) == 1:
        return transcripts[0]
    
    # 简单的合并策略：去除重叠区域的重复内容
    # 更复杂的策略可以使用编辑距离或词级别的对齐
    merged = transcripts[0]
    
    for i in range(1, len(transcripts)):
        current_text = transcripts[i]
        
        # 简单策略：直接拼接（假设 pipeline 已经处理了重叠）
        # 或者可以尝试移除可能的重复词
        words_prev = merged.split()[-20:]  # 取前一个片段的最后20个词
        words_curr = current_text.split()[:20]  # 取当前片段的前20个词
        
        # 寻找最长公共子序列作为重叠点
        overlap_found = False
        for j in range(min(len(words_prev), len(words_curr)), 0, -1):
            if words_prev[-j:] == words_curr[:j]:
                # 找到重叠，移除重复部分
                merged += " " + " ".join(words_curr[j:])
                overlap_found = True
                break
        
        if not overlap_found:
            # 没有找到明显重叠，直接拼接
            merged += " " + current_text
    
    return merged.strip()


def transcribe_long_audio_chunked(
    model_path: str,
    audio_path: str,
    chunk_length_s: int = 30,
    overlap_s: int = 4,
    language: str = "en",
    device: str = "cuda",
    batch_size: int = 1,
) -> Tuple[str, List[dict]]:
    """
    使用手动分块方法转录长音频
    
    Args:
        model_path: 微调后的 Whisper 模型路径
        audio_path: 音频文件路径
        chunk_length_s: 每个块的长度（秒）
        overlap_s: 重叠长度（秒）
        language: 语言代码
        device: 设备（cuda/cpu）
        batch_size: 批处理大小（>1 时批量处理多个块）
    
    Returns:
        完整转录文本和带时间戳的片段列表
    """
    logger.info(f"Loading model from {model_path}")
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    processor = WhisperProcessor.from_pretrained(model_path)
    
    # 加载音频
    logger.info(f"Loading audio from {audio_path}")
    audio_array, sr = librosa.load(audio_path, sr=16000)
    total_duration = len(audio_array) / sr
    logger.info(f"Audio duration: {total_duration:.2f} seconds")
    
    # 计算分块参数
    chunk_samples = chunk_length_s * sr
    overlap_samples = overlap_s * sr
    stride = chunk_samples - overlap_samples
    
    # 准备批处理
    chunks = []
    chunk_info = []
    
    for start in range(0, len(audio_array), stride):
        end = min(start + chunk_samples, len(audio_array))
        chunk = audio_array[start:end]
        
        # 如果最后一个块太短，填充到最小长度
        if len(chunk) < sr * 1:  # 至少1秒
            if chunks:  # 如果不是第一个块，跳过过短的块
                break
            else:
                # 如果是唯一的块，保留它
                pass
        
        chunks.append(chunk)
        chunk_info.append({
            'start_time': start / sr,
            'end_time': end / sr,
            'start_sample': start,
            'end_sample': end
        })
        
        # 达到批处理大小或是最后一个块时，进行处理
        if len(chunks) == batch_size or end >= len(audio_array):
            # 批处理转录
            transcripts_batch = process_audio_batch(
                model, processor, chunks, language, device
            )
            
            # 更新 chunk_info
            for i, transcript in enumerate(transcripts_batch):
                idx = len(chunk_info) - len(transcripts_batch) + i
                chunk_info[idx]['text'] = transcript
            
            chunks = []  # 清空批次
        
        if end >= len(audio_array):
            break
    
    # 提取转录文本
    transcripts = [info['text'] for info in chunk_info]
    timestamps = [info['start_time'] for info in chunk_info]
    
    # 合并转录
    logger.info(f"Merging {len(transcripts)} chunks...")
    full_transcript = merge_overlapping_transcripts(transcripts, timestamps, overlap_s)
    
    logger.info(f"Transcription complete. Total length: {len(full_transcript)} characters")
    
    return full_transcript, chunk_info


def process_audio_batch(
    model,
    processor,
    audio_chunks: List[np.ndarray],
    language: str,
    device: str
) -> List[str]:
    """
    批量处理音频块
    
    Args:
        model: Whisper 模型
        processor: Whisper 处理器
        audio_chunks: 音频块列表
        language: 语言代码
        device: 设备
    
    Returns:
        转录文本列表
    """
    if not audio_chunks:
        return []
    
    # 处理所有音频块
    input_features_list = []
    for chunk in audio_chunks:
        input_features = processor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features
        input_features_list.append(input_features)
    
    # 批处理
    batch_input_features = torch.cat(input_features_list, dim=0).to(device)
    
    # 设置语言和任务
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language,
        task="transcribe"
    )
    
    # 生成转录
    with torch.no_grad():
        predicted_ids = model.generate(
            batch_input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_length=448,
            num_beams=1,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
    
    # 解码
    transcripts = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcripts


def transcribe_with_vad_chunking(
    model_path: str,
    audio_path: str,
    language: str = "en",
    device: str = "cuda"
) -> str:
    """
    使用 VAD（Voice Activity Detection）进行智能分块
    这种方法会在静音处切分，避免在说话中间切断
    
    需要安装: pip install webrtcvad
    
    Args:
        model_path: 模型路径
        audio_path: 音频路径
        language: 语言
        device: 设备
    
    Returns:
        完整转录
    """
    try:
        import webrtcvad
    except ImportError:
        logger.error("webrtcvad not installed. Install with: pip install webrtcvad")
        raise
    
    logger.info("Loading model...")
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    processor = WhisperProcessor.from_pretrained(model_path)
    
    # 加载音频
    logger.info(f"Loading audio from {audio_path}")
    audio_array, sr = librosa.load(audio_path, sr=16000)
    
    # 初始化 VAD
    vad = webrtcvad.Vad(2)  # 0-3, 越大越激进
    
    # 将音频分割成小帧（VAD 需要 10/20/30ms 的帧）
    frame_duration = 30  # ms
    frame_length = int(sr * frame_duration / 1000)
    
    # 检测语音活动
    frames = []
    speech_frames = []
    
    for i in range(0, len(audio_array), frame_length):
        frame = audio_array[i:i+frame_length]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        
        # 转换为 16-bit PCM
        frame_bytes = (frame * 32768).astype(np.int16).tobytes()
        
        # VAD 检测
        is_speech = vad.is_speech(frame_bytes, sr)
        
        frames.append((i, frame, is_speech))
        if is_speech:
            speech_frames.append(i // frame_length)
    
    # 合并连续的语音段
    segments = []
    if speech_frames:
        segment_start = speech_frames[0]
        prev_frame = speech_frames[0]
        
        for frame_idx in speech_frames[1:]:
            if frame_idx - prev_frame > 10:  # 超过10帧（300ms）的间隔，分段
                segment_end = prev_frame
                segments.append((
                    segment_start * frame_length,
                    segment_end * frame_length + frame_length
                ))
                segment_start = frame_idx
            prev_frame = frame_idx
        
        # 添加最后一段
        segments.append((
            segment_start * frame_length,
            prev_frame * frame_length + frame_length
        ))
    
    logger.info(f"Detected {len(segments)} speech segments")
    
    # 转录每个语音段
    transcripts = []
    for i, (start, end) in enumerate(segments):
        logger.info(f"Transcribing segment {i+1}/{len(segments)} ({start/sr:.2f}s - {end/sr:.2f}s)")
        
        segment_audio = audio_array[start:end]
        
        # 如果段太长（>30秒），进一步分块
        if len(segment_audio) > 30 * sr:
            # 递归使用标准分块
            # 这里简化处理，实际应该递归调用分块函数
            segment_chunks = []
            for chunk_start in range(0, len(segment_audio), 30 * sr):
                chunk_end = min(chunk_start + 30 * sr, len(segment_audio))
                segment_chunks.append(segment_audio[chunk_start:chunk_end])
        else:
            segment_chunks = [segment_audio]
        
        # 转录所有块
        for chunk in segment_chunks:
            input_features = processor(
                chunk,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(device)
            
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
                )
            
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            if text.strip():
                transcripts.append(text)
    
    # 合并所有转录
    full_transcript = " ".join(transcripts)
    
    return full_transcript


# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="手动分块长音频转录")
    parser.add_argument("--model-path", required=True, help="微调后的模型路径")
    parser.add_argument("--audio-path", required=True, help="音频文件路径")
    parser.add_argument("--chunk-length", type=int, default=30, help="块长度（秒）")
    parser.add_argument("--overlap", type=int, default=4, help="重叠长度（秒）")
    parser.add_argument("--language", default="en", help="语言代码")
    parser.add_argument("--device", default="cuda", help="设备")
    parser.add_argument("--batch-size", type=int, default=1, help="批处理大小")
    parser.add_argument("--output", help="输出文件路径")
    parser.add_argument("--use-vad", action="store_true", help="使用VAD智能分块")
    
    args = parser.parse_args()
    
    if args.use_vad:
        logger.info("Using VAD-based chunking")
        transcript = transcribe_with_vad_chunking(
            args.model_path,
            args.audio_path,
            args.language,
            args.device
        )
    else:
        logger.info("Using fixed-length chunking")
        transcript, chunks = transcribe_long_audio_chunked(
            args.model_path,
            args.audio_path,
            args.chunk_length,
            args.overlap,
            args.language,
            args.device,
            args.batch_size
        )
        
        # 打印每个块的信息
        logger.info("\nChunk details:")
        for i, chunk in enumerate(chunks):
            logger.info(
                f"Chunk {i+1}: {chunk['start_time']:.2f}s - {chunk['end_time']:.2f}s | "
                f"Text: {chunk['text'][:50]}..."
            )
    
    # 输出结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(transcript)
        logger.info(f"Transcript saved to {args.output}")
    else:
        print("\n" + "="*80)
        print("TRANSCRIPT:")
        print("="*80)
        print(transcript)
        print("="*80)


