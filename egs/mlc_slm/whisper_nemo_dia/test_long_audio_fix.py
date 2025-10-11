"""
测试脚本：验证长音频转录修复是否有效

使用方法:
python test_long_audio_fix.py \
    --model-path /path/to/finetuned/model \
    --audio-path /path/to/test/audio.wav \
    --output-dir ./test_results
"""

import argparse
import os
import time
import json
import torch
import librosa
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def test_original_method(model_path, audio_path, language, device):
    """
    测试原始的方法（单次 generate，会导致长音频丢失）
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    
    logger.info("="*80)
    logger.info("测试原始方法（预期会有问题）")
    logger.info("="*80)
    
    start_time = time.time()
    
    # 加载模型
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    processor = WhisperProcessor.from_pretrained(model_path)
    
    # 加载音频
    audio_array, sr = librosa.load(audio_path, sr=16000)
    audio_duration = len(audio_array) / sr
    logger.info(f"音频时长: {audio_duration:.2f} 秒")
    
    # 处理音频（原始方法）
    try:
        input_features = processor(
            audio_array,
            sampling_rate=sr,
            return_tensors="pt"
        ).input_features.to(device)
        
        logger.info(f"输入特征形状: {input_features.shape}")
        logger.info(f"预期最大特征长度: 3000 (对应 30秒)")
        
        # 检查是否被截断
        if input_features.shape[-1] >= 3000:
            logger.warning("⚠️  输入特征已被截断！大部分音频将被丢失！")
        
        # 生成转录
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language,
            task="transcribe"
        )
        
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids
        )
        
        transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
    except Exception as e:
        logger.error(f"❌ 原始方法出错: {e}")
        transcript = ""
    
    elapsed = time.time() - start_time
    
    # 分析结果
    word_count = len(transcript.split())
    char_count = len(transcript)
    
    logger.info(f"转录用时: {elapsed:.2f} 秒")
    logger.info(f"转录长度: {char_count} 字符, {word_count} 词")
    logger.info(f"转录内容预览: {transcript[:200]}...")
    
    # 估算覆盖率（假设每秒约15个词）
    expected_words = audio_duration * 2.5  # 英文平均每秒2-3个词
    coverage = (word_count / expected_words) * 100 if expected_words > 0 else 0
    logger.info(f"估算覆盖率: {coverage:.1f}% (预期词数: {expected_words:.0f}, 实际: {word_count})")
    
    if coverage < 20:
        logger.warning("⚠️  覆盖率严重偏低！存在大面积转录丢失！")
    
    del model
    torch.cuda.empty_cache()
    
    return {
        'method': 'original',
        'duration': audio_duration,
        'elapsed_time': elapsed,
        'transcript': transcript,
        'char_count': char_count,
        'word_count': word_count,
        'coverage': coverage,
        'input_shape': str(input_features.shape)
    }


def test_pipeline_method(model_path, audio_path, language, device, batch_size=8):
    """
    测试修复后的方法（使用 pipeline + chunking）
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
    
    logger.info("="*80)
    logger.info("测试修复后的方法（pipeline + chunking）")
    logger.info("="*80)
    
    start_time = time.time()
    
    # 加载模型
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    processor = WhisperProcessor.from_pretrained(model_path)
    
    # 加载音频
    audio_array, sr = librosa.load(audio_path, sr=16000)
    audio_duration = len(audio_array) / sr
    logger.info(f"音频时长: {audio_duration:.2f} 秒")
    
    # 创建 pipeline
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        stride_length_s=(4, 2),
        batch_size=batch_size,
        device=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    # 设置语言
    asr_pipeline.model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language,
        task="transcribe"
    )
    
    # 转录
    try:
        result = asr_pipeline(
            audio_array,
            return_timestamps=True,
            generate_kwargs={
                "max_length": 448,
                "num_beams": 1,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
            }
        )
        
        transcript = result["text"]
        chunks_info = result.get("chunks", [])
        
        logger.info(f"处理的块数: {len(chunks_info)}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline 方法出错: {e}")
        transcript = ""
        chunks_info = []
    
    elapsed = time.time() - start_time
    
    # 分析结果
    word_count = len(transcript.split())
    char_count = len(transcript)
    
    logger.info(f"转录用时: {elapsed:.2f} 秒")
    logger.info(f"转录长度: {char_count} 字符, {word_count} 词")
    logger.info(f"转录内容预览: {transcript[:200]}...")
    
    # 估算覆盖率
    expected_words = audio_duration * 2.5
    coverage = (word_count / expected_words) * 100 if expected_words > 0 else 0
    logger.info(f"估算覆盖率: {coverage:.1f}% (预期词数: {expected_words:.0f}, 实际: {word_count})")
    
    if coverage >= 80:
        logger.info("✅ 覆盖率良好！转录完整。")
    elif coverage >= 50:
        logger.warning("⚠️  覆盖率中等，可能需要调整参数。")
    else:
        logger.warning("⚠️  覆盖率偏低，请检查配置。")
    
    del model
    del asr_pipeline
    torch.cuda.empty_cache()
    
    return {
        'method': 'pipeline',
        'duration': audio_duration,
        'elapsed_time': elapsed,
        'transcript': transcript,
        'char_count': char_count,
        'word_count': word_count,
        'coverage': coverage,
        'num_chunks': len(chunks_info)
    }


def compare_results(result1, result2):
    """
    比较两种方法的结果
    """
    logger.info("="*80)
    logger.info("结果对比")
    logger.info("="*80)
    
    metrics = [
        ('处理时间', 'elapsed_time', 'seconds'),
        ('字符数', 'char_count', 'chars'),
        ('词数', 'word_count', 'words'),
        ('覆盖率', 'coverage', '%'),
    ]
    
    for metric_name, key, unit in metrics:
        val1 = result1.get(key, 0)
        val2 = result2.get(key, 0)
        
        if key == 'elapsed_time':
            logger.info(f"{metric_name}:")
            logger.info(f"  原始方法: {val1:.2f} {unit}")
            logger.info(f"  修复方法: {val2:.2f} {unit}")
        elif key == 'coverage':
            logger.info(f"{metric_name}:")
            logger.info(f"  原始方法: {val1:.1f}{unit}")
            logger.info(f"  修复方法: {val2:.1f}{unit}")
            improvement = val2 - val1
            logger.info(f"  改善: {'+' if improvement > 0 else ''}{improvement:.1f}{unit}")
        else:
            logger.info(f"{metric_name}:")
            logger.info(f"  原始方法: {val1} {unit}")
            logger.info(f"  修复方法: {val2} {unit}")
    
    # 判断修复效果
    logger.info("\n" + "="*80)
    logger.info("结论")
    logger.info("="*80)
    
    if result2['coverage'] > result1['coverage'] * 2:
        logger.info("✅ 修复非常成功！覆盖率显著提升。")
    elif result2['coverage'] > result1['coverage'] * 1.5:
        logger.info("✅ 修复成功！覆盖率明显改善。")
    elif result2['coverage'] > result1['coverage']:
        logger.info("✅ 修复有效，但可能需要进一步优化参数。")
    else:
        logger.warning("⚠️  修复效果不明显，请检查配置或音频质量。")


def save_results(output_dir, results):
    """
    保存测试结果
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for result in results:
        method = result['method']
        
        # 保存转录文本
        txt_path = os.path.join(output_dir, f"{method}_transcript.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result['transcript'])
        
        # 保存元数据
        metadata = {k: v for k, v in result.items() if k != 'transcript'}
        json_path = os.path.join(output_dir, f"{method}_metadata.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"已保存 {method} 方法的结果到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="测试长音频转录修复效果")
    parser.add_argument("--model-path", required=True, help="微调后的模型路径")
    parser.add_argument("--audio-path", required=True, help="测试音频路径（建议 >10分钟）")
    parser.add_argument("--language", default="en", help="语言代码")
    parser.add_argument("--device", default="cuda", help="设备")
    parser.add_argument("--batch-size", type=int, default=8, help="批处理大小")
    parser.add_argument("--output-dir", default="./test_results", help="输出目录")
    parser.add_argument("--skip-original", action="store_true", help="跳过原始方法测试")
    
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.exists(args.audio_path):
        logger.error(f"音频文件不存在: {args.audio_path}")
        return
    
    if not os.path.exists(args.model_path):
        logger.error(f"模型路径不存在: {args.model_path}")
        return
    
    # 检查音频时长
    audio_array, sr = librosa.load(args.audio_path, sr=16000)
    duration = len(audio_array) / sr
    
    if duration < 600:  # 10分钟
        logger.warning(f"⚠️  音频时长仅 {duration:.1f} 秒 (< 10分钟)，可能无法充分体现长音频问题。")
        logger.warning("建议使用至少 20 分钟的音频进行测试。")
    
    results = []
    
    # 测试原始方法
    if not args.skip_original:
        try:
            result1 = test_original_method(
                args.model_path,
                args.audio_path,
                args.language,
                args.device
            )
            results.append(result1)
        except Exception as e:
            logger.error(f"原始方法测试失败: {e}")
            result1 = None
    else:
        logger.info("跳过原始方法测试")
        result1 = None
    
    # 测试修复方法
    try:
        result2 = test_pipeline_method(
            args.model_path,
            args.audio_path,
            args.language,
            args.device,
            args.batch_size
        )
        results.append(result2)
    except Exception as e:
        logger.error(f"修复方法测试失败: {e}")
        result2 = None
    
    # 比较结果
    if result1 and result2:
        compare_results(result1, result2)
    
    # 保存结果
    if results:
        save_results(args.output_dir, results)
        logger.info(f"\n所有结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()


