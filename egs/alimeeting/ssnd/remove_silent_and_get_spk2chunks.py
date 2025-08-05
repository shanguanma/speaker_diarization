#!/usr/bin/env python3 -u
"""
非并行的VoxCeleb2数据集处理脚本
移除torchrun依赖，顺序处理所有文件并保存为JSON格式
支持多种压缩格式：gzip, bz2, lzma, zstandard
"""
import gzip
import os
import sys
import json
import argparse
import logging
from collections import defaultdict
import numpy as np
import soundfile as sf
import librosa
from funasr import AutoModel
from tqdm import tqdm

# 尝试导入更好的压缩库
try:
    import bz2
    HAS_BZ2 = True
except ImportError:
    HAS_BZ2 = False

try:
    import lzma
    HAS_LZMA = True
except ImportError:
    HAS_LZMA = False

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

# 设置日志
def setup_logging():
    """设置日志配置"""
    # 强制重新配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
        force=True,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

# 初始化日志
logger = setup_logging()

def load_dataset_info(voxceleb2_dataset_dir):
    """加载数据集信息"""
    wavscp = f"{voxceleb2_dataset_dir}/wav.scp"
    spk2utt = f"{voxceleb2_dataset_dir}/spk2utt"
    
    if not os.path.exists(wavscp):
        raise FileNotFoundError(f"wav.scp文件不存在: {wavscp}")
    if not os.path.exists(spk2utt):
        raise FileNotFoundError(f"spk2utt文件不存在: {spk2utt}")
    
    spk2wav = defaultdict(list)
    wav2scp = {}
    
    # 读取wav.scp文件
    logger.info("读取wav.scp文件...")
    with open(wavscp, 'r') as fscp:
        for line in fscp:
            line = line.strip().split()
            key = line[0]
            wav2scp[key] = line[1]

    # 读取spk2utt文件
    logger.info("读取spk2utt文件...")
    with open(spk2utt, 'r') as fspk:
        for line in fspk:
            line = line.strip().split()
            key = line[0]
            paths = [wav2scp[i] for i in line[1:]]
            spk2wav[key].extend(paths)
    
    return spk2wav

# 全局VAD模型，避免重复初始化
_vad_model = None

def get_vad_model():
    """获取VAD模型（单例模式）"""
    global _vad_model
    if _vad_model is None:
        try:
            logger.info("初始化VAD模型...")
            _vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4", disable_update=True)
            logger.info("VAD模型初始化成功")
        except Exception as e:
            logger.error(f"VAD模型初始化失败: {e}")
            return None
    return _vad_model

def vad_detect(wav, sr):
    """VAD检测函数"""
    try:
        # 检查音频长度
        if len(wav) < sr * 0.1:  # 小于0.1秒的音频跳过
            return []
        
        # 确保音频是1D数组
        if wav.ndim > 1:
            wav = wav.flatten()
        
        # 检查音频数据是否包含NaN或无穷大值
        if np.any(np.isnan(wav)) or np.any(np.isinf(wav)):
            return []
        
        # 确保音频数据在合理范围内
        wav = np.clip(wav, -1.0, 1.0)
        
        # 获取VAD模型
        vad_model = get_vad_model()
        if vad_model is None:
            return []
        
        # 更安全的数据类型转换
        if wav.dtype != np.int16:
            wav_normalized = np.clip(wav, -1.0, 1.0)
            wav_int16 = (wav_normalized * 32767).astype(np.int16)
        else:
            wav_int16 = wav

        wav_input=wav_int16
        result = vad_model.generate(wav_input, fs=sr)
        time_stamp = result[0]['value']
        # https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/fsmn_vad_streaming/demo.py#L16
        return time_stamp  # in ms
        
    except Exception as e:
        logger.error(f"VAD检测失败: {e}")
        return []

def process_audio_file(wav_path, spk_id):
    """处理单个音频文件"""
    try:
        # 检查文件是否存在
        if not os.path.exists(wav_path):
            return {
                'spk_id': spk_id,
                'wav_path': wav_path,
                'time_stamp_list': [],
                'success': False,
                'error': f"文件不存在: {wav_path}"
            }
        
        # 检查文件大小
        file_size = os.path.getsize(wav_path)
        if file_size == 0:
            return {
                'spk_id': spk_id,
                'wav_path': wav_path,
                'time_stamp_list': [],
                'success': False,
                'error': f"文件大小为0: {wav_path}"
            }
        
        # 读取音频文件
        try:
            wav, sr = sf.read(wav_path)
        except Exception as e:
            return {
                'spk_id': spk_id,
                'wav_path': wav_path,
                'time_stamp_list': [],
                'success': False,
                'error': f"音频文件读取失败: {e}"
            }
        
        # 检查音频数据
        if wav is None or len(wav) == 0:
            return {
                'spk_id': spk_id,
                'wav_path': wav_path,
                'time_stamp_list': [],
                'success': False,
                'error': f"音频数据为空: {wav_path}"
            }
        
        # 重采样到16kHz
        if sr != 16000:
            try:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            except Exception as e:
                return {
                    'spk_id': spk_id,
                    'wav_path': wav_path,
                    'time_stamp_list': [],
                    'success': False,
                    'error': f"音频重采样失败: {e}"
                }
        
        # 执行VAD检测
        time_stamp_list = vad_detect(wav, sr=16000)
        # in ms ->(/1000) in second ->(*16000) in sample points
        #time_stamp_list = [wav[int(s*16):int(e*16)].tolist() for s, e in time_stamp_list]        
        return {
            'spk_id': spk_id,
            'wav_path': wav_path,
            'time_stamp_list': time_stamp_list,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"处理音频文件失败 {wav_path}: {e}")
        return {
            'spk_id': spk_id,
            'wav_path': wav_path,
            'time_stamp_list': [],
            'success': False,
            'error': str(e)
        }

def save_results_to_json(results, output_file, format_type="jsonl_gzip", compression_level=6):
    """
    保存结果为JSON格式，支持多种压缩方式
    
    Args:
        results: 处理结果
        output_file: 输出文件路径
        format_type: 格式类型
            - "jsonl_gzip": JSONL格式 + gzip压缩 (默认)
            - "jsonl_bz2": JSONL格式 + bzip2压缩
            - "jsonl_lzma": JSONL格式 + lzma压缩
            - "jsonl_zstd": JSONL格式 + zstandard压缩
            - "json_gzip": 单个JSON对象 + gzip压缩
            - "json_bz2": 单个JSON对象 + bzip2压缩
            - "json_lzma": 单个JSON对象 + lzma压缩
            - "json_zstd": 单个JSON对象 + zstandard压缩
        compression_level: 压缩级别 (1-9, 越高压缩率越好但速度越慢)
    """
    logger.info(f"保存结果，格式: {format_type}, 压缩级别: {compression_level}")
    
    # 准备数据
    if format_type.startswith("jsonl_"):
        # JSONL格式：每行一个JSON对象
        save_jsonl_format(results, output_file, format_type, compression_level)
    elif format_type.startswith("json_"):
        # 单个JSON对象格式
        save_single_json_format(results, output_file, format_type, compression_level)
    else:
        raise ValueError(f"不支持的格式类型: {format_type}")

def save_jsonl_format(results, output_file, format_type, compression_level):
    """保存为JSONL格式"""
    compression_type = format_type.split("_", 1)[1]
    
    if compression_type == "gzip":
        with gzip.open(output_file, "wt", encoding='utf-8', compresslevel=compression_level) as f:
            for spk_id, spk_results in results.items():
                spk2chunks = defaultdict(list)
                spk2wav_paths = defaultdict(list)
                for item in spk_results:
                    spk2chunks[spk_id].append(item['time_stamp_list'])
                    spk2wav_paths[spk_id].append(item['wav_path'])
                
                res = {
                    'spk_id': spk_id,
                    'wav_paths': spk2wav_paths[spk_id],
                    'results': spk2chunks[spk_id],
                }
                json.dump(res, f, ensure_ascii=False, separators=(',', ':'))
                f.write("\n")
    
    elif compression_type == "bz2" and HAS_BZ2:
        with bz2.open(output_file, "wt", encoding='utf-8', compresslevel=compression_level) as f:
            for spk_id, spk_results in results.items():
                spk2chunks = defaultdict(list)
                spk2wav_paths = defaultdict(list)
                for item in spk_results:
                    spk2chunks[spk_id].append(item['time_stamp_list'])
                    spk2wav_paths[spk_id].append(item['wav_path'])
                
                res = {
                    'spk_id': spk_id,
                    'wav_paths': spk2wav_paths[spk_id],
                    'results': spk2chunks[spk_id],
                }
                json.dump(res, f, ensure_ascii=False, separators=(',', ':'))
                f.write("\n")
    
    elif compression_type == "lzma" and HAS_LZMA:
        with lzma.open(output_file, "wt", encoding='utf-8', preset=compression_level) as f:
            for spk_id, spk_results in results.items():
                spk2chunks = defaultdict(list)
                spk2wav_paths = defaultdict(list)
                for item in spk_results:
                    spk2chunks[spk_id].append(item['time_stamp_list'])
                    spk2wav_paths[spk_id].append(item['wav_path'])
                
                res = {
                    'spk_id': spk_id,
                    'wav_paths': spk2wav_paths[spk_id],
                    'results': spk2chunks[spk_id],
                }
                json.dump(res, f, ensure_ascii=False, separators=(',', ':'))
                f.write("\n")
    
    elif compression_type == "zstd" and HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=compression_level)
        with open(output_file, "wb") as f:
            for spk_id, spk_results in results.items():
                spk2chunks = defaultdict(list)
                spk2wav_paths = defaultdict(list)
                for item in spk_results:
                    spk2chunks[spk_id].append(item['time_stamp_list'])
                    spk2wav_paths[spk_id].append(item['wav_path'])
                
                res = {
                    'spk_id': spk_id,
                    'wav_paths': spk2wav_paths[spk_id],
                    'results': spk2chunks[spk_id],
                }
                json_str = json.dumps(res, ensure_ascii=False, separators=(',', ':')) + "\n"
                compressed_data = cctx.compress(json_str.encode('utf-8'))
                f.write(compressed_data)
    
    else:
        raise ValueError(f"不支持的压缩类型: {compression_type}")

def save_single_json_format(results, output_file, format_type, compression_level):
    """保存为单个JSON对象格式"""
    compression_type = format_type.split("_", 1)[1]
    
    # 构建完整的JSON对象
    complete_data = {}
    for spk_id, spk_results in results.items():
        spk2chunks = defaultdict(list)
        spk2wav_paths = defaultdict(list)
        for item in spk_results:
            spk2chunks[spk_id].append(item['time_stamp_list'])
            spk2wav_paths[spk_id].append(item['wav_path'])
        
        complete_data[spk_id] = {
            'wav_paths': spk2wav_paths[spk_id],
            'results': spk2chunks[spk_id],
        }
    
    if compression_type == "gzip":
        with gzip.open(output_file, "wt", encoding='utf-8', compresslevel=compression_level) as f:
            json.dump(complete_data, f, ensure_ascii=False, separators=(',', ':'))
    
    elif compression_type == "bz2" and HAS_BZ2:
        with bz2.open(output_file, "wt", encoding='utf-8', compresslevel=compression_level) as f:
            json.dump(complete_data, f, ensure_ascii=False, separators=(',', ':'))
    
    elif compression_type == "lzma" and HAS_LZMA:
        with lzma.open(output_file, "wt", encoding='utf-8', preset=compression_level) as f:
            json.dump(complete_data, f, ensure_ascii=False, separators=(',', ':'))
    
    elif compression_type == "zstd" and HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=compression_level)
        json_str = json.dumps(complete_data, ensure_ascii=False, separators=(',', ':'))
        compressed_data = cctx.compress(json_str.encode('utf-8'))
        with open(output_file, "wb") as f:
            f.write(compressed_data)
    
    else:
        raise ValueError(f"不支持的压缩类型: {compression_type}")

def process_all_files(spk2wav, output_file, format_type="jsonl_gzip", compression_level=6, max_batches=None):
    """处理所有文件"""
    results = defaultdict(list)
    failed_files = []
    
    # 准备所有任务
    all_tasks = []
    for spk_id, wav_paths in spk2wav.items():
        for wav_path in wav_paths:
            all_tasks.append((wav_path, spk_id))
    
    logger.info(f"开始处理 {len(all_tasks)} 个文件")
    if max_batches:
        logger.info(f"限制处理前 {max_batches} 个批次")
    
    # 分批处理，避免内存溢出
    batch_size = 50  # 每批处理50个文件
    total_batches = (len(all_tasks) + batch_size - 1) // batch_size
    
    for i in range(0, len(all_tasks), batch_size):
        batch_num = i // batch_size + 1
        
        # 如果设置了最大批次限制，检查是否超过
        if max_batches and batch_num > max_batches:
            logger.info(f"已达到最大批次限制 {max_batches}，停止处理")
            break
            
        batch_tasks = all_tasks[i:i+batch_size]
        logger.info(f"处理批次 {batch_num}/{total_batches} (包含 {len(batch_tasks)} 个文件)")
        
        for wav_path, spk_id in tqdm(batch_tasks, desc=f"批次 {batch_num}"):
            try:
                result = process_audio_file(wav_path, spk_id)
                
                if result['success']:
                    results[spk_id].append({
                        'wav_path': result['wav_path'],
                        'time_stamp_list': result['time_stamp_list']
                    })
                else:
                    failed_files.append(result)
            except Exception as e:
                logger.error(f"处理文件失败 {wav_path}: {e}")
                failed_files.append({
                    'spk_id': spk_id,
                    'wav_path': wav_path,
                    'time_stamp_list': [],
                    'success': False,
                    'error': str(e)
                })
        
        # 每处理完一个批次就保存一次临时结果
        if max_batches:  # 只在测试模式下保存临时结果
            temp_output_file = f"{output_file}.batch_{batch_num}"
            save_results_to_json(results, temp_output_file, format_type, compression_level)
            logger.info(f"批次 {batch_num} 临时结果保存到: {temp_output_file}")
        
        # 清理内存
        import gc
        gc.collect()
    
    # 保存最终结果
    save_results_to_json(results, output_file, format_type, compression_level)
    
    # 保存失败文件信息
    if failed_files:
        failed_output_file = f"{output_file}.failed"
        with gzip.open(failed_output_file, "wt", encoding='utf-8') as f:
            json.dump(failed_files, f, indent=2, ensure_ascii=False)
        logger.info(f"失败文件信息保存到: {failed_output_file}")
    
    logger.info(f"处理完成，结果保存到: {output_file}")
    logger.info(f"成功: {sum(len(spk_results) for spk_results in results.values())}, 失败: {len(failed_files)}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="非并行的VoxCeleb2数据集处理")
    parser.add_argument("--voxceleb2-dataset-dir", type=str, 
                       default="/maduo/datasets/voxceleb2/vox2_dev/", 
                       help="VoxCeleb2数据集Kaldi格式路径")
    parser.add_argument("--out-text", type=str, 
                       default="/maduo/datasets/voxceleb2/vox2_dev/train.json.gz", 
                       help="输出JSON文件路径")
    parser.add_argument("--format", type=str, default="jsonl_gzip",
                       choices=["jsonl_gzip", "jsonl_bz2", "jsonl_lzma", "jsonl_zstd", 
                               "json_gzip", "json_bz2", "json_lzma", "json_zstd"],
                       help="输出格式和压缩类型")
    parser.add_argument("--compression-level", type=int, default=6,
                       help="压缩级别 (1-9, 越高压缩率越好但速度越慢)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="详细输出模式")
    parser.add_argument("--max-batches", type=int, default=None,
                       help="最大处理批次数量（用于测试，如设置为3则只处理前3个批次）")
    
    args = parser.parse_args()
    
    # 测试日志是否正常工作
    print("="*50)
    print("日志测试开始")
    print("="*50)
    
    logger.info("开始非并行处理VoxCeleb2数据集...")
    logger.info(f"数据集路径: {args.voxceleb2_dataset_dir}")
    logger.info(f"输出文件: {args.out_text}")
    logger.info(f"输出格式: {args.format}")
    logger.info(f"压缩级别: {args.compression_level}")
    logger.info(f"可用压缩库: bz2={HAS_BZ2}, lzma={HAS_LZMA}, zstd={HAS_ZSTD}")
    
    # 检查数据集路径是否存在
    if not os.path.exists(args.voxceleb2_dataset_dir):
        logger.error(f"数据集路径不存在: {args.voxceleb2_dataset_dir}")
        return
    
    # 加载数据集信息
    try:
        spk2wav = load_dataset_info(args.voxceleb2_dataset_dir)
        logger.info(f"加载了 {len(spk2wav)} 个说话人的数据")
        
        # 显示前几个说话人的信息
        for i, (spk_id, wav_paths) in enumerate(spk2wav.items()):
            if i < 3:  # 只显示前3个
                logger.info(f"说话人 {spk_id}: {len(wav_paths)} 个文件")
            else:
                break
        
        # 处理所有文件
        results = process_all_files(spk2wav, args.out_text, args.format, args.compression_level, args.max_batches)
        
        logger.info("处理完成!")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 
