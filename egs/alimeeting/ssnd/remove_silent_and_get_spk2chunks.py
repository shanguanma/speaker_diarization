from funasr import AutoModel # pip install funasr # only for simu data
import numpy as np
import librosa
import soundfile as sf
import json
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import logging
from tqdm import tqdm
import threading

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局VAD模型，使用线程本地存储
_vad_model_local = threading.local()

def get_vad_model():
    """获取线程本地的VAD模型"""
    if not hasattr(_vad_model_local, 'model'):
        try:
            _vad_model_local.model = AutoModel(model="fsmn-vad", model_revision="v2.0.4", disable_update=True)
        except Exception as e:
            logger.error(f"VAD模型初始化失败: {e}")
            return None
    return _vad_model_local.model

def vad_func(wav, sr):
    """VAD函数，检测语音活动"""
    try:
        # 检查音频长度，如果太短则跳过
        if len(wav) < sr * 0.1:  # 小于0.1秒的音频跳过
            logger.warning(f"音频太短，跳过处理: {len(wav)} samples")
            return []
        
        # 确保音频是1D数组
        if wav.ndim > 1:
            wav = wav.flatten()
        
        # 检查音频数据是否包含NaN或无穷大值
        if np.any(np.isnan(wav)) or np.any(np.isinf(wav)):
            logger.warning(f"音频数据包含NaN或无穷大值，跳过处理")
            return []
        
        # 确保音频数据在合理范围内
        wav = np.clip(wav, -1.0, 1.0)
        
        # 获取VAD模型
        fsmn_vad_model = get_vad_model()
        if fsmn_vad_model is None:
            return []
        
        # 更安全的数据类型转换
        if wav.dtype != np.int16:
            # 确保音频数据在[-1, 1]范围内
            wav_normalized = np.clip(wav, -1.0, 1.0)
            # 转换为int16，避免溢出
            wav_int16 = (wav_normalized * 32767).astype(np.int16)
        else:
            wav_int16 = wav
        
        # 检查转换后的数据是否有效
        if len(wav_int16) == 0:
            logger.warning(f"转换后的音频数据为空")
            return []
        
        # 添加额外的安全检查
        if len(wav_int16) > sr * 3600:  # 超过1小时的音频跳过
            logger.warning(f"音频太长，跳过处理: {len(wav_int16)} samples")
            return []
        
        result = fsmn_vad_model.generate(wav_int16, fs=sr)
        time_stamp = result[0]['value']
        return time_stamp  # in ms
    except Exception as e:
        logger.error(f"VAD处理失败: {e}")
        return []

def process_single_wav(args):
    """处理单个音频文件的函数，用于并行执行"""
    wav_path, spk_id = args
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
        time_stamp_list = vad_func(wav, sr=16000)
        
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

def write_vad_list():
    pass

def spktochunks(args):
    voxceleb2_dataset_dir = args.voxceleb2_dataset_dir
    wavscp = f"{args.voxceleb2_dataset_dir}/wav.scp"
    spk2utt = f"{args.voxceleb2_dataset_dir}/spk2utt"
    
    # 检查文件是否存在
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
            if key in spk2wav:
                spk2wav[key].extend(paths)
            else:
                spk2wav[key] = paths

    # 准备并行处理的任务
    tasks = []
    for spk_id, wav_paths in spk2wav.items():
        for wav_path in wav_paths:
            tasks.append((wav_path, spk_id))
    
    logger.info(f"总共需要处理 {len(tasks)} 个音频文件，涉及 {len(spk2wav)} 个说话人")
    
    # 使用线程池并行处理
    if hasattr(args, 'max_workers') and args.max_workers is not None:
        max_workers = args.max_workers
    else:
        # 默认使用较少的线程数，避免内存和资源竞争问题
        max_workers = min(16, os.cpu_count() + 2)
    
    logger.info(f"使用 {max_workers} 个线程进行并行处理")
    logger.info(f"系统CPU核心数: {os.cpu_count()}")
    
    # 如果线程数设置过高，给出警告
    if max_workers > 32:
        logger.warning(f"线程数设置较高 ({max_workers})，可能会导致内存不足或VAD模型冲突")
        logger.warning("建议将线程数设置为16-24之间")
    
    results = defaultdict(list)
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_single_wav, task): task for task in tasks}
        
        # 使用tqdm显示进度
        with tqdm(total=len(tasks), desc="处理音频文件") as pbar:
            for future in as_completed(future_to_task):
                result = future.result()
                pbar.update(1)
                
                if result['success']:
                    spk_id = result['spk_id']
                    results[spk_id].append({
                        'wav_path': result['wav_path'],
                        'time_stamp_list': result['time_stamp_list']
                    })
                else:
                    failed_files.append(result)
    
    # 输出处理结果统计
    logger.info(f"成功处理: {len(tasks) - len(failed_files)} 个文件")
    logger.info(f"处理失败: {len(failed_files)} 个文件")
    
    if failed_files:
        logger.warning("以下文件处理失败:")
        for failed in failed_files[:10]:  # 只显示前10个失败的文件
            logger.warning(f"  {failed['wav_path']}: {failed.get('error', 'Unknown error')}")
        if len(failed_files) > 10:
            logger.warning(f"  ... 还有 {len(failed_files) - 10} 个失败文件")
    
    # 保存结果到JSON文件
    logger.info(f"保存结果到 {args.out_text}")
    with open(args.out_text, "w") as outs:
        for spk_id, spk_results in results.items():
            spk2chunks = defaultdict(list)
            for item in spk_results:
                spk2chunks[spk_id].append(item['time_stamp_list'])
            
            res = {
                'spk_id': spk_id,
                'results': spk2chunks,
            }
            json.dump(res, outs)
            outs.write("\n")
    
    logger.info("处理完成!")

def get_args():
    parser = argparse.ArgumentParser(description="并行处理VoxCeleb2数据集，移除静音片段")
    parser.add_argument("--voxceleb2-dataset-dir", type=str, 
                       default="/maduo/datasets/voxceleb2/vox2_dev/", 
                       help="VoxCeleb2数据集Kaldi格式路径")
    parser.add_argument("--out-text", type=str, 
                       default="/maduo/datasets/voxceleb2/vox2_dev/train.json", 
                       help="输出JSON文件路径，包含移除静音后的说话人片段信息")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="最大工作线程数，默认为CPU核心数+4")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    spktochunks(args)
    

