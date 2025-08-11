import numpy as np
import torch
import torchaudio

# 参数设置
audio_length = 8  # 秒
sample_rate = 16000
total_samples = audio_length * sample_rate
window_size = 0.025  # 25ms
window_shift = 0.01  # 10ms
downsample_factor = 4

# 创建模拟音频数据 (8秒，16000Hz)
waveform = torch.randn(1, total_samples)  # 随机噪声模拟音频

# 创建模拟标签数据：前3秒有说话人(1)，后5秒无(0)
labels = np.zeros(total_samples, dtype=np.int32)
speech_start = 0
speech_end = 3 * sample_rate  # 3秒位置
labels[speech_start:speech_end] = 1

# 计算FBANK特征
fbank = torchaudio.compliance.kaldi.fbank(
    waveform,
    num_mel_bins=80,
    frame_length=window_size * 1000,  # ms
    frame_shift=window_shift * 1000,  # ms
    sample_frequency=sample_rate
)
original_frames = fbank.shape[0]
print(f"原始特征帧数: {original_frames}")

# 特征下采样 (取每4帧的第1帧)
downsampled_fbank = fbank[::downsample_factor]
downsampled_frames = downsampled_fbank.shape[0]
print(f"下采样后特征帧数: {downsampled_frames}")

# 标签处理函数
def process_labels(labels, sample_rate, window_size, window_shift, downsample_factor):
    # 计算帧参数
    frame_length = int(window_size * sample_rate)  # 400个采样点
    frame_shift = int(window_shift * sample_rate)  # 160个采样点
    
    # 计算总帧数 (与FBANK特征提取逻辑一致)
    num_frames = (total_samples - frame_length) // frame_shift + 1
    
    # 初始化帧级标签
    frame_labels = np.zeros(num_frames, dtype=np.int32)
    
    # 为每帧分配标签 (只要帧内有1就标记为1)
    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_length
        frame_labels[i] = 1 if np.any(labels[start:end]) else 0
    
    # 标签下采样 (取每4帧的第1帧)
    downsampled_labels = frame_labels[::downsample_factor]
    
    return frame_labels, downsampled_labels

# 处理标签
frame_labels, downsampled_labels = process_labels(
    labels, sample_rate, window_size, window_shift, downsample_factor
)

# 验证结果
def validate_results(labels, downsampled_labels, sample_rate, window_size, window_shift, downsample_factor):
    frame_shift = window_shift * sample_rate
    downsampled_frame_shift = frame_shift * downsample_factor
    frame_length = window_size * sample_rate
    
    # 计算下采样后每帧的时间范围
    frame_times = []
    for i in range(len(downsampled_labels)):
        start_time = i * downsampled_frame_shift / sample_rate
        end_time = (i * downsampled_frame_shift + frame_length) / sample_rate
        frame_times.append((start_time, end_time))
    
    # 检查关键时间点
    test_points = [
        (2.5, "应有说话人", 1),
        (3.5, "应无说话人", 0),
        (0.1, "应有说话人", 1),
        (7.9, "应无说话人", 0)
    ]
    
    print("\n验证关键时间点:")
    for time, desc, expected in test_points:
        # 找到包含该时间点的帧
        frame_idx = None
        for i, (start, end) in enumerate(frame_times):
            if start <= time < end:
                frame_idx = i
                break
        
        if frame_idx is None:
            print(f"  {time}s: 未找到对应帧 | {desc}")
            continue
        
        actual = downsampled_labels[frame_idx]
        status = "通过" if actual == expected else "失败"
        print(f"  {time:.1f}s -> 帧{frame_idx}: 预期={expected}, 实际={actual} | {status} {desc}")
    
    # 验证标签长度匹配
    print("\n长度匹配验证:")
    print(f"特征帧数: {downsampled_frames}, 标签帧数: {len(downsampled_labels)}")
    print("长度匹配:", "通过" if downsampled_frames == len(downsampled_labels) else "失败")
    
    # 验证语音边界
    last_speech_frame = int((speech_end - frame_length) / frame_shift)
    first_non_speech_frame = last_speech_frame + 1
    
    # 计算下采样后的对应帧
    last_speech_down = last_speech_frame // downsample_factor
    first_non_speech_down = first_non_speech_frame // downsample_factor
    
    print("\n语音边界验证:")
    print(f"原始最后一帧语音: {last_speech_frame} (时间: {(last_speech_frame*frame_shift+frame_length)/sample_rate:.3f}s)")
    print(f"下采样后最后一帧语音: {last_speech_down} (值: {downsampled_labels[last_speech_down]})")
    print(f"下采样后第一帧静音: {first_non_speech_down} (值: {downsampled_labels[first_non_speech_down]})")
    
    # 检查边界帧值
    boundary_ok = (
        downsampled_labels[last_speech_down] == 1 and
        downsampled_labels[first_non_speech_down] == 0
    )
    print("边界验证:", "通过" if boundary_ok else "失败")

# 执行验证
validate_results(labels, downsampled_labels, sample_rate, window_size, window_shift, downsample_factor)

# 打印部分标签
print("\n下采样后前5帧标签:", downsampled_labels[:5])
print("下采样后最后5帧标签:", downsampled_labels[-5:])
