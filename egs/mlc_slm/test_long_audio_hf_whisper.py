import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa

def transcribe_long_audio(model_name, audio_path, device="cuda", language="en"):
    """
    转录超长音频文件
    
    参数:
        model_name (str): Whisper 模型名称（如 "openai/whisper-large-v3"）
        audio_path (str): 音频文件路径
        device (str): 设备类型 ("cuda" 或 "cpu")
        language (str): 音频语言代码（如 "en" 表示英语）
    
    返回:
        str: 转录文本
    """
    
    # 1. 加载模型和处理器
    print("Loading model and processor...")
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    processor = WhisperProcessor.from_pretrained(model_name)
   

    # 关键修改：在生成之前，将模型配置中的 forced_decoder_ids 设置为 None
    model.generation_config.forced_decoder_ids = None  # 清除生成配置中的项
    model.config.forced_decoder_ids = None  # 清除模型配置中的项
    # 2. 加载和预处理音频
    print("Loading audio file...")
    audio_array, sampling_rate = librosa.load(audio_path, sr=16000, mono=True)
    
    # 3. 提取音频特征
    print("Extracting audio features...")
    input_features = processor(
        audio_array, 
        sampling_rate=sampling_rate, 
        return_tensors="pt"
    ).input_features.to(device)
    
    # 4. 设置生成参数（关键步骤）
    #forced_decoder_ids = processor.get_decoder_prompt_ids(
    #    language=language, 
    #    task="transcribe"
    #)
    # 使用新的生成参数，不再使用forced_decoder_ids
    generate_kwargs = {
        "language": language,  # 直接指定语言
        "task": "transcribe",  # 指定任务类型
        "num_beams": 5,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "compression_ratio_threshold": 2.4,
        "no_speech_threshold": 0.6,
    }

    
    # 5. 生成转录（自动处理长音频）
    print("Transcribing...")
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            **generate_kwargs,
            #forced_decoder_ids=forced_decoder_ids,
            # 以下参数可优化输出质量，防止重复
            #num_beams=5,               # 使用束搜索提高准确性
            #repetition_penalty=1.2,    # 重复惩罚系数
            #no_repeat_ngram_size=3,    # 禁止重复3个词的词组
            #temperature=0.8,           # 控制随机性
            #compression_ratio_threshold=2.4,  # 压缩比阈值
            #no_speech_threshold=0.6,   # 无语音阈值
        )
    
    # 6. 解码输出
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# 使用示例
if __name__ == "__main__":
    # 配置参数
    #MODEL_NAME = "openai/whisper-large-v3"  # 可替换为你微调的模型路径
    #AUDIO_PATH = "path/to/your/long_audio.wav"  # 替换为你的长音频文件路径
    MODEL_NAME = "/maduo/exp/asr_sd/whisper_nemo_dia/finetune_whisper_large_v3_on_mlc_slm_english_1epoch_train_strategy_steps_lr_1e-7_max_step5000/checkpoint-1000_tmp/"
    AUDIO_PATH = "/maduo/datasets/mlc-slm/dev/dev_wav/American/American_0517_007.wav"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LANGUAGE = "en"  # 根据你的带口音英文数据集设置
    
    print(f"Using device: {DEVICE}")
    
    # 执行转录
    try:
        transcript = transcribe_long_audio(
            model_name=MODEL_NAME,
            audio_path=AUDIO_PATH,
            device=DEVICE,
            language=LANGUAGE
        )
        
        print("\n" + "="*50)
        print("转录完成！")
        print("="*50)
        print(transcript)
        
    except Exception as e:
        print(f"转录过程中出错: {e}")
