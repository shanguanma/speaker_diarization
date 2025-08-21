from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch
import time
from typing import Tuple
import  numpy as np
import soundfile as sf
#torch.set_num_interop_threads(32)
#torch.set_num_threads(32)

def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate

ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='iic/speech_zipenhancer_ans_multiloss_16k_base',
    device="cuda:1") # require modescope>=1.20

input_samples, input_sample_rate = load_audio('/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/target_audio/R8001_M8004_MS801/all.wav')
print(f"input_samples len: {len(input_samples)}")
start = time.time()
result = ans(
    #'./temp/speech_with_noise1.wav',
    '/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/target_audio/R8001_M8004_MS801/all.wav',
    output_path='output.wav')
samples, sample_rate = load_audio('output.wav')
print(f"output samples len: {len(samples)}")
print(f"consume time: {time.time() - start}!!!")
print("done")
