from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='iic/speech_zipenhancer_ans_multiloss_16k_base') # require modescope>=1.20

#help(ans)
#result = ans(
#    'https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/speech_with_noise1.wav',
#    output_path='output.wav')
#print("done")
def audio_norm(x):
    rms = (x**2).mean()**0.5
    scalar = 10**(-25 / 20) / rms
    x = x * scalar
    pow_x = x**2
    avg_pow_x = pow_x.mean()
    rmsx = pow_x[pow_x > avg_pow_x].mean()**0.5
    scalarx = 10**(-25 / 20) / rmsx
    x = x * scalarx
    return x

import soundfile as sf
import numpy as np
import torch
wav,_ = sf.read('simulated_codes3_sample_0.wav')
wav = audio_norm(wav).astype(np.float32)
noisy_wav_bytes_or_str_or_numpy = wav

#noisy_wav = torch.from_numpy(np.reshape(wav, [1, wav.shape[0]]))

result = ans(noisy_wav_bytes_or_str_or_numpy)
#print(f"result type: {type(result)}")
#print(f"output_path:{enhance_wav}")
#print("done")
enhance_bytes = result['output_pcm']
enhance_numpy = np.frombuffer(enhance_bytes, dtype=np.int16 )
pcm_float32_data = enhance_numpy.astype(np.float32) / 32768.0
print(f"enhance_wav: {pcm_float32_data}, its shape: {pcm_float32_data.shape}")
