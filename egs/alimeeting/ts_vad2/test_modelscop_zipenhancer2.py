from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch
import time
#torch.set_num_interop_threads(32)
#torch.set_num_threads(32)
ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='iic/speech_zipenhancer_ans_multiloss_16k_base',
    device="cuda:1") # require modescope>=1.20

#help(ans)
start = time.time()
result = ans(
    'speech_with_noise1.wav',
    output_path='output.wav')
print(time.time() - start)
print("done")
