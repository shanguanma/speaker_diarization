from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

import time

ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='iic/speech_frcrn_ans_cirm_16k',
    device="cuda:1")

start = time.time()
result = ans(
    './temp/speech_with_noise1.wav',
    output_path='output.wav')
print(time.time() - start)

