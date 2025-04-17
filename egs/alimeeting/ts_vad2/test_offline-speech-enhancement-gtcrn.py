#!/usr/bin/env python3

"""
This file shows how to use the speech enhancement API.

Please download files used this script from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models

Example:

 wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
 wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/speech_with_noise.wav
"""

import time
from pathlib import Path
from typing import Tuple

import numpy as np
import sherpa_onnx
import soundfile as sf


def create_speech_denoiser():
    model_filename = "/maduo/model_hub/speech_enhancement_model/gtcrn/gtcrn_simple.onnx"
    if not Path(model_filename).is_file():
        raise ValueError(
            "Please first download a model from "
            "https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models"
        )

    config = sherpa_onnx.OfflineSpeechDenoiserConfig(
        model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
            gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(
                model=model_filename
            ),
            debug=False,
            num_threads=1,
            provider="cpu",
            #provider="gpu",
        )
    )
    if not config.validate():
        print(config)
        raise ValueError("Errors in config. Please check previous error logs")
    return sherpa_onnx.OfflineSpeechDenoiser(config)


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def main():
    sd = create_speech_denoiser()
    #test_wave = "/maduo/model_hub/speech_enhancement_model/gtcrn/speech_with_noise.wav"
    #test_wave = "temp/speech_with_noise1.wav"
    test_wave = "/maduo/datasets/alimeeting/Eval_Ali/Eval_Ali_far/target_audio/R8001_M8004_MS801/all.wav"
    if not Path(test_wave).is_file():
        raise ValueError(
            f"{test_wave} does not exist. You can download it from "
            "https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models"
        )

    samples, sample_rate = load_audio(test_wave)
    print(f"samples len: {len(samples)}")
    start = time.time()
    denoised = sd(samples, sample_rate)
    print(f"denoised.samples len: {len(denoised.samples)}")
    
    end = time.time()

    elapsed_seconds = end - start
    audio_duration = len(samples) / sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    sf.write("./enhanced_16k_1.wav", denoised.samples, denoised.sample_rate)
    print("Saved to ./enhanced_16k_1.wav")
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"Audio duration in seconds: {audio_duration:.3f}")
    print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")


if __name__ == "__main__":
    main()
