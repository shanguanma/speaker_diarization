import torch.nn as nn
import torch
import librosa
import soundfile as sf
import glob
import numpy as np
import wave
import random
from pathlib import Path
import os
from utils import fix_random_seed,none_or_str
import sys
import argparse
from scipy import signal

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--musan-path",
        type=none_or_str,
        nargs="?",
        default=None,
        help="musan noise wavform directory path",
    )
    parser.add_argument(
        "--rir-path",
        type=none_or_str,
        nargs="?",
        default=None,
        help="rir noise wavform directory path",
    )
    parser.add_argument(
        "--mix-audio-mono-dir",
        type=str,
        default="/maduo/datasets/alimeeting/Train_Ali_far/target_audio",  # mono mix audio path
        help="path to mix audio directory.",
    )
    parser.add_argument(
        "--speech-enhancement-model-type",
        type=str,
        default="modelscope_zipenhancer",
        help="choice it from `modelscope_zipenhancer` and `sherpa_onnx_gtcrn`"
    )
    parser.add_argument("--output-dir", type=str, help="speech enhanced audio store directory")
    return parser
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

class AddNoiseRirSpeechEnhancement(nn.Module):
    def __init__(self,mix_audio_mono_dir: str, musan_path: str=None, rir_path: str=None,noise_ratio: float=0.8, sample_rate: int=16000, speech_enhancement_model_type: str="modelscope_zipenhancer"):
        self.speech_enhancement_model_type=speech_enhancement_model_type
        self.mix_audio_mono_dir=mix_audio_mono_dir
        self.sample_rate = sample_rate
        self.noise_ratio=noise_ratio
        self.musan_path = musan_path
        self.rir_path = rir_path
        #audio_path = os.path.join(self.mix_audio_mono_dir, file + "/all.wav") # mono mix audio

        if musan_path is not None or  rir_path is not None:
            self.noisesnr, self.numnoise, self.noiselist, self.rir_files = self.load_musan_or_rirs(musan_path,rir_path)
        if self.speech_enhancement_model_type=="modelscope_zipenhancer":
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            self.ans =  pipeline(Tasks.acoustic_noise_suppression,model='iic/speech_zipenhancer_ans_multiloss_16k_base',device="cuda:1") # require modescope>=1.20
        elif self.speech_enhancement_model_type=="sherpa_onnx_gtcrn":
            import sherpa_onnx
            self.ans = create_speech_denoiser()
        else:
             raise Exception(
                    f"The given self.speech_enhancement_model_type {self.speech_enhancement_model_type} is not supported."
                )
    def process(self, output_dir: str):
        fix_random_seed(1337)  # fairseq1 seed=1337
        # 0. load mix audio to list
        mix_audio_list= glob.glob(os.path.join(self.mix_audio_mono_dir, "*/all.wav"))
        for audio_path in mix_audio_list:
            ref_speech,_ = sf.read(audio_path,always_2d=True,dtype="float32") #(sample_points, 1)
            print(f"ref_speech shape: {ref_speech.shape} in input")
            #if len(ref_speech.shape) == 1:
            #    ref_speech = np.expand_dims(np.array(ref_speech), axis=0)
            ref_speech = np.transpose(ref_speech) # (1,sample_points)
            print(f"ref_speech shape: {ref_speech.shape} after transpose")
            frame_len = ref_speech.size
            #  /maduo/datasets/MagicData-RAMC/maduo_processed/kaldi_format/train/target_audio/CTS-CN-F2F-2019-11-15-995/all.wav
            dest_path="/".join(audio_path.split("/")[-4:-1]) # 'train/target_audio/CTS-CN-F2F-2019-11-15-995'
            dest_dir = os.path.join(output_dir, dest_path)
            os.makedirs(dest_dir, exist_ok=True)
            #dest_file = os.path.join(dest_dir,"/all.wav")
            #print(f"dest_file: {dest_file}")
            # 1. add noise and add rir
            if self.rir_path is not None:
                print(f"ref_speech shape: {ref_speech.shape} in input fn add_rev")
                ref_speech = self.add_rev(ref_speech,length=frame_len)
                print(f"ref_speech shape: {ref_speech.shape} in outputput fn add_rev")
            if self.musan_path is not None:
                print(f"ref_speech shape: {ref_speech.shape} in input fn add_noise")
                ref_speech = self.choose_and_add_noise(
                    random.randint(0, 2), ref_speech, frame_len
                ) 
                print(f"ref_speech shape: {ref_speech.shape} in output fn add_noise")
            # add speech enhancement
            if self.speech_enhancement_model_type=="modelscope_zipenhancer":
                print(f"ref_speech shape: {ref_speech.shape}")
                ref_speech = self.add_speech_enhance_augment_modelscope_model_online(ref_speech,frame_len)
            elif self.speech_enhancement_model_type=="sherpa_onnx_gtcrn":
                ref_speech = self.add_speech_enhance_augment_sherpa_onnx_online(ref_speech,frame_len)

            # save 
            sf.write(f"{dest_dir}/all.wav",ref_speech,self.sample_rate)
                    
    def choose_and_add_noise(self, noise_type, ref_speech, frame_len):
        assert self.musan_path is not None
        if noise_type == 0:
            return self.add_noise(ref_speech, "speech", length=frame_len)
        elif noise_type == 1:
            return self.add_noise(ref_speech, "music", length=frame_len)
        elif noise_type == 2:
            return self.add_noise(ref_speech, "noise", length=frame_len)

    def add_speech_enhance_augment_modelscope_model_online(self,ref_speech: np.ndarray, length):
        #ans = pipeline(Tasks.acoustic_noise_suppression,model='iic/speech_zipenhancer_ans_multiloss_16k_base') # require modescope>=1.20
        # I modified the codes ` /maduo/miniconda3/envs/dia_cuda11.8_py311/lib/python3.11/site-packages/modelscope/pipelines/audio/ans_pipeline.py` to support np.arrary input
        # ref_speech is np.float32
        ref_speech = np.squeeze(ref_speech,axis=0)
        result = self.ans(ref_speech)
        enhance_bytes = result['output_pcm']
        enhance_numpy = np.frombuffer(enhance_bytes, dtype=np.int16 )
        # convert int16 to float32
        pcm_float32_data = enhance_numpy.astype(np.float32) / 32768.0

        pcm_float32_data = np.array(pcm_float32_data)
        return pcm_float32_data[:length] #(sample_points)


    def add_speech_enhance_augment_sherpa_onnx_online(self,ref_speech: np.ndarray, length):
        ref_speech = np.squeeze(ref_speech,axis=0) # (1, sample_points) ->(sample_points)
        samples = np.ascontiguousarray(ref_speech)
        result = self.ans(samples,self.sample_rate)
        pcm_float32_data = np.array(result.samples)
        return pcm_float32_data[:length] #(sample_points)


    def load_musan_or_rirs(self, musan_path, rir_path):
        # add noise and rir augment
        if musan_path is not None:
            noiselist = {}
            noisetypes = ["noise", "speech", "music"]
            noisesnr = {"noise": [0, 15], "speech": [13, 20], "music": [5, 15]}
            numnoise = {"noise": [1, 1], "speech": [3, 8], "music": [1, 1]}

            augment_files = glob.glob(
                os.path.join(musan_path, "*/*/*.wav")
            )  # musan/*/*/*.wav
            for file in augment_files:
                if file.split("/")[-3] not in noiselist:
                    noiselist[file.split("/")[-3]] = []
                noiselist[file.split("/")[-3]].append(file)

        if rir_path is not None:
            rir_files = glob.glob(
                os.path.join(rir_path, "*/*.wav")
            )  # RIRS_NOISES/*/*.wav
        return noisesnr, numnoise, noiselist, rir_files
    
    def add_rev(self, audio,length):
        rir_file = random.choice(self.rir_files)
        rir, _ = self.read_audio_with_resample(rir_file)
        rir = np.expand_dims(rir.astype(float), 0)
        rir = rir / np.sqrt(np.sum(rir**2))
        print(f"audio shape: {audio.shape}")
        rev_audio =  signal.convolve(audio, rir, mode="full")
        print(f"afer rir audio shape: {rev_audio.shape}")
        rev_audio = rev_audio[:,:length]
        return rev_audio
    def add_noise(self, audio, noisecat, length):
        clean_db = 10 * np.log10(max(1e-4, np.mean(audio**2)))
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(
            self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1])
        )
        noises = []
        for noise in noiselist:
            noiselength = wave.open(noise, "rb").getnframes()
            noise_sample_rate = wave.open(noise, "rb").getframerate()
            if noise_sample_rate != self.sample_rate:
                noiselength = int(noiselength * self.sample_rate / noise_sample_rate)
            if noiselength <= length:
                noiseaudio, _ = self.read_audio_with_resample(noise)
                noiseaudio = np.pad(noiseaudio, (0, length - noiselength), "wrap")
            else:
                start_frame = np.int64(random.random() * (noiselength - length))
                noiseaudio, _ = self.read_audio_with_resample(
                    noise, start=start_frame, length=length
                )
            noiseaudio = np.stack([noiseaudio], axis=0)
            noise_db = 10 * np.log10(max(1e-4, np.mean(noiseaudio**2)))
            noisesnr = random.uniform(
                self.noisesnr[noisecat][0], self.noisesnr[noisecat][1]
            )
            noises.append(
                np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio
            )
        noise = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        if noise.shape[1] < length:
            assert length - noise.shape[1] < 10
            audio[:, : noise.shape[1]] = noise + audio[:, : noise.shape[1]]
            return audio
        else:
            return noise[:, :length] + audio

    def read_audio_with_resample(
        self, audio_path, start=None, length=None, sr_cur=None, support_mc=False, rc=-1
    ):
        if sr_cur is None:
            sr_cur = self.sample_rate
        audio_sr = librosa.get_samplerate(audio_path)

        if audio_sr != self.sample_rate:
            try:
                if start is not None:
                    audio, _ = librosa.load(
                        audio_path,
                        offset=start / sr_cur,
                        duration=length / sr_cur,
                        sr=sr_cur,
                        mono=False,
                    )
                else:
                    audio, _ = librosa.load(audio_path, sr=sr_cur, mono=False)
            except Exception as e:
                logger.info(e)
                audio, _ = librosa.load(audio_path, sr=sr_cur, mono=False)
                audio = audio[start : start + length]
            if len(audio.shape) > 1:
                # use first channel
                audio = audio[0, :]
        else:
            try:
                if start is not None:
                    audio, _ = sf.read(
                        audio_path, start=start, stop=start + length, dtype="float32"
                    )
                else:
                    audio, _ = sf.read(audio_path, dtype="float32")
            except Exception as e:
                logger.info(e)
                audio, _ = sf.read(audio_path, dtype="float32")
                audio = audio[start : start + length]
            if len(audio.shape) > 1:
                # use first channel
                audio = audio[:, 0]
            else:
                audio = np.transpose(audio)

        return audio, rc

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    enhancer =  AddNoiseRirSpeechEnhancement(args.mix_audio_mono_dir, musan_path=args.musan_path, rir_path=args.rir_path,speech_enhancement_model_type=args.speech_enhancement_model_type)
    enhancer.process(args.output_dir)
