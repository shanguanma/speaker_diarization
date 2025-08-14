import whisperx
import gc

device = "cuda"
audio_file = "audio.mp3"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
# 3. Assign speaker labels
use_auth_token="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"
diarize_model = whisperx.DiarizationPipeline(use_auth_token=use_auth_token, device=device)

