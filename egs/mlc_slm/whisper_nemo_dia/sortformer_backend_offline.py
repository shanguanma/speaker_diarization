import numpy as np
import torch
import logging

from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
import librosa

logger = logging.getLogger(__name__)

def load_model():

    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2")
    diar_model.eval()

    #if torch.cuda.is_available():
    #    diar_model.to(torch.device("cuda"))

    #we target 1 second lag for the moment. chunk_len could be reduced.
    diar_model.sortformer_modules.chunk_len = 10
    diar_model.sortformer_modules.subsampling_factor = 10 #8 would be better ideally

    diar_model.sortformer_modules.chunk_right_context = 0 #no.
    diar_model.sortformer_modules.chunk_left_context = 10 #big so it compensiate the problem with no padding later.

    diar_model.sortformer_modules.spkcache_len = 188
    diar_model.sortformer_modules.fifo_len = 188
    diar_model.sortformer_modules.spkcache_update_period = 144
    diar_model.sortformer_modules.log = False
    diar_model.sortformer_modules._check_streaming_parameters()


    audio2mel = AudioToMelSpectrogramPreprocessor(
            window_size= 0.025, 
            normalize="NA",
            n_fft=512,
            features=128,
            pad_to=0) #pad_to 16 works better than 0. On test audio, we detect a third speaker for 1 second with pad_to=0. To solve that : increase left context to 10.

    return diar_model, audio2mel

diar_model, audio2mel = load_model()

class StreamingSortformerState:
    """
    This class creates a class instance that will be used to store the state of the
    streaming Sortformer model.

    Attributes:
        spkcache (torch.Tensor): Speaker cache to store embeddings from start
        spkcache_lengths (torch.Tensor): Lengths of the speaker cache
        spkcache_preds (torch.Tensor): The speaker predictions for the speaker cache parts
        fifo (torch.Tensor): FIFO queue to save the embedding from the latest chunks
        fifo_lengths (torch.Tensor): Lengths of the FIFO queue
        fifo_preds (torch.Tensor): The speaker predictions for the FIFO queue parts
        spk_perm (torch.Tensor): Speaker permutation information for the speaker cache
        mean_sil_emb (torch.Tensor): Mean silence embedding
        n_sil_frames (torch.Tensor): Number of silence frames
    """

    spkcache = None  # Speaker cache to store embeddings from start
    spkcache_lengths = None  #
    spkcache_preds = None  # speaker cache predictions
    fifo = None  # to save the embedding from the latest chunks
    fifo_lengths = None
    fifo_preds = None
    spk_perm = None
    mean_sil_emb = None
    n_sil_frames = None


def init_streaming_state(self, batch_size: int = 1, async_streaming: bool = False, device: torch.device = None):
    """
    Initializes StreamingSortformerState with empty tensors or zero-valued tensors.

    Args:
        batch_size (int): Batch size for tensors in streaming state
        async_streaming (bool): True for asynchronous update, False for synchronous update
        device (torch.device): Device for tensors in streaming state

    Returns:
        streaming_state (SortformerStreamingState): initialized streaming state
    """
    streaming_state = StreamingSortformerState()
    if async_streaming:
        streaming_state.spkcache = torch.zeros((batch_size, self.spkcache_len, self.fc_d_model), device=device)
        streaming_state.spkcache_preds = torch.zeros((batch_size, self.spkcache_len, self.n_spk), device=device)
        streaming_state.spkcache_lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)
        streaming_state.fifo = torch.zeros((batch_size, self.fifo_len, self.fc_d_model), device=device)
        streaming_state.fifo_lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)
    else:
        streaming_state.spkcache = torch.zeros((batch_size, 0, self.fc_d_model), device=device)
        streaming_state.fifo = torch.zeros((batch_size, 0, self.fc_d_model), device=device)
    streaming_state.mean_sil_emb = torch.zeros((batch_size, self.fc_d_model), device=device)
    streaming_state.n_sil_frames = torch.zeros((batch_size,), dtype=torch.long, device=device)
    return streaming_state


def process_diarization(chunks):
    """ 
    what it does:
    1. Preprocessing: Applies dithering and pre-emphasis (high-pass filter) if enabled
    2. STFT: Computes the Short-Time Fourier Transform using:
        - the window of window_size=0.025 --> size of a window : 400 samples
        - the hop parameter : n_window_stride = 0.01 -> every 160 samples, a new window
    3. Magnitude Calculation: Converts complex STFT output to magnitude spectrogram
    4. Mel Conversion: Applies Mel filterbanks (128 filters in this case) to get Mel spectrogram
    5. Logarithm: Takes the log of the Mel spectrogram (if `log=True`)
    6. Normalization: Skips normalization since `normalize="NA"`
    7. Padding: Pads the time dimension to a multiple of `pad_to` (default 16)    
    """
    previous_chunk = None
    l_chunk_feat_seq_t = []
    print(f"diar_model.device: {diar_model.device}")
    for chunk in chunks:
        audio_signal_chunk = torch.tensor(chunk).unsqueeze(0).to(diar_model.device)
        audio_signal_length_chunk = torch.tensor([audio_signal_chunk.shape[1]]).to(diar_model.device)
        print(f"audio_signal_chunk device: {audio_signal_chunk.device}, audio_signal_length_chunk device: {audio_signal_length_chunk.device}")
        processed_signal_chunk, processed_signal_length_chunk = audio2mel.get_features(audio_signal_chunk, audio_signal_length_chunk)
        if previous_chunk is not None:
            to_add = previous_chunk[:, :, -99:]
            total = torch.concat([to_add, processed_signal_chunk], dim=2)
        else:
            total = processed_signal_chunk
        previous_chunk = processed_signal_chunk
        l_chunk_feat_seq_t.append(torch.transpose(total, 1, 2))

    batch_size = 1
    streaming_state = init_streaming_state(diar_model.sortformer_modules,
        batch_size = batch_size,
        async_streaming = True,
        device = diar_model.device
    )
    total_preds = torch.zeros((batch_size, 0, diar_model.sortformer_modules.n_spk), device=diar_model.device)

    chunk_duration_seconds = diar_model.sortformer_modules.chunk_len * diar_model.sortformer_modules.subsampling_factor * diar_model.preprocessor._cfg.window_stride

    l_speakers = [
        {'start_time': 0,
        'end_time': 0,
        'speaker': 0
        }
    ]
    len_prediction = None
    left_offset = 0
    right_offset = 8
    for i, chunk_feat_seq_t in enumerate(l_chunk_feat_seq_t):
        with torch.inference_mode():
                print(f"chunk_feat_seq_t device: {chunk_feat_seq_t.device}, ")
                streaming_state, total_preds = diar_model.forward_streaming_step(
                    processed_signal=chunk_feat_seq_t.to(diar_model.device),
                    processed_signal_length=torch.tensor([chunk_feat_seq_t.shape[1]]).to(diar_model.device),
                    streaming_state=streaming_state,
                    total_preds=total_preds,
                    left_offset=left_offset,
                    right_offset=right_offset,
                )
                left_offset = 8
                preds_np = total_preds[0].cpu().numpy()
                active_speakers = np.argmax(preds_np, axis=1)
                if len_prediction is None:
                    len_prediction = len(active_speakers) # we want to get the len of 1 prediction
                frame_duration = chunk_duration_seconds / len_prediction
                active_speakers = active_speakers[-len_prediction:]
                for idx, spk in enumerate(active_speakers):
                    if spk != l_speakers[-1]['speaker']:
                        l_speakers.append(
                            {'start_time': (i * chunk_duration_seconds + idx * frame_duration),
                            'end_time': (i * chunk_duration_seconds + (idx + 1) * frame_duration),
                            'speaker': spk
                        })                    
                    else:
                        l_speakers[-1]['end_time'] = i * chunk_duration_seconds + (idx + 1) * frame_duration
                    
        
        """
        Should print
        [{'start_time': 0, 'end_time': 8.72, 'speaker': 0}, 
        {'start_time': 8.72, 'end_time': 18.88, 'speaker': 1},
        {'start_time': 18.88, 'end_time': 24.96, 'speaker': 2},
        {'start_time': 24.96, 'end_time': 31.68, 'speaker': 0}]
        """
    for speaker in l_speakers:
        print(f"Speaker {speaker['speaker']}: {speaker['start_time']:.2f}s - {speaker['end_time']:.2f}s")    
    

if __name__ == '__main__':
    
    an4_audio = '/maduo/datasets/mlc-slm/dev/dev_wav/American/American_0517_007.wav'
    signal, sr = librosa.load(an4_audio, sr=16000)
    signal = signal[:16000*30]
    # signal = signal[:-(len(signal)%16000)]

    print("\n" + "=" * 50)
    print("Expected ground truth:")
    print("Speaker 0: 0:00 - 0:09")
    print("Speaker 1: 0:09 - 0:19") 
    print("Speaker 2: 0:19 - 0:25")
    print("Speaker 0: 0:25 - 0:30")
    print("=" * 50)

    chunk_size = 16000  # 1 second
    chunks = []
    for i in range(0, len(signal), chunk_size):
        chunk = signal[i:i+chunk_size]
        chunks.append(chunk)
    
    process_diarization(chunks)
