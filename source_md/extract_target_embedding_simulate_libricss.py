import torch, os, tqdm, numpy, soundfile, argparse, glob, wave
from ts_vad.models.modules.speakerEncoder2 import ECAPA_TDNN
from joblib import Parallel, delayed


def init_speaker_encoder(source):
    speaker_encoder = ECAPA_TDNN(C=1024).cuda()
    speaker_encoder.eval()
    loadedState = torch.load(source, map_location="cuda")
    selfState = speaker_encoder.state_dict()
    for name, param in loadedState.items():
        name = name.replace("speaker_encoder.", "")
        if name in selfState:
            selfState[name].copy_(param)
        else:
            print("Not exist ", name)
    for param in speaker_encoder.parameters():
        param.requires_grad = False
    return speaker_encoder


def extract_embeddings(batch, model):
    batch = torch.stack(batch)
    with torch.no_grad():
        embeddings = model.forward(batch.cuda())
    return embeddings


def extract_embed(args, file, model):
    output_file = (
        args.target_embedding_path
        + "/"
        + file.split("/")[-2]
        + "/"
        + file.split("/")[-1].replace(".wav", ".pt")
    )
    if "all" not in file and "_" not in file.split("/")[-1]:
        batch = []
        embeddings = []
        wav_length = wave.open(
            file, "rb"
        ).getnframes()  # entire length for target speech
        if wav_length > int(args.length_embedding * 16000):
            for start in range(
                0,
                wav_length - int(args.length_embedding * 16000),
                int(args.step_embedding * 16000),
            ):
                stop = start + int(args.length_embedding * 16000)
                target_speech, _ = soundfile.read(file, start=start, stop=stop)
                target_speech = torch.FloatTensor(numpy.array(target_speech))
                batch.append(target_speech)
                if len(batch) == args.batch_size:
                    embeddings.extend(extract_embeddings(batch, model))
                    batch = []
        else:
            embeddings.extend(
                extract_embeddings(
                    [torch.FloatTensor(numpy.array(soundfile.read(file)[0]))], model
                )
            )

        if len(batch) != 0:
            embeddings.extend(extract_embeddings(batch, model))

        embeddings = torch.stack(embeddings)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        torch.save(embeddings, output_file)


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--target_audio_path", help="the path for the audio")
    parser.add_argument(
        "--target_embedding_path", help="the path for the output embeddings"
    )
    parser.add_argument("--source", help="the part for the speaker encoder")
    parser.add_argument(
        "--length_embedding",
        type=float,
        default=6,
        help="length of embeddings, seconds",
    )
    parser.add_argument(
        "--step_embedding", type=float, default=1, help="step of embeddings, seconds"
    )
    parser.add_argument(
        "--batch_size", type=int, default=96, help="step of embeddings, seconds"
    )
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    # files = []
    # with open(args.target_audio_path) as f:
    #     for line in f:
    #         files.append(line.strip('\n'))
    files = glob.glob(args.target_audio_path + "/*/*wav")
    model = init_speaker_encoder(args.source)
    Parallel(n_jobs=3)(delayed(extract_embed)(args, file, model) for file in files)


if __name__ == "__main__":
    main()
