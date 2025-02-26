from typing import List, Dict
from tqdm import tqdm

from datetime import datetime, timedelta

from pyannote.core import SlidingWindow
import sys
import numpy as np
import numpy

import scipy
#import sklearn
import json
import os
import torch
import logging
import soundfile

import torchaudio.compliance.kaldi as Kaldi
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
#import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster._kmeans import k_means

from examples.speaker_diarization.ts_vad.models.modules.cam_pplus_wespeaker import  CAMPPlus



class FBank(object):
    def __init__(
        self,
        n_mels,
        sample_rate,
        mean_nor: bool = False,
    ):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav, dither=0):
        sr = 16000
        assert sr == self.sample_rate
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        # select single channel
        if wav.shape[0] > 1:
            wav = wav[0, :]
        assert len(wav.shape) == 2 and wav.shape[0] == 1
        feat = Kaldi.fbank(
            wav, num_mel_bins=self.n_mels, sample_frequency=sr, dither=dither
        )
        # feat: [T, N]
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat
def init_speaker_encoder(pretrained_model):
#    speaker_encoder = CAMPPlus().cuda()
#    speaker_encoder.eval()
#    loadedState = torch.load(source, map_location="cuda")
#    selfState = speaker_encoder.state_dict()
#    for name, param in loadedState.items():
#        name = name.replace("speaker_encoder.", "")
#        if name in selfState:
#            print(f"name: {name}!")
#            selfState[name].copy_(param)
#        else:
#            print("Not exist ", name)
#    for param in speaker_encoder.parameters():
#        param.requires_grad = False

    if torch.cuda.is_available():
        msg = "Using gpu for inference."
        logging.info(f"{msg}")
        device = torch.device("cuda")
    else:
        msg = "No cuda device is detected. Using cpu."
        logging.info(f"{msg}")
        device = torch.device("cpu")

    pretrained_state = torch.load(pretrained_model, map_location=device)
    model =  CAMPPlus(embedding_size=192,feat_dim=80)
    model.load_state_dict(pretrained_state,strict=False)
    model.to(device)
    model.eval()
    return model


def extract_embeddings(data, model):
    #batch = torch.stack(batch)
    with torch.no_grad():
        embeddings = model.forward(data)
    return embeddings


def get_obj_from_rttm(rttm_file):
    rttm_dict = dict()
    speaker_set = set()
    filename_set = set()
    total_time = 0

    for line in open(rttm_file).readlines():
        items = line.replace("\n", "").split()
        filename, start_time_seconds, duration_seconds, speaker_id = (
            items[1],
            items[3],
            items[4],
            items[7],
        )
        speaker_set.add(speaker_id)
        total_time += float(duration_seconds)
        # SPEAKER CTS-CN-F2F-2019-11-15-160 1 1.008 7.5 <NA> <NA> G00000697 <NA> <NA>
        # SPEAKER CTS-CN-F2F-2019-11-15-160 1 11.355 2.78 <NA> <NA> G00000697 <NA> <NA>
        filename_set.add(filename)
        if filename not in rttm_dict.keys():
            rttm_dict[filename] = dict()
        if speaker_id not in rttm_dict[filename].keys():
            rttm_dict[filename][speaker_id] = []
        rttm_dict[filename][speaker_id].append(
            (
                float(start_time_seconds),
                float(start_time_seconds) + float(duration_seconds),
            )
        )
    return rttm_dict


def wavscp_to_dict(wav_scp: str):
    wavscp2dict=dict()
    with open(wav_scp,'r')as f:
        for line in f:
            line = line.strip().split()
            wavscp2dict[line[0]] = line[1]
    return wavscp2dict

def get_vad_dict(vad_type: str, oracle_rttm: str, predict_vad_path_dir: str, uttids: List ):
    vad_dict = dict()
    if vad_type == "oracle":
        from pyannote.core import (
            Segment,
            Timeline,
            Annotation,
        )

        rttm_dict = get_obj_from_rttm(oracle_rttm)

        for filename in rttm_dict.keys():
            timeline = Timeline()
            vad_dict[filename] = []
            for spkid in rttm_dict[filename].keys():
                for start, end in rttm_dict[filename][
                    spkid
                ]:
                    timeline.add(Segment(start, end))

            for seg in timeline.support():
                vad_dict[filename].append(
                    (seg.start, seg.end)
                )

    elif vad_type == "transfomer_vad":
        print(f"prepared vad!!!!")
        vad_pred_files =predict_vad_path_dir
        for filename in uttids:
            print(f"uttid: {filename}!!")
            vad_dict[filename] = []
            json_file = (
                os.path.join(vad_pred_files, f"{filename}.json")
            )
            pred_json_obj = json.load(open(json_file))
            for item in pred_json_obj["activities"]:
                print(f"before: start: {item['start']}, end: {item['end']}")
                #format_timedelta_to_milliseconds(parse_timecode_to_timedelta("00:00:01.550"))
                start = format_timedelta_to_milliseconds(parse_timecode_to_timedelta(item["start"]))/1000, # "00:00:00.811" -> 0.811
                end = format_timedelta_to_milliseconds(parse_timecode_to_timedelta(item["end"]))/1000,
                print(f"after: start: {start[0]} end: {end[0]}")
                vad_dict[filename].append((start[0],end[0]))

    #Convert and write JSON object to file
    with open("vad_dict_sample2_oral.json", "w") as outfile:
        json.dump(vad_dict, outfile)
    return vad_dict

def format_timedelta_to_milliseconds(t: timedelta) -> int:
    return int(t.total_seconds()*1000)

def parse_timecode_to_timedelta(timecode: str) -> timedelta:

    epoch = datetime(year=1900, month=1, day=1)
    return datetime.strptime(timecode, "%H:%M:%S.%f") - epoch

def get_cluster_label(emb_dict: Dict, cluster_type: str, vad_seg_dict: Dict):
    clust = Spec_Clust_unorm(
        min_num_spkrs=2, max_num_spkrs=2
    )
    clustering_id = dict()
    for k in tqdm(emb_dict.keys()):
        utt, emb = k, emb_dict[k]

        assert len(vad_seg_dict[k]) == len(emb_dict[k])

        if cluster_type == "sc":
            print("sc")

            N = 15

            if N > len(emb_dict[k]):
                p_val = 1.0
            else:
                p_val = N / len(emb_dict[k])

            clust.do_spec_clust(
                emb, k_oracle=2, p_val=p_val
            )
            # clust.do_spec_clust(emb, k_oracle=num, p_val=1.0)

            labels = clust.labels_
            assert len(vad_seg_dict[k]) == len(labels)
            clustering_id[k] = labels
        #elif cluster_type == "wq":
        #    print("wq")
        #    labels = clusterer.predict(np.array(emb))
        #    assert len(vad_seg_dict[k]) == len(labels)
        #    clustering_id[k] = labels
        elif cluster_type.startswith("ahc"):
            print("ahc")
            clustering = AgglomerativeClustering(
                affinity="cosine",
                linkage=cluster_type.split("_")[-1],
            ).fit(np.array(emb))
            #                                         import pdb;pdb.set_trace()
            assert (
                len(vad_seg_dict[k])
                == clustering.labels_.shape[0]
            )
            clustering_id[k] = clustering.labels_.tolist()

    print(f"clustering_id: {clustering_id}, {type(clustering_id)}")
    #with open("clustering_id.json", "w") as outfile:
    #    json.dump(clustering_id, outfile)
    #print(f"clustering_id: {clustering_id}, {type(clustering_id)}")
    return clustering_id

def get_speech_speaker_embedding(vad_dict: Dict, wav_scp: str, model, skip_chunk_size=0.93 ):
    emb_dict = dict()
    vad_seg_dict = dict()
    # 切分的时候按照多少秒去切，实测，越短效果越差，这里选取2秒或3秒，最终用的是3秒
    chunk_size=3
    segment_duration = int(chunk_size * 16000)
    step_duration = int(chunk_size * 16000)
    feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)

    if torch.cuda.is_available():
        msg = "Using gpu for inference."
        logging.info(f"{msg}")
        device = torch.device("cuda")
    else:
        msg = "No cuda device is detected. Using cpu."
        logging.info(f"{msg}")
        device = torch.device("cpu")


    for wav_file in tqdm(vad_dict.keys()):
        #wav_path = "%s%s.wav" % (wav_file_path, wav_file)
        wavscp2dict = wavscp_to_dict(wav_scp)
        wav_path = wavscp2dict[wav_file]
        assert os.path.exists(wav_path)
        audio, _ = soundfile.read(wav_path)

        if wav_file not in emb_dict.keys():
            emb_dict[wav_file] = []
            vad_seg_dict[wav_file] = []

        for start_seconds, end_seconds in vad_dict[
            wav_file
        ]:
            print(f"start_seconds: {start_seconds}, end_seconds: {end_seconds}")
            start_frames = int(start_seconds * 16000)
            end_frames = int(end_seconds * 16000)

            window = SlidingWindow(
                start=start_frames,
                end=end_frames,
                duration=segment_duration,
                step=step_duration,
            )

            last_end = None

            for i, duration in enumerate(window):

                if end_frames <= duration.end:
                    duration_end = end_frames
                else:
                    duration_end = duration.end
                audio_obj = audio[
                    duration.start : duration_end
                ]

                if (
                    audio_obj.shape[0]
                    < 16000 * skip_chunk_size
                ):
                    continue
                data = torch.FloatTensor(np.array(audio_obj))
                #data = torch.FloatTensor(
                #    np.stack([audio_obj], axis=0)
                #).cuda()
                # compute feat
                with torch.no_grad():
                    feat = feature_extractor(data)  # (T,F)
                    embedding = model.forward(feat.unsqueeze(0).to(device))
                emb_dict[wav_file].append(embedding.cpu())
                #del embedding
                torch.cuda.empty_cache()
                if len(window) != 1:
                    if i == 0:
                        # start
                        vad_seg_dict[wav_file].append(
                            (
                                duration.start / 16000,
                                duration.start / 16000
                                + (
                                    audio_obj.shape[0]
                                    + step_duration
                                )
                                / 2
                                / 16000,
                            )
                        )
                        last_end = (
                            duration.start / 16000
                            + (
                                audio_obj.shape[0]
                                + step_duration
                            )
                            / 2
                            / 16000
                        )
                    elif i == len(window) - 1:
                        # end
                        vad_seg_dict[wav_file].append(
                            (
                                last_end,
                                (
                                    duration.start
                                    + audio_obj.shape[0]
                                )
                                / 16000,
                            )
                        )


                    else:
                        vad_seg_dict[wav_file].append(
                            (
                                last_end,
                                duration.start / 16000
                                + (
                                    audio_obj.shape[0]
                                    + step_duration
                                )
                                / 2
                                / 16000,
                            )
                        )
                        last_end = (
                            duration.start / 16000
                            + (
                                audio_obj.shape[0]
                                + step_duration
                            )
                            / 2
                            / 16000
                        )
                #                                         import pdb;pdb.set_trace()
                else:
                    vad_seg_dict[wav_file].append(
                        (
                            duration.start / 16000,
                            (
                                duration.start
                                + audio_obj.shape[0]
                            )
                            / 16000,
                        )
                    )

    for key in emb_dict.keys():
        new_numpy_lists = []
        for item in emb_dict[key]:
            new_numpy_lists.append(
                item.cpu().detach().numpy().tolist()[0]
            )
        emb_dict[key] = new_numpy_lists

        assert len(vad_seg_dict[key]) == len(emb_dict[key])

     #Convert and write JSON object to file
    with open("vad_seg_dict2_oral.json", "w") as outfile:
        json.dump(vad_seg_dict, outfile)

    with open("emb_dict2_oral.json", "w") as outfile:
        json.dump(emb_dict, outfile)

    return vad_seg_dict, emb_dict

def write_output_rttm_result(saved_to_rttm_file_path: str, vad_seg_dict: Dict, clustering_id: Dict):
    saved_to_rttm_file_path_obj = open(
        saved_to_rttm_file_path, "w"
    )

    for wav_filename in vad_seg_dict.keys():
        for vad_seg, spkid in zip(
            vad_seg_dict[wav_filename],
            clustering_id[wav_filename],
        ):
            saved_to_rttm_file_path_obj.write(
                "SPEAKER %s 1 %f %f <NA> <NA> %s <NA> <NA>\n"
                % (
                    wav_filename,
                    vad_seg[0],
                    vad_seg[1] - vad_seg[0],
                    spkid,
                )
            )

    saved_to_rttm_file_path_obj.close()

class Spec_Clust_unorm:
    def __init__(self, min_num_spkrs=1, max_num_spkrs=10):

        self.min_num_spkrs = min_num_spkrs
        self.max_num_spkrs = max_num_spkrs

    def do_spec_clust(self, X, k_oracle, p_val):
        """Function for spectral clustering.
        Arguments
        ---------
        X : array
            (n_samples, n_features).
            Embeddings extracted from the model.
        k_oracle : int
            Number of speakers (when oracle number of speakers).
        p_val : float
            p percent value to prune the affinity matrix.
        """

        # Similarity matrix computation
        sim_mat = self.get_sim_mat(X)

        # Refining similarity matrix with p_val
        prunned_sim_mat = self.p_pruning(sim_mat, p_val)

        # Symmetrization
        sym_prund_sim_mat = 0.5 * (prunned_sim_mat + prunned_sim_mat.T)

        # Laplacian calculation
        laplacian = self.get_laplacian(sym_prund_sim_mat)

        # Get Spectral Embeddings
        emb, num_of_spk = self.get_spec_embs(laplacian, k_oracle)

        # Perform clustering
        # import pdb;pdb.set_trace()
        self.cluster_embs(emb, num_of_spk)

    def get_sim_mat(self, X):
        """Returns the similarity matrix based on cosine similarities.
        Arguments
        ---------
        X : array
            (n_samples, n_features).
            Embeddings extracted from the model.
        Returns
        -------
        M : array
            (n_samples, n_samples).
            Similarity matrix with cosine similarities between each pair of embedding.
        """

        # Cosine similarities
        M = cosine_similarity(X, X)
        return M

    def p_pruning(self, A, pval):
        """Refine the affinity matrix by zeroing less similar values.
        Arguments
        ---------
        A : array
            (n_samples, n_samples).
            Affinity matrix.
        pval : float
            p-value to be retained in each row of the affinity matrix.
        Returns
        -------
        A : array
            (n_samples, n_samples).
            Prunned affinity matrix based on p_val.
        """

        n_elems = int((1 - pval) * A.shape[0])

        # For each row in a affinity matrix
        for i in range(A.shape[0]):
            low_indexes = np.argsort(A[i, :])
            low_indexes = low_indexes[0:n_elems]

            # Replace smaller similarity values by 0s
            A[i, low_indexes] = 0

        return A

    def get_laplacian(self, M):
        """Returns the un-normalized laplacian for the given affinity matrix.
        Arguments
        ---------
        M : array
            (n_samples, n_samples)
            Affinity matrix.
        Returns
        -------
        L : array
            (n_samples, n_samples)
            Laplacian matrix.
        """

        M[np.diag_indices(M.shape[0])] = 0
        D = np.sum(np.abs(M), axis=1)
        D = np.diag(D)
        # L is a signed laplacian given the definition of D
        L = D - M
        return L

    def get_spec_embs(self, L, k_oracle):
        """Returns spectral embeddings and estimates the number of speakers
        using maximum Eigen gap.
        Arguments
        ---------
        L : array (n_samples, n_samples)
            Laplacian matrix.
        k_oracle : int
            Number of speakers when the condition is oracle number of speakers,
            else None.
        Returns
        -------
        emb : array (n_samples, n_components)
            Spectral embedding for each sample with n Eigen components.
        num_of_spk : int
            Estimated number of speakers. If the condition is set to the oracle
            number of speakers then returns k_oracle.
        """

        lambdas, eig_vecs = scipy.linalg.eigh(L)

        # if params["oracle_n_spkrs"] is True:
        if k_oracle is not None:
            num_of_spk = k_oracle
        else:
            lambda_gap_list = self.getEigenGaps(lambdas[0 : self.max_num_spkrs])

            num_of_spk = (
                np.argmax(
                    lambda_gap_list[
                        : min(self.max_num_spkrs, len(lambda_gap_list))
                    ]
                )
                + 1
            )

            if num_of_spk < self.min_num_spkrs:
                num_of_spk = self.min_num_spkrs

        emb = eig_vecs[:, 0:num_of_spk]

        return emb, num_of_spk

    def cluster_embs(self, emb, k):
        """Clusters the embeddings using kmeans.
        Arguments
        ---------
        emb : array (n_samples, n_components)
            Spectral embedding for each sample with n Eigen components.
        k : int
            Number of clusters to kmeans.
        Returns
        -------
        self.labels_ : self
            Labels for each sample embedding.
        """
        _, self.labels_, _ = k_means(emb, k, random_state=1)

    def getEigenGaps(self, eig_vals):
        """Returns the difference (gaps) between the Eigen values.
        Arguments
        ---------
        eig_vals : list
            List of eigen values
        Returns
        -------
        eig_vals_gap_list : list
            List of differences (gaps) between adjancent Eigen values.
        """

        eig_vals_gap_list = []
        for i in range(len(eig_vals) - 1):
            gap = float(eig_vals[i + 1]) - float(eig_vals[i])
            # eig_vals_gap_list.append(float(eig_vals[i + 1]) - float(eig_vals[i]))
            eig_vals_gap_list.append(gap)

        print(eig_vals_gap_list, file=sys.stderr)
        return eig_vals_gap_list

if __name__ == "__main__":
    #wavscp2dict = wavscp_to_dict(wav_scp)
    #for
    ## this test_set actually is uttids
    test_set_dir="data/magicdata-RAMC/test/"
    uttids = set(line.split()[1] for line in open(f"{test_set_dir}/rttm").readlines())
    vad_threshold=0.9

    vad_dict = get_vad_dict(vad_type="oracle", oracle_rttm="data/magicdata-RAMC/test/rttm", predict_vad_path_dir="data/magicdata-RAMC/test/predict_vad", uttids=uttids)
    pretrain_speaker_model_ckpt="/mntcephfs/lab_data/maduo/model_hub/speaker_pretrain_model/zh/modelscope/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
    model = init_speaker_encoder(pretrain_speaker_model_ckpt)
    vad_seg_dict, emb_dict = get_speech_speaker_embedding(vad_dict=vad_dict, wav_scp="data/magicdata-RAMC/test/wav.scp", model=model, skip_chunk_size=0.93 )
    clustering_id = get_cluster_label(emb_dict=emb_dict, cluster_type='sc', vad_seg_dict=vad_seg_dict)
    predict_rttm_path="data/magicdata-RAMC/test/rttm_predict2_oral"
    write_output_rttm_result(saved_to_rttm_file_path=predict_rttm_path, vad_seg_dict=vad_seg_dict, clustering_id=clustering_id)

    """
    vad_seg_dict_file="vad_seg_dict.json"
    emb_dict_file="emb_dict.json"
    with open(vad_seg_dict_file,'r')as fv,open(emb_dict_file,'r')as fe:
        emb_dict = json.load(fe)
        vad_seg_dict = json.load(fv)
        clustering_id = get_cluster_label(emb_dict=emb_dict, cluster_type='sc', vad_seg_dict=vad_seg_dict)
        predict_rttm_path="data/magicdata-RAMC/test/rttm_predict"
        write_output_rttm_result(saved_to_rttm_file_path=predict_rttm_path, vad_seg_dict=vad_seg_dict, clustering_id=clustering_id)
    """
