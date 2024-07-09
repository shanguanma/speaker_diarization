import os, json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys, soundfile, torch, numpy
from tqdm import tqdm

wav_file_path = "/publicdb/diarization/CSSD/WAVS/"
from sklearn.cluster import AgglomerativeClustering

sys.path.append("/home/liutaw/code/sd/ECAPA-TDNN")
from ECAPAModel import ECAPAModel
import numpy as np

train_set = set(line.split()[1] for line in open("../rttm_gt_train_nog0").readlines())
val_set = set(line.split()[1] for line in open("../rttm_gt_dev_nog0").readlines())
test_set = set(line.split()[1] for line in open("../rttm_gt_test_nog0").readlines())
eval_set = set(
    line.split()[1]
    for line in open("../test_data_extractor/rttm_final_eval").readlines()
)
print("train_set:", len(train_set))
print("val_set:", len(val_set))
print("test_set:", len(test_set))
print("eval_set:", len(eval_set))
from pyannote.core import SlidingWindow
import sys
import numpy as np
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster._kmeans import k_means
from spectralcluster import SpectralClusterer
from scipy.io import wavfile as wav


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




# 定义好了提取 spk emb 的模型
ecapa_lists = []
dict_ = dict()
dict_["ecapa_model"] = (
    "/home/liutaw/code/sd/ECAPA-TDNN/exps/voxceleb_cnceleb_all/model/model_0186.model"
)
dict_["n_class"] = 10365
dict_["name"] = "10365"
ecapa_lists.append(dict_)


for chunk_size in [
    2,
    3,
]:  # 切分的时候按照多少秒去切，实测，越短效果越差，这里选取2秒或3秒，最终用的是3秒
    segment_duration = int(chunk_size * 16000)
    for step_size in [chunk_size]:
        step_duration = int(step_size * 16000)

        for current_set, current_rttm in zip(
            [val_set, eval_set],
            ["../rttm_gt_dev_nog0", "../test_data_extractor/ref_utf8.rttm"],
        ):  # 遍历val 还是 test，test 是后面比赛结束后提供的
            if "ref_utf8" in current_rttm:
                is_test = True
            else:
                is_test = False
            if not is_test:
                continue

            for vad_type in [
                "transfomer_vad",
                "oracle",
            ]:  # 当时比赛的vad系统，已经提前提取好了，默认是transfomer_vad

                skip_chunk_size_lists = [
                    0.93
                ]  # 多短的段落应该被舍弃，是一个超参数，也就是低于 0.93 秒的都要被舍弃，当时也搜了很多参数，0.93 最好
                for skip_chunk_size in skip_chunk_size_lists:
                    for cluster_type in [
                        "ahc_single",
                        "ahc_complete",
                        "ahc_average",
                        "sc",
                    ]:  # 聚类算法，最终用的是 sc

                        threshold_lists = [
                            0.9
                        ]  # 确定 vad 的阈值，大于 0.9 概率确定是说话（逐帧的特征）
                        for threshold in threshold_lists:
                            for dict_ in ecapa_lists:
                                if not is_test:
                                    saved_to_rttm_file_path = (
                                        "./rttms/"
                                        + current_rttm.split("/")[-1]
                                        + "#vad_%s" % (vad_type)
                                        + "#chunksize_%0.2f" % (chunk_size)
                                        + "#skip_%0.3f" % (skip_chunk_size)
                                        + "#vadthreshold_%0.2f" % (threshold)
                                        + "#sctype_%s" % (cluster_type)
                                        + "#step_duration%0.2f" % (step_size)
                                        + "#"
                                        + dict_["name"]
                                        + ".rttm"
                                    )
                                else:
                                    saved_to_rttm_file_path = (
                                        "./rttms/"
                                        + current_rttm.split("/")[-1]
                                        + "#vad_%s" % (vad_type)
                                        + "#chunksize_%0.2f" % (chunk_size)
                                        + "#skip_%0.3f" % (skip_chunk_size)
                                        + "#vadthreshold_%0.2f" % (threshold)
                                        + "#sctype_%s" % (cluster_type)
                                        + "#step_duration%0.2f" % (step_size)
                                        + "#"
                                        + dict_["name"]
                                        + ".rttm"
                                    )

                                print(saved_to_rttm_file_path)
                                if os.path.exists(saved_to_rttm_file_path):
                                    print("continue")
                                    continue
                                # 这里是另一个模型已经提取好的 vad
                                vad_file_path = "./vad/use.wavlist.%s" % (
                                    current_rttm.split("/")[-1]
                                )
                                vadmlf_file_path = "./vad/vad.mlf.%s" % (
                                    current_rttm.split("/")[-1]
                                )

                                vad_dict = dict()
                                if vad_type == "oracle":
                                    from pyannote.core import (
                                        Segment,
                                        Timeline,
                                        Annotation,
                                    )

                                    rttm_dict = get_obj_from_rttm(current_rttm)

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
                                    if not is_test:
                                        vad_pred_files = (
                                            "/privatedb/codes/vad/voice-activity-detection/results/cssd_preficts_trainval_%0.2f/"
                                            % (threshold)
                                        )
                                        #                                         vad_pred_files = "/home/liutaw/code/vad/voice-activity-detection/results/cssd_preficts/"
                                        for filename in current_set:
                                            vad_dict[filename] = []
                                            json_file = (
                                                vad_pred_files + filename + ".json"
                                            )
                                            pred_json_obj = json.load(open(json_file))
                                            for item in pred_json_obj["activities"]:
                                                vad_dict[filename].append(
                                                    (
                                                        item["start"]["total_second"],
                                                        item["end"]["total_second"],
                                                    )
                                                )
                                    else:
                                        vad_pred_files = (
                                            "/privatedb/codes/vad/voice-activity-detection/results/cssd_preficts_test_%0.2f/"
                                            % (threshold)
                                        )
                                        for filename in current_set:
                                            vad_dict[filename] = []
                                            json_file = (
                                                vad_pred_files + filename + ".json"
                                            )
                                            pred_json_obj = json.load(open(json_file))
                                            for item in pred_json_obj["activities"]:
                                                vad_dict[filename].append(
                                                    (
                                                        item["start"]["total_second"],
                                                        item["end"]["total_second"],
                                                    )
                                                )

                                ecapa_model = dict_["ecapa_model"]
                                s = ECAPAModel(
                                    lr=0.001,
                                    lr_decay=0.97,
                                    C=1024,
                                    n_class=dict_["n_class"],
                                    m=0.2,
                                    s=30,
                                    test_step=1,
                                )
                                s.load_parameters(ecapa_model)
                                emb_dict = dict()
                                vad_seg_dict = dict()
                                for wav_file in tqdm(vad_dict.keys()):
                                    wav_path = "%s%s.wav" % (wav_file_path, wav_file)
                                    assert os.path.exists(wav_path)
                                    audio, _ = soundfile.read(wav_path)

                                    if wav_file not in emb_dict.keys():
                                        emb_dict[wav_file] = []
                                        vad_seg_dict[wav_file] = []

                                    for start_seconds, end_seconds in vad_dict[
                                        wav_file
                                    ]:

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

                                            data = torch.FloatTensor(
                                                numpy.stack([audio_obj], axis=0)
                                            ).cuda()
                                            embedding = s.predict_seg(data)
                                            emb_dict[wav_file].append(embedding)

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
                                    elif cluster_type == "wq":
                                        print("wq")
                                        labels = clusterer.predict(np.array(emb))
                                        assert len(vad_seg_dict[k]) == len(labels)
                                        clustering_id[k] = labels
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
