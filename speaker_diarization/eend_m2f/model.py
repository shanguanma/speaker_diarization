#!/usr/bin/env python3
import logging
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from eend.eend.pytorch_backend.eend_m2f.mask2former_matcher import (
    HungarianMatcher as mask2formerHungarianMatcher,
)
from eend.eend.pytorch_backend.eend_m2f.fastinst_matcher import (
    HungarianMatcher as fastHungarianMatcher,
)
from eend.eend.pytorch_backend.eend_m2f.criterion import SetCriterion


class EendM2F(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pixel_decoder: nn.Module,
        transformer_decoder: nn.Module,
        # criterion: nn.Module,
        num_queries: int = 50,
        deep_supervision: bool = True,
        no_object_weight: float = 0.1,
        class_weight: float = 2.0,
        mask_weight: float = 5.0,
        dice_weight: float = 5.0,
        location_weight: float = 1000.0,
        proposal_weight: float = 20.0,
        train_num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
    ):
        super(EendM2F, self).__init__()

        self.backbone = backbone
        self.pixel_decoder = pixel_decoder
        self.transformer_decoder = transformer_decoder

        self.num_queries = num_queries

        ## loss args
        self.deep_supervision = deep_supervision
        self.no_object_weight = no_object_weight
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.location_weight = location_weight
        self.proposal_weight = proposal_weight

        self.train_num_points = train_num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        if (
            self.transformer_decoder.transformer_decoder_name
            == "OneScaleMaskedTransformerDecoder"
        ):
            # building criterion
            matcher = mask2formerHungarianMatcher(
                cost_class=self.class_weight,
                cost_mask=self.mask_weight,
                cost_dice=self.dice_weight,
                num_points=self.train_num_points,
            )
        elif self.transformer_decoder.transformer_decoder_name == "FastInstDecoder":
            # building criterion
            matcher = fastHungarianMatcher(
                cost_class=self.class_weight,
                cost_mask=self.mask_weight,
                cost_dice=self.dice_weight,
                cost_location=self.location_weight,
                num_points=self.train_num_points,
            )

        weight_dict = {
            "loss_ce": self.class_weight,
            "loss_mask": self.mask_weight,
            "loss_dice": self.dice_weight,
        }

        if self.deep_supervision:
            dec_layers = self.transformer_decoder.dec_layer
            aux_weight_dict = {}
            for i in range(2 * dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        weight_dict.update({"loss_proposal": self.proposal_weight})

        losses = ["labels", "masks"]
        self.criterion = SetCriterion(
            self.transformer_decoder.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=self.no_object_weight,
            losses=losses,
            num_points=self.train_num_points,
            oversample_ratio=self.oversample_ratio,
            importance_sample_ratio=self.importance_sample_ratio,
        )

        self._logger = logging.getLogger(__name__)

    def forward_model(self, src: List[Tensor]):

        device = src[0].device
        # self._logger.warn(f"src[0].device: {src[0].device} in forward_embedding function")
        src_ilens = [x.shape[0] for x in src]  # [utt1_T,utt2_T,...], len(ilens) = B
        src_ilens = torch.tensor(src_ilens, dtype=torch.int32, device=device)
        src = nn.utils.rnn.pad_sequence(
            src, padding_value=-1, batch_first=True
        )  # (B,T,C)
        src = src.to(device)  # (B,T,C)

        # batch_input['feat'] shape: (B,T,D), D: feature dimension, T: number of frames, B: batch size
        # batch_input['feat_length'] shape:(B,)
        # batch_input["target"]["masks"] shape: (B,T,S) ,S: num_speakers
        # batch_input["target"]['label'] shape: (B,)

        # speech feature pass into backbone network
        features = self.backbone(src, src_ilens)  # BxTxD
        # self._logger.warn(f"self.backbone input shape: {src.shape}")
        # self._logger.warn(f"self.backbone output shape: {features.shape}")

        # self._logger.warn(f"self.pixel_decoder input shape: {features.shape}")#  torch.Size([2,256, 500, 1]), feature shape: torch.Size([2, 256, 500, 1])
        mask_feature, feature = self.pixel_decoder(
            features
        )  # mask_feature shape: (B,D,T',1), feature shape:(B,D,T',1)
        # self._logger.warn(f"self.pixel_decoder output shape: mask_feature shape: {mask_feature.shape}, feature shape: {feature[-1].shape}")
        outputs = self.transformer_decoder(feature, mask_feature)
        # self._logger.warn(f"self.transformer_decoder output['pred_logits'] shape:{outputs['pred_logits'].shape} ")
        # self._logger.warn(f"self.transformer_decoder output['pred_masks'] shape:{outputs['pred_masks'].shape} ")

        # outputs["pred_logits"] shape:#(batch_size,num_queries,num_class+1)
        # outputs['pred_masks'] shape:#(batch_size, num_queries,T,1)
        return outputs

    def forward(
        self, src: List[Tensor], target_mask: List[Tensor], target_label: List[Tensor]
    ):
        outputs = self.forward_model(src)

        targets = self.prepared_target(
            target_mask=target_mask, target_label=target_label
        )

        # bipartite matching-based loss
        losses = self.criterion(outputs, targets)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        loss = sum(losses.values())  # for backward()
        errors = self.calc_diarization_error(outputs, target_mask)
        #self._logger.warn(f"loss: {loss}, its type: {type(loss)}")
        #self._logger.warn(f"errors: {errors}")
        losses_report = {}
        for k in list(losses.keys()):
            losses_report[k] = losses[k].detach().item()
        for k in list(errors.keys()):
            losses_report[k] = errors[k]
        stats = dict(loss=loss.detach().item(), details=losses_report)
        return loss, stats

    def prepared_target(self, target_mask: List[Tensor], target_label: List[Tensor]):
        """

        Generated list of dicts, such that len(targets) == batch_size.
        The expected keys in each dict depends on the losses applied, see each loss' doc
        for meeting input of `SetCriterion` class.
        Args:
            target_mask: list, its element shape:(T,num_segments_included_speaker)
            target_label: list, its element shape: (num_segments_included_speaker)
        Return: List, every element is dict, the dict two keys, first key is labels, shape: (num_segments_included_speaker,T,1)
                second key is masks, shape: (num_segments_included_speaker) # it is used to comput class loss
        """
        new_targets = []
        for i, targets_per_speech in enumerate(
            target_mask
        ):  # len(targets)== batch_size
            new_targets.append(
                {"labels": target_label[i], "masks": targets_per_speech.T.unsqueeze(-1)}
            )
        return new_targets

    def infer(self,src: List[Tensor],threshold_discard=0.8):

        # reference from EEND infer processing
        outputs = self.forward_model(src)
        # outputs["pred_logits"] shape:#(batch_size, num_queries,num_class+1)
        # outputs['pred_masks'] shape:#(batch_size, num_queries,T,1)
        logits = [torch.nn.functional.sigmoid(p.squeeze(-1)) for p in outputs['pred_masks']]
        #pprobs = [torch.nn.functional.sigmoid(p) for p in outputs['pred_logits']]
        ys_active = []
        for y in logits: # for loop B axis
            if threshold_discard is not None:
               # 保留概率大于或等于threshold_discard的预测
               retained_predictions = y >= threshold_discard #(num_queries,T), bool
               ys_active.append(retained_predictions.T)
            else:
                NotImplementedError(
                    'infer_num_speakers or attractor_threshold has to be given.')
        return ys_active # [(T,n_spk)], because I will assume batch size=1.
        
    def infer2(self,src: List[Tensor],threshold_discard=0.8,max_infer_num_speaker=2):
        """
        处理EEND-M2F模型的输出，根据给定的阈值过滤并确定说话人活动。

        :param pred_logits: 模型的概率输出，形状为(batch_size, num_queries, num_class+1)
        :param pred_mask: 模型的掩码输出，形状为(batch_size, num_queries, sequence_length)
        :param pi_threshold: 用于过滤说话人的概率阈值
        :param activity_threshold: 用于确定说话人活动的阈值
        :return: 过滤并阈值化后的掩码输出，形状为(batch_size, num_queries, sequence_length)
        """

        num_speakers = max_infer_num_speaker
        outputs = self.forward_model(src)
        # outputs["pred_logits"] shape:#(batch_size, num_queries,num_class+1)
        # outputs['pred_masks'] shape:#(batch_size, num_queries,T,1)
        
        # case1: bad, der is about 98%
        #pred_mask =F.sigmoid(outputs['pred_masks'].squeeze(-1))
        #pred_logits =F.softmax(outputs["pred_logits"],dim=-1)
        

        # case2: good der is about 48.61% 
        pred_mask =F.sigmoid(outputs['pred_masks'].squeeze(-1))
        pred_logits =outputs["pred_logits"]

        # case3: good der is about 79.18%  
        #pred_mask =outputs['pred_masks'].squeeze(-1)
        #pred_logits =outputs["pred_logits"]

        # pred_logits 形状：(batch size, num_queries, num_class+1)
        # pred_mask 形状：(batch size, num_queries, sequence_length)

        batch_size, num_queries, num_classes_plus_one = pred_logits.shape
        _, _, sequence_length = pred_mask.shape

        outputs = []

        # Step 1: 应用 pi 阈值
        for i in range(batch_size):
            valid_queries_mask = pred_logits[i, :, -1] >= threshold_discard  # 获取所有有效的查询

            # Step 2: 使用 pred_mask 生成每一帧可能的说话人活动
            valid_masks = pred_mask[i][valid_queries_mask]  # 选择有效的查询的掩码

            if valid_masks.size == 0:
                # 如果没有有效的查询，返回全零矩阵
                frame_speaker_activity = np.zeros((sequence_length, num_speakers))
            else:
                # Step 3: 每帧最多选择两个说话人
                frame_speaker_activity = np.zeros((sequence_length, num_speakers))

                for t in range(sequence_length):
                    # 获取每一帧的所有有效查询的值
                    frame_values = valid_masks[:, t]

                    # 选择概率最高的两个说话人
                    top_speakers = np.argsort(frame_values)[-num_speakers:]  # 获取概率最高的两个索引
                    frame_speaker_activity[t, :len(top_speakers)] = frame_values[top_speakers]

            outputs.append(frame_speaker_activity)

        return outputs # [(T,n_spk)], because I will assume batch size=1. here (T,n_spk) is numpy array.
        

    @torch.no_grad()
    def calc_diarization_error(
        self, transformer_decoder_output: Dict[str, Tensor], target_mask: List[Tensor]
    ):
        """
        This algorithm comes from List1 of `EEND-M2F: Masked-attention mask transformers for speaker diarization`
        """
        from scipy.optimize import linear_sum_assignment

        pred_mask = transformer_decoder_output["pred_masks"]  ##(B, num_queries,T,1)
        pred_mask = pred_mask.permute(0, 2, 1, 3).squeeze(
            -1
        )  # (B,T,num_queries), dtype: torch.float32
        bs = transformer_decoder_output["pred_masks"].size(0)

        batch_ms = []
        batch_fa = []
        batch_se = []
        batch_der = []
        # Iterate through batch size
        for b in range(bs):
            ref = target_mask[b]  # (T, num_instances)
            ref_T = ref.size(0)
            sys = pred_mask[b][:ref_T, :]  # (T, num_queries)

            self._logger.warn(f"sys shape: {sys.shape}")
            self._logger.warn(f"ref shape: {ref.shape}")
            nsys = sys.sum(1)
            nref = ref.sum(1)
            correct = torch.logical_and(
                sys[:, :, None], ref[:, None, :]
            )  # (T,num_queries, num_speaker),type: torch.bool
            correct = correct.cpu().float()
            ncor_ = []
            # iterate throught frames size
            for i in range(correct.size(0)):
                matching = linear_sum_assignment(correct[i], maximize=True)
                cor = correct[i][
                    matching
                ]  # every time frame, cor shape: (num_speaker), dtype: torch.bool
                ncor_.append(cor)
            # self._logger.warn(f"every t, ncor_: {ncor_}")
            ncor = sum(sum(ncor_))  # sum of T and sum of num_speaker
            ms = torch.clamp(nref - nsys, 0).sum()
            fa = torch.clamp(nsys - nref, 0).sum()
            se = torch.min(nsys, nref).sum() - ncor
            de = torch.max(nsys, nref).sum() - ncor  #  == ms + fa + se
            z = nref.sum()
            if z == 0: # skip no speaker utterance
                continue

            self._logger.warn(f"z: {z}")
            ms = ms / z  # speaker miss
            fa = fa / z  # false alarms
            se = se / z  # speaker errors
            der = de / z
            batch_ms.append(ms.item())
            batch_fa.append(fa.item())
            batch_se.append(se.item())
            batch_der.append(der.item())

        return {
            "ms": sum(batch_ms),
            "fa": sum(batch_fa),
            "se": sum(batch_se),
            "der": sum(batch_der),
        }
