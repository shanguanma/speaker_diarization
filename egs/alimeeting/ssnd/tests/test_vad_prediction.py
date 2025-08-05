#!/usr/bin/env python3
"""
测试VAD预测分布的脚本
"""

import torch
import torch.nn.functional as F
import numpy as np

def test_vad_prediction_distribution():
    """测试VAD预测的分布"""
    
    # 模拟不同的bias值对预测的影响
    print("=== 测试不同bias值对VAD预测的影响 ===")
    
    # 模拟DetectionDecoder的输出logits
    batch_size, num_speakers, seq_len = 1, 30, 200
    
    # 测试不同的bias值
    bias_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    for bias in bias_values:
        # 模拟logits（假设模型输出接近0）
        logits = torch.randn(batch_size, num_speakers, seq_len) * 0.1  # 小的随机值
        logits = logits + bias  # 添加bias
        
        # 计算概率
        probs = torch.sigmoid(logits)
        
        # 计算正样本预测比例
        positive_preds = (probs > 0.5).float().mean()
        
        print(f"Bias={bias:.1f}: 正样本预测比例={positive_preds.item():.4f}")
    
    print("\n=== 测试不同pos_weight对BCE loss的影响 ===")
    
    # 模拟标签（假设10%是正样本）
    labels = torch.zeros(batch_size, num_speakers, seq_len)
    positive_mask = torch.rand(batch_size, num_speakers, seq_len) < 0.1
    labels[positive_mask] = 1.0
    
    # 模拟预测（假设模型倾向于不检测）
    logits = torch.randn(batch_size, num_speakers, seq_len) * 0.1 - 0.5  # 偏向负值
    
    # 测试不同的pos_weight
    pos_weights = [1.0, 5.0, 10.0, 15.0, 20.0]
    
    for pos_weight in pos_weights:
        # 计算BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, labels, 
            pos_weight=torch.tensor([pos_weight])
        )
        
        # 计算预测概率
        probs = torch.sigmoid(logits)
        positive_preds = (probs > 0.5).float().mean()
        
        print(f"pos_weight={pos_weight}: BCE_loss={bce_loss.item():.4f}, 正样本预测比例={positive_preds.item():.4f}")
    
    print("\n=== 建议的修复策略 ===")
    print("1. 增加pos_weight到15.0-20.0来减少漏检")
    print("2. 修改DetectionDecoder的bias从-1.0到0.5")
    print("3. 增加ArcFace loss权重到0.5")
    print("4. 减少正则化惩罚")

if __name__ == "__main__":
    test_vad_prediction_distribution() 