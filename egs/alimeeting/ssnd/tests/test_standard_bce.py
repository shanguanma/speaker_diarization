#!/usr/bin/env python3
"""
测试标准BCE loss的功能
"""

import torch
import torch.nn.functional as F
import numpy as np

def test_standard_bce_loss():
    """
    测试标准BCE loss和focal loss的区别
    """
    # 创建测试数据
    batch_size, num_speakers, seq_len = 2, 4, 100
    
    # 创建logits（模型输出）
    logits = torch.randn(batch_size, num_speakers, seq_len)
    
    # 创建标签（真实值）
    labels = torch.randint(0, 2, (batch_size, num_speakers, seq_len)).float()
    
    print(f"Logits shape: {logits.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels mean: {labels.mean().item():.4f}")
    
    # 计算标准BCE loss
    standard_bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    standard_bce_mean = standard_bce.mean()
    
    # 计算focal loss
    probs = torch.sigmoid(logits)
    pt = probs * labels + (1 - probs) * (1 - labels)
    focal_weight = (1 - pt) ** 2.0  # gamma=2.0
    alpha_weight = 0.75 * labels + (1 - 0.75) * (1 - labels)  # alpha=0.75
    focal_loss = alpha_weight * focal_weight * standard_bce
    focal_loss_mean = focal_loss.mean()
    
    print(f"\n标准BCE loss: {standard_bce_mean.item():.4f}")
    print(f"Focal loss: {focal_loss_mean.item():.4f}")
    print(f"Focal/BCE ratio: {focal_loss_mean.item() / standard_bce_mean.item():.4f}")
    
    # 分析不同预测概率下的loss
    print(f"\n不同预测概率下的loss分析:")
    for prob_threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        prob_mask = (probs > prob_threshold) & (probs < prob_threshold + 0.1)
        if prob_mask.sum() > 0:
            bce_in_range = standard_bce[prob_mask].mean()
            focal_in_range = focal_loss[prob_mask].mean()
            print(f"预测概率 {prob_threshold:.1f}-{prob_threshold+0.1:.1f}: BCE={bce_in_range:.4f}, Focal={focal_in_range:.4f}")
    
    # 分析正负样本的loss
    positive_mask = labels == 1
    negative_mask = labels == 0
    
    if positive_mask.sum() > 0:
        bce_pos = standard_bce[positive_mask].mean()
        focal_pos = focal_loss[positive_mask].mean()
        print(f"\n正样本 (labels=1): BCE={bce_pos:.4f}, Focal={focal_pos:.4f}")
    
    if negative_mask.sum() > 0:
        bce_neg = standard_bce[negative_mask].mean()
        focal_neg = focal_loss[negative_mask].mean()
        print(f"负样本 (labels=0): BCE={bce_neg:.4f}, Focal={focal_neg:.4f}")

def test_loss_gradients():
    """
    测试不同loss的梯度特性
    """
    print("\n" + "="*50)
    print("测试loss梯度特性")
    print("="*50)
    
    # 创建需要梯度的logits
    logits = torch.randn(2, 4, 50, requires_grad=True)
    labels = torch.randint(0, 2, (2, 4, 50)).float()
    
    # 标准BCE loss
    logits_bce = logits.clone().detach().requires_grad_(True)
    bce_loss = F.binary_cross_entropy_with_logits(logits_bce, labels)
    bce_loss.backward()
    bce_grad_norm = logits_bce.grad.norm().item()
    
    # Focal loss
    logits_focal = logits.clone().detach().requires_grad_(True)
    probs = torch.sigmoid(logits_focal)
    pt = probs * labels + (1 - probs) * (1 - labels)
    focal_weight = (1 - pt) ** 2.0
    alpha_weight = 0.75 * labels + (1 - 0.75) * (1 - labels)
    focal_loss = (alpha_weight * focal_weight * F.binary_cross_entropy_with_logits(logits_focal, labels, reduction='none')).mean()
    focal_loss.backward()
    focal_grad_norm = logits_focal.grad.norm().item()
    
    print(f"标准BCE loss: {bce_loss.item():.4f}, 梯度范数: {bce_grad_norm:.4f}")
    print(f"Focal loss: {focal_loss.item():.4f}, 梯度范数: {focal_grad_norm:.4f}")
    print(f"Focal/BCE 梯度比: {focal_grad_norm / bce_grad_norm:.4f}")

def test_class_imbalance():
    """
    测试类别不平衡情况下的表现
    """
    print("\n" + "="*50)
    print("测试类别不平衡情况")
    print("="*50)
    
    # 创建高度不平衡的数据
    batch_size, num_speakers, seq_len = 2, 4, 1000
    
    # 只有10%的正样本
    labels = torch.zeros(batch_size, num_speakers, seq_len)
    num_positive = int(0.1 * labels.numel())
    positive_indices = torch.randperm(labels.numel())[:num_positive]
    labels.view(-1)[positive_indices] = 1
    
    print(f"正样本比例: {labels.mean().item():.4f}")
    
    # 创建不同的预测
    predictions = []
    
    # 1. 随机预测
    pred1 = torch.rand(batch_size, num_speakers, seq_len)
    predictions.append(("随机预测", pred1))
    
    # 2. 偏向负样本的预测
    pred2 = torch.rand(batch_size, num_speakers, seq_len) * 0.3
    predictions.append(("偏向负样本", pred2))
    
    # 3. 偏向正样本的预测
    pred3 = torch.rand(batch_size, num_speakers, seq_len) * 0.3 + 0.7
    predictions.append(("偏向正样本", pred3))
    
    for name, pred in predictions:
        logits = torch.log(pred / (1 - pred))  # 转换为logits
        
        # 标准BCE
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        # Focal loss
        probs = torch.sigmoid(logits)
        pt = probs * labels + (1 - probs) * (1 - labels)
        focal_weight = (1 - pt) ** 2.0
        alpha_weight = 0.75 * labels + (1 - 0.75) * (1 - labels)
        focal_loss = (alpha_weight * focal_weight * F.binary_cross_entropy_with_logits(logits, labels, reduction='none')).mean()
        
        print(f"{name}: BCE={bce_loss.item():.4f}, Focal={focal_loss.item():.4f}")

if __name__ == "__main__":
    test_standard_bce_loss()
    test_loss_gradients()
    test_class_imbalance()
    
    print("\n" + "="*50)
    print("总结:")
    print("1. 标准BCE loss对所有样本一视同仁")
    print("2. Focal loss对困难样本（预测错误的样本）给予更高权重")
    print("3. 在类别不平衡情况下，Focal loss可能更有效")
    print("4. 如果数据相对平衡，标准BCE loss可能更稳定")
    print("="*50) 