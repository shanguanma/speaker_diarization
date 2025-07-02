#!/usr/bin/env python3
"""
训练分析脚本 - 用于监控SSND训练过程中的关键指标
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_training_log(log_file):
    """解析训练日志文件"""
    train_data = []
    valid_data = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # 解析训练日志
            if '[Train]' in line and 'loss=' in line:
                # 提取loss信息
                loss_match = re.search(r"'loss': ([0-9.]+)", line)
                bce_match = re.search(r"'bce_loss': ([0-9.]+)", line)
                arcface_match = re.search(r"'arcface_loss': ([0-9.]+)", line)
                der_match = re.search(r"'DER': ([0-9.]+)", line)
                acc_match = re.search(r"'ACC': ([0-9.]+)", line)
                
                if all([loss_match, bce_match, arcface_match, der_match, acc_match]):
                    train_data.append({
                        'loss': float(loss_match.group(1)),
                        'bce_loss': float(bce_match.group(1)),
                        'arcface_loss': float(arcface_match.group(1)),
                        'DER': float(der_match.group(1)),
                        'ACC': float(acc_match.group(1))
                    })
            
            # 解析验证日志
            elif '[Eval]' in line and 'validation:' in line:
                # 提取验证信息
                loss_match = re.search(r'loss=([0-9.]+)', line)
                bce_match = re.search(r'bce_loss=([0-9.]+)', line)
                arcface_match = re.search(r'arcface_loss=([0-9.]+)', line)
                der_match = re.search(r'DER=([0-9.]+)', line)
                acc_match = re.search(r'ACC=([0-9.]+)', line)
                
                if all([loss_match, bce_match, arcface_match, der_match, acc_match]):
                    valid_data.append({
                        'loss': float(loss_match.group(1)),
                        'bce_loss': float(bce_match.group(1)),
                        'arcface_loss': float(arcface_match.group(1)),
                        'DER': float(der_match.group(1)),
                        'ACC': float(acc_match.group(1))
                    })
    
    return train_data, valid_data

def plot_training_curves(train_data, valid_data, save_path=None):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 准备数据
    train_steps = list(range(len(train_data)))
    valid_steps = list(range(len(valid_data)))
    
    # 1. 总Loss
    axes[0, 0].plot(train_steps, [d['loss'] for d in train_data], label='Train', color='blue')
    if valid_data:
        axes[0, 0].plot(valid_steps, [d['loss'] for d in valid_data], label='Valid', color='red')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. BCE Loss
    axes[0, 1].plot(train_steps, [d['bce_loss'] for d in train_data], label='Train', color='blue')
    if valid_data:
        axes[0, 1].plot(valid_steps, [d['bce_loss'] for d in valid_data], label='Valid', color='red')
    axes[0, 1].set_title('BCE Loss')
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('BCE Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. ArcFace Loss
    axes[1, 0].plot(train_steps, [d['arcface_loss'] for d in train_data], label='Train', color='blue')
    if valid_data:
        axes[1, 0].plot(valid_steps, [d['arcface_loss'] for d in valid_data], label='Valid', color='red')
    axes[1, 0].set_title('ArcFace Loss')
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('ArcFace Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. DER
    axes[1, 1].plot(train_steps, [d['DER'] for d in train_data], label='Train', color='blue')
    if valid_data:
        axes[1, 1].plot(valid_steps, [d['DER'] for d in valid_data], label='Valid', color='red')
    axes[1, 1].set_title('DER')
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('DER')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    
    plt.show()

def analyze_training_issues(train_data, valid_data):
    """分析训练问题"""
    print("=== 训练问题分析 ===")
    
    if not train_data:
        print("没有找到训练数据")
        return
    
    # 分析ArcFace Loss
    train_arcface = [d['arcface_loss'] for d in train_data]
    print(f"ArcFace Loss 统计:")
    print(f"  训练集 - 平均值: {np.mean(train_arcface):.3f}, 最大值: {np.max(train_arcface):.3f}, 最小值: {np.min(train_arcface):.3f}")
    
    if valid_data:
        valid_arcface = [d['arcface_loss'] for d in valid_data]
        print(f"  验证集 - 平均值: {np.mean(valid_arcface):.3f}, 最大值: {np.max(valid_arcface):.3f}, 最小值: {np.min(valid_arcface):.3f}")
        
        # 检查过拟合
        if np.mean(valid_arcface) > np.mean(train_arcface) * 1.2:
            print("  ⚠️  警告: ArcFace Loss 可能存在过拟合")
    
    # 分析DER
    train_der = [d['DER'] for d in train_data]
    print(f"DER 统计:")
    print(f"  训练集 - 平均值: {np.mean(train_der):.3f}, 最小值: {np.min(train_der):.3f}")
    
    if valid_data:
        valid_der = [d['DER'] for d in valid_data]
        print(f"  验证集 - 平均值: {np.mean(valid_der):.3f}, 最小值: {np.min(valid_der):.3f}")
        
        if np.mean(valid_der) > 0.05:
            print("  ⚠️  警告: DER 过高，模型性能需要改进")
    
    # 分析Loss趋势
    if len(train_data) > 10:
        recent_train_loss = [d['loss'] for d in train_data[-10:]]
        early_train_loss = [d['loss'] for d in train_data[:10]]
        
        if np.mean(recent_train_loss) > np.mean(early_train_loss) * 0.9:
            print("  ✅ Loss 在下降，训练正常")
        else:
            print("  ⚠️  警告: Loss 下降不明显，可能需要调整学习率")

def main():
    # 使用示例
    log_file = "path/to/your/training.log"  # 替换为实际的日志文件路径
    
    if not Path(log_file).exists():
        print(f"日志文件不存在: {log_file}")
        print("请修改脚本中的 log_file 路径")
        return
    
    # 解析日志
    train_data, valid_data = parse_training_log(log_file)
    
    print(f"解析到 {len(train_data)} 条训练记录, {len(valid_data)} 条验证记录")
    
    # 分析问题
    analyze_training_issues(train_data, valid_data)
    
    # 绘制曲线
    plot_training_curves(train_data, valid_data, "training_analysis.png")

if __name__ == "__main__":
    main() 