# SSND模型训练改进措施

## 问题分析

根据训练日志分析，发现以下关键问题：

1. **DER指标异常高且不下降**：训练集和验证集的DER都在0.92-0.95之间
2. **Loss权重不平衡**：ArcFace loss（12-14）远大于BCE loss（0.1-0.2）
3. **可学习权重问题**：BCE loss权重被过度抑制
4. **模型架构问题**：DetectionDecoder输入全为0，影响学习效果

## 改进措施

### 1. 调整Loss权重初始化

**修改前：**
```python
self.log_s_bce = nn.Parameter(torch.tensor(0.0))  # exp(-0.0) = 1.0
self.log_s_arcface = nn.Parameter(torch.tensor(2.3026))  # exp(-2.3026) = 0.1
```

**修改后：**
```python
self.log_s_bce = nn.Parameter(torch.tensor(0.0))  # exp(-0.0) = 1.0, BCE weight = 1.0
self.log_s_arcface = nn.Parameter(torch.tensor(4.6052))  # exp(-4.6052) = 0.01, ArcFace weight = 0.01
```

**效果：** 大幅降低ArcFace loss权重，让BCE loss主导训练

### 2. 使用可学习的Query Embedding

**修改前：**
```python
x_det_dec = torch.zeros((B,N,self.d_model),device=device)
x_rep_dec = torch.zeros((B,N,vad_labels.shape[2]),device=device)
```

**修改后：**
```python
# 添加可学习参数
self.det_query_emb = nn.Parameter(torch.randn(max_speakers, d_model))
self.rep_query_emb = nn.Parameter(torch.randn(max_speakers, vad_out_len))

# 使用可学习embedding
x_det_dec = self.det_query_emb.unsqueeze(0).expand(B, N, self.d_model)
x_rep_dec = self.rep_query_emb.unsqueeze(0).expand(B, N, vad_labels.shape[2])
```

**效果：** 提高模型表达能力，让decoder有更好的学习起点

### 3. 引入Focal Loss处理类别不平衡

**新增方法：**
```python
def focal_bce_loss(self, logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal loss for binary classification to handle class imbalance.
    """
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - pt) ** gamma
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    alpha_weight = alpha * targets + (1 - alpha) * (1 - targets)
    focal_loss = alpha_weight * focal_weight * bce_loss
    return focal_loss.mean()
```

**效果：** 更好地处理VAD预测中的类别不平衡问题

### 4. 改进训练超参数

**新的训练脚本 `run_ssnd_aistation_improved.sh`：**
- 降低学习率：`--lr 5e-5`（从1e-4）
- 减小batch size：`--batch-size 32`（从64）
- 启用梯度裁剪：`--grad-clip true`
- 调整ArcFace参数：`--arcface-margin 0.1 --arcface-scale 16.0`
- 降低mask概率：`--mask-prob 0.3`（从0.5）

### 5. 增强调试信息

在`compute_loss`函数中添加详细诊断信息：
- 标签统计信息
- 预测概率分布
- 有效说话人数
- Loss权重值

## 预期效果

1. **DER下降**：通过调整loss权重和引入focal loss，预期DER从0.92+降至0.5以下
2. **训练稳定性**：可学习query embedding提供更好的学习起点
3. **收敛速度**：更合适的超参数设置加快收敛
4. **泛化能力**：focal loss提高对困难样本的学习能力

## 使用方法

1. 使用改进的模型代码
2. 运行改进的训练脚本：
```bash
bash run_ssnd_aistation_improved.sh
```

3. 监控训练日志中的DER变化
4. 如果DER仍然不下降，可以进一步调整：
   - 进一步降低ArcFace loss权重
   - 调整focal loss的alpha和gamma参数
   - 尝试不同的学习率调度策略 