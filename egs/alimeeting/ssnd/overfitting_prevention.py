"""
过拟合预防建议和配置

针对600步快速收敛可能导致的过拟合问题，提供以下解决方案：
"""

import torch
import torch.nn as nn

def get_overfitting_prevention_config():
    """
    返回防止过拟合的配置建议
    """
    return {
        # 1. 学习率调整
        "learning_rate": {
            "current": 3.4e-5,
            "suggested": 1e-5,  # 降低学习率
            "warmup_steps": 1000,  # 增加warmup步数
            "decay_factor": 0.95,  # 每epoch衰减5%
        },
        
        # 2. 正则化
        "regularization": {
            "weight_decay": 1e-4,  # 增加权重衰减
            "dropout_rate": 0.2,   # 增加dropout
            "label_smoothing": 0.1,  # 标签平滑
        },
        
        # 3. 数据增强
        "data_augmentation": {
            "spec_augment": True,
            "time_masking": 0.1,
            "frequency_masking": 0.1,
            "noise_injection": 0.05,
        },
        
        # 4. 训练策略
        "training_strategy": {
            "batch_size": 32,  # 减小批次大小
            "gradient_clipping": 1.0,
            "early_stopping_patience": 10,
            "reduce_lr_patience": 5,
        },
        
        # 5. 模型调整
        "model_adjustment": {
            "reduce_model_capacity": True,
            "fewer_layers": 2,  # 减少层数
            "smaller_hidden_dim": 128,  # 减小隐藏维度
        }
    }

def apply_regularization_to_model(model, config):
    """
    对模型应用正则化
    """
    # 增加dropout
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = config["regularization"]["dropout_rate"]
    
    # 添加权重衰减到优化器
    # 在get_optimizer_scheduler函数中设置weight_decay
    
    return model

def create_early_stopping_callback(patience=10, min_delta=1e-4):
    """
    创建早停回调
    """
    class EarlyStopping:
        def __init__(self, patience=patience, min_delta=min_delta):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = float('inf')
            self.should_stop = False
        
        def __call__(self, val_loss):
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"Early stopping triggered after {self.patience} epochs without improvement")
            
            return self.should_stop
    
    return EarlyStopping()

def get_recommended_hyperparameters():
    """
    基于当前训练情况推荐的超参数
    """
    return {
        "optimizer": {
            "lr": 1e-5,  # 降低学习率
            "weight_decay": 1e-4,  # 增加权重衰减
            "betas": (0.9, 0.999),
        },
        "scheduler": {
            "warmup_steps": 1000,
            "decay_factor": 0.95,
            "min_lr": 1e-7,
        },
        "training": {
            "batch_size": 32,
            "gradient_clip": 1.0,
            "max_epochs": 50,
        },
        "model": {
            "dropout": 0.2,
            "num_layers": 2,  # 减少层数
            "hidden_dim": 128,  # 减小维度
        }
    }

def analyze_overfitting_risk(current_metrics):
    """
    分析过拟合风险
    """
    risk_factors = []
    
    # 检查收敛速度
    if current_metrics.get("steps", 0) < 1000:
        risk_factors.append("快速收敛 (< 1000步)")
    
    # 检查训练验证差距
    train_der = current_metrics.get("train_der", 0)
    valid_der = current_metrics.get("valid_der", 0)
    if train_der - valid_der > 0.1:
        risk_factors.append(f"训练验证DER差距过大 ({train_der:.4f} vs {valid_der:.4f})")
    
    # 检查损失下降模式
    if current_metrics.get("loss_decrease_rate", 0) > 0.1:
        risk_factors.append("损失下降过快")
    
    # 风险评估
    risk_level = "低"
    if len(risk_factors) >= 3:
        risk_level = "高"
    elif len(risk_factors) >= 2:
        risk_level = "中"
    
    return {
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "recommendations": get_recommendations_for_risk(risk_level)
    }

def get_recommendations_for_risk(risk_level):
    """
    根据风险等级给出建议
    """
    if risk_level == "高":
        return [
            "立即降低学习率到1e-5",
            "增加权重衰减到1e-3",
            "增加dropout到0.3",
            "减少模型容量",
            "增加数据增强",
            "实施早停机制"
        ]
    elif risk_level == "中":
        return [
            "降低学习率到5e-5",
            "增加权重衰减到1e-4",
            "增加dropout到0.2",
            "监控验证集性能"
        ]
    else:
        return [
            "继续监控训练过程",
            "保持当前配置",
            "定期检查验证集性能"
        ]

if __name__ == "__main__":
    # 示例使用
    config = get_overfitting_prevention_config()
    print("过拟合预防配置:")
    for category, settings in config.items():
        print(f"\n{category}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
    
    # 分析当前训练情况
    current_metrics = {
        "steps": 600,
        "train_der": 0.05,
        "valid_der": 0.076,
        "loss_decrease_rate": 0.15
    }
    
    risk_analysis = analyze_overfitting_risk(current_metrics)
    print(f"\n过拟合风险分析:")
    print(f"风险等级: {risk_analysis['risk_level']}")
    print(f"风险因素: {risk_analysis['risk_factors']}")
    print(f"建议: {risk_analysis['recommendations']}") 