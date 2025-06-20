import torch
import numpy as np
from ssnd_model import SSNDModel  # 假设模型结构已实现
from feature import extract_logmel

# 假设有模型权重、音频分块、伪embedding等加载方法

def W(y):
    """计算embedding权重：非重叠说话帧数之和"""
    # y: [T']，0/1
    return float((y > 0.5).sum())


def infer_ssnd(
    model: SSNDModel,
    audio_blocks: list,  # 每个block为原始音频数据
    e_pse: torch.Tensor,  # 伪说话人embedding [S]
    e_non: torch.Tensor,  # 非说话embedding [S]
    t1: float,  # 新说话人阈值
    t2: float,  # 已知说话人阈值
    lc: int,    # 当前chunk帧数
    lr: int,    # 右context帧数
    N: int,     # 最大说话人数
    S: int,     # embedding维度
    device: str = 'cuda',
):
    model.eval()
    model.to(device)
    dia_result = {}  # {spk_id: [vad序列]}
    emb_buffer = {}  # {spk_id: [(embedding, weight), ...]}
    num_frames = 0
    spk_id_counter = 1

    for audio_block in audio_blocks:
        # 1. 组装输入embedding
        emb_list = [e_pse.to(device)]
        spk_list = [spk_id_counter]  # 新说话人id预分配
        for spk_id in emb_buffer.keys():
            e_sum = torch.zeros(S, device=device)
            w_sum = 0.0
            for e_i, w_i in emb_buffer[spk_id]:
                e_sum += w_i * e_i
                w_sum += w_i
            emb_list.append(e_sum / (w_sum + 1e-8))
            spk_list.append(spk_id)
        while len(emb_list) < N:
            emb_list.append(e_non.to(device))
        emb_tensor = torch.stack(emb_list)  # [N, S]

        # 2. 特征提取
        feats = extract_logmel(audio_block)  # [T, F]
        feats = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, F]
        emb_tensor = emb_tensor.unsqueeze(0)  # [1, N, S]

        # 3. 前向推理
        with torch.no_grad():
            vad_pred, emb_pred = model.infer(feats, emb_tensor)  # vad_pred: [1, N, T'] emb_pred: [1, N, S]
        vad_pred = vad_pred[0]  # [N, T']
        emb_pred = emb_pred[0]  # [N, S]

        # 4. 处理pseudo-speaker
        y_pse = vad_pred[0]  # [T']
        e_pse_pred = emb_pred[0]  # [S]
        w_pse = W(y_pse)
        if w_pse > t1:
            elapsed_y = torch.zeros(num_frames, device=device)
            current_y = y_pse[-(lc+lr):-lr] if lr > 0 else y_pse[-lc:]
            new_id = spk_id_counter
            dia_result[new_id] = torch.cat([elapsed_y, current_y.cpu()])
            emb_buffer[new_id] = [(e_pse_pred.cpu(), w_pse)]
            spk_id_counter += 1

        # 5. 处理已知说话人
        for n in range(1, len(spk_list)):
            y_n = vad_pred[n]
            e_n = emb_pred[n]
            w_n = W(y_n)
            spk_id = spk_list[n]
            if spk_id in dia_result:
                dia_result[spk_id] = torch.cat([dia_result[spk_id], y_n[-(lc+lr):-lr].cpu() if lr > 0 else y_n[-lc:].cpu()])
            else:
                dia_result[spk_id] = y_n[-(lc+lr):-lr].cpu() if lr > 0 else y_n[-lc:].cpu()
            if w_n > t2:
                if spk_id not in emb_buffer:
                    emb_buffer[spk_id] = []
                emb_buffer[spk_id].append((e_n.cpu(), w_n))
        num_frames += lc

    # 输出结果整理
    for spk_id in dia_result:
        dia_result[spk_id] = dia_result[spk_id].numpy()
    return dia_result, emb_buffer

# 用法示例（需补充模型加载、音频分块、embedding初始化等）
if __name__ == '__main__':
    # 加载模型、权重、e_pse, e_non等
    # audio_blocks = ...
    # model = SSNDModel(...)
    # model.load_state_dict(torch.load('ssnd.pth'))
    # e_pse = torch.randn(256)
    # e_non = torch.zeros(256)
    # dia_result, emb_buffer = infer_ssnd(model, audio_blocks, e_pse, e_non, t1=20, t2=10, lc=48, lr=16, N=30, S=256)
    pass 