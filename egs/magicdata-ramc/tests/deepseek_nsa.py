#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenCompression(nn.Module):
    def __init__(self, block_length, sliding_stride, in_dim, out_dim):
        super(TokenCompression, self).__init__()
        self.block_length = block_length
        self.sliding_stride = sliding_stride
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim, bias=False)
        )

    def forward(self, keys):
        batch_size, seq_length, _ = keys.size()
        num_blocks = (seq_length - self.block_length) // self.sliding_stride + 1
        compressed_keys = []
        for i in range(num_blocks):
            start = i * self.sliding_stride
            end = start + self.block_length
            block = keys[:, start:end, :]
            # 这里可以添加位置编码，暂时省略
            compressed_block = self.mlp(block.mean(dim=1))
            compressed_keys.append(compressed_block)
        compressed_keys = torch.stack(compressed_keys, dim=1)
        return compressed_keys


class TokenSelection(nn.Module):
    def __init__(self, block_size, num_selected_blocks):
        super(TokenSelection, self).__init__()
        self.block_size = block_size
        self.num_selected_blocks = num_selected_blocks

    def forward(self, keys, compression_scores):
        batch_size, _, key_dim = keys.size()
        num_blocks = keys.size(1) // self.block_size
        block_scores = compression_scores.view(batch_size, num_blocks)
        top_blocks = torch.topk(block_scores, self.num_selected_blocks, dim=1).indices
        selected_keys = []
        for b in range(batch_size):
            block_indices = top_blocks[b]
            block_selected_keys = []
            for idx in block_indices:
                start = idx * self.block_size
                end = start + self.block_size
                block_selected_keys.append(keys[b, start:end, :])
            selected_keys.append(torch.cat(block_selected_keys, dim=0))
        selected_keys = torch.stack(selected_keys, dim=0)
        return selected_keys


class SlidingWindow(nn.Module):
    def __init__(self, window_size):
        super(SlidingWindow, self).__init__()
        self.window_size = window_size

    def forward(self, keys, values):
        return keys[:, -self.window_size:], values[:, -self.window_size:]


class NSA(nn.Module):
    def __init__(self, block_length, sliding_stride, block_size, num_selected_blocks, window_size, in_dim, out_dim):
        super(NSA, self).__init__()
        self.token_compression = TokenCompression(block_length, sliding_stride, in_dim, out_dim)
        self.token_selection = TokenSelection(block_size, num_selected_blocks)
        self.sliding_window = SlidingWindow(window_size)
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.ReLU(),
            nn.Linear(out_dim, 3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, query, keys, values):
        batch_size, _, _ = query.size()
        # 令牌压缩
        compressed_keys = self.token_compression(keys)
        print(f"compressed_keys shape: {compressed_keys.shape}")
        compressed_values = self.token_compression(values)
        compression_scores = F.softmax(torch.bmm(query, compressed_keys.transpose(1, 2)), dim=2)
        print(f"compression_scores shape: {compression_scores.shape}")
        # 令牌选择
        selected_keys = self.token_selection(keys, compression_scores)
        selected_values = self.token_selection(values, compression_scores)
        # 滑动窗口
        window_keys, window_values = self.sliding_window(keys, values)
        # 门控机制
        gate_scores = self.gate_mlp(query)
        gate_compression, gate_selection, gate_window = gate_scores.chunk(3, dim=2)
        # 计算注意力输出
        output_compression = F.softmax(torch.bmm(query, compressed_keys.transpose(1, 2)), dim=2) @ compressed_values
        output_selection = F.softmax(torch.bmm(query, selected_keys.transpose(1, 2)), dim=2) @ selected_values
        output_window = F.softmax(torch.bmm(query, window_keys.transpose(1, 2)), dim=2) @ window_values
        output = gate_compression * output_compression + gate_selection * output_selection + gate_window * output_window
        return output


# 示例使用
if __name__ == "__main__":
    batch_size = 2
    seq_length = 1024
    in_dim = 512
    out_dim = 512
    block_length = 32
    sliding_stride = 16
    block_size = 64
    num_selected_blocks = 16
    window_size = 512

    query = torch.randn(batch_size, 1, in_dim)
    keys = torch.randn(batch_size, seq_length, in_dim)
    values = torch.randn(batch_size, seq_length, in_dim)

    nsa = NSA(block_length, sliding_stride, block_size, num_selected_blocks, window_size, in_dim, out_dim)
    output = nsa(query, keys, values)
    print(output.size())
