# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 09:28:40 2024

@author: Olivia
"""

import torch
from torch import nn
import numpy as np
from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout=0.1, max_len=204):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = torch.zeros(max_len, embedding_size)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-np.log(10000.0) / embedding_size))

        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, 0:x.shape[1], :].to(x.device)
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, embedding_size, heads_num, dropout):
        super().__init__()
        self.heads_num = heads_num
        self.scale = 1.0 / np.sqrt(embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size, bias=False)
        self.value = nn.Linear(embedding_size, embedding_size, bias=False)
        self.query = nn.Linear(embedding_size, embedding_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(embedding_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.heads_num, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.heads_num, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.heads_num, -1).transpose(1, 2)

        attn = torch.matmul(q, k) * self.scale
        attn = nn.functional.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, -1)
        out = self.to_out(out)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, embedding_size, heads_num, fnn_num, dropout):
        super().__init__()
        self.LayerNorm1 = nn.LayerNorm(embedding_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(embedding_size, eps=1e-5)
        self.attention_layer = Attention(embedding_size, heads_num, dropout)
        self.FeedForward = nn.Sequential(
            nn.Linear(embedding_size, fnn_num),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fnn_num, embedding_size),
            nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attention_layer(x)
        x = self.LayerNorm1(x)
        x = x + self.FeedForward(x)
        x = self.LayerNorm2(x)
        return x


class AstronomyTransformer(nn.Module):
    def __init__(self, input_shape, embedding_size, heads_num, fnn_num, 
                 num_classes, dropout):
        super().__init__()
        channel_num, seq_len = input_shape

        self.position_encode = PositionalEncoding(embedding_size)

        self.embed_layer = nn.Sequential(
            nn.Linear(channel_num, embedding_size),
            nn.LayerNorm(embedding_size, eps=1e-5)
        )

        self.EncoderBlock1 = EncoderBlock(embedding_size, heads_num, fnn_num, dropout)
        self.EncoderBlock2 = EncoderBlock(embedding_size, heads_num, fnn_num, dropout)
        self.EncoderBlock3 = EncoderBlock(embedding_size, heads_num, fnn_num, dropout)
        self.EncoderBlock4 = EncoderBlock(embedding_size, heads_num, fnn_num, dropout)
        self.EncoderBlock5 = EncoderBlock(embedding_size, heads_num, fnn_num, dropout)
        self.EncoderBlock6 = EncoderBlock(embedding_size, heads_num, fnn_num, dropout)
        # self.EncoderBlock7 = EncoderBlock(embedding_size, heads_num, fnn_num, dropout)
        # self.EncoderBlock8 = EncoderBlock(embedding_size, heads_num, fnn_num, dropout)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        x = self.embed_layer(x.permute(0, 2, 1))
        x = self.position_encode(x)
        
        x = self.EncoderBlock1(x)
        x = self.EncoderBlock2(x)
        x = self.EncoderBlock3(x)
        x = self.EncoderBlock4(x)
        x = self.EncoderBlock5(x)
        x = self.EncoderBlock6(x)
        # x = self.EncoderBlock7(x)
        # x = self.EncoderBlock8(x)
        
        x = x.permute(0, 2, 1)
        x = self.gap(x)
        x = self.flatten(x)
        out = self.out(x)
        return out
