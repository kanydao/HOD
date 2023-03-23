import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1, head_size=8):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.attn_size = attn_size = d_model // head_size
        self.scale = attn_size ** -0.5

        self.linear_q = nn.Linear(d_model, head_size * attn_size, bias=False)
        self.linear_k = nn.Linear(d_model, head_size * attn_size, bias=False)
        self.linear_v = nn.Linear(d_model, head_size * attn_size, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

        self.output_layer = nn.Linear(head_size * attn_size, d_model,
                                      bias=False)

    def forward(self, memory_p, memory):
        # d_model = c = #head * att
        # s_len = hw
        orig_q_size = memory_p.size()           # [b, s_len, d_model]
        b, hw, c = memory_p.shape               # [b, s_len, d_model]

        pe = Positonal_Encoding(d_model=c, max_len=hw).cuda()
        memory_p = pe(memory_p)                 # [b, s_len, d_model]
        memory = pe(memory)                     # [b, s_len, d_model]

        d_q = self.attn_size
        d_k = self.attn_size
        d_v = self.attn_size

        q = self.linear_q(memory_p).view(b, -1, self.head_size, d_q)   # [b, s_len, #head, attn]
        k = self.linear_k(memory).view(b, -1, self.head_size, d_k)   # [b, s_len, #head, attn]
        v = self.linear_v(memory).view(b, -1, self.head_size, d_v)   # [b, s_len, #head, attn]

        q = q.transpose(1, 2)                   # [b, #head, s_len, d_k]
        v = v.transpose(1, 2)                   # [b, #head, s_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)   # [b, #head, d_k, s_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q.mul_(self.scale)
        x = torch.matmul(q, k)                  # [b, #head, s_len, s_len]
        x = torch.softmax(x, dim=3)             # [b, #head, s_len, s_len]
        x = self.attn_dropout(x)                # [b, #head, s_len, s_len]
        x = x.matmul(v)                         # [b, #head, s_len, attn]

        x = x.transpose(1, 2).contiguous()      # [b, s_len, #head, attn]
        x = x.view(b, -1, self.head_size * d_v) # [b, s_len, #head * attn]

        x = self.output_layer(x)                # [b, s_len, d_model]

        assert x.size() == orig_q_size
        return x


class Positonal_Encoding(nn.Module):
    def __init__(self, d_model, max_len=20000, dropout=0.1):
        super(Positonal_Encoding, self).__init__()
        # 增强模型的鲁棒性，需要加一层dropout
        self.dropout = nn.Dropout(p=dropout)

        position_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (math.log(10000.0) / d_model))
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        position_encoding = position_encoding.unsqueeze(0)
        self.register_buffer('position_encoding', position_encoding) 
    
    def forward(self, img_feature):
        output = img_feature + Variable(self.position_encoding[:, :img_feature.shape[1]], requires_grad=False)
        return self.dropout(output)