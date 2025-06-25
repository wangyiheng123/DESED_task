import torch
import torch.nn as nn

#自注意力机制的时序聚合网络
class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=3, window_size=1, qkv_bias=False, qk_scale=None, dropout=0., causal=True, device=None):
        super().__init__()
        assert dim % heads == 0, "dim {dim} should be divided by num_heads {heads}."

        # self.total_ops = torch.DoubleTensor([0])
        self.dim = dim
        self.num_heads = heads
        self.causal = causal
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        self.mask = torch.tril(torch.ones(window_size, window_size)).to(device)
        # self.register_buffer('mask', torch.tril(torch.ones(window_size, window_size)))

    def forward(self, x):
        B_prev, T_prev, C_prev = x.shape
        if self.window_size > 0:
            x = x.reshape(-1, self.window_size, C_prev)  # [B*T/window_size,window_size,ndim]
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  #[B*T/window_size, self.num_heads, window_size, C // self.num_heads]

        # merge key padding and attention masks
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [b, heads, T, T]
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if self.window_size > 0:  # reshape to the original size
            x = x.reshape(B_prev, T_prev, C_prev)
        return x

#门控网络
# class TemporalAttention(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_classes):
#         super().__init__()
#         self.linear1 = nn.Linear(input_dim,hidden_dim)
#         self.relu = nn.ReLU()
#         self.linear2 = nn.Linear(hidden_dim, 1)
#         self.sigmoid = nn.Sigmoid()
#         self.pooled = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.classifier = nn.Linear(192, num_classes)
#
#     def forward(self, x):
#         # x: (B, T, D)
#         att = self.linear1(x)
#         att = self.relu(att)
#         att = self.linear2(att)
#         gate_scores = self.sigmoid(att)
#         gated_x = x * gate_scores  # (B, T, D)
#         pooled = self.pooled(gated_x).mean(1)  # (B, D)
#
#         out = self.classifier(pooled)  # (B, num_classes)
#         return out
