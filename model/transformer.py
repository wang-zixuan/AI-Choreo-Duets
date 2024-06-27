import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionModule, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        # [seq_length, batch_size, embed_dim]
        attn_output, _ = self.multihead_attn(x, x, x)
        return attn_output


class CrossAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionModule, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, q1, k1, v1, q2, k2, v2):
        # q1, k1, v1 shape: [seq_length, batch_size, embed_dim]
        # q2, k2, v2 shape: [seq_length, batch_size, embed_dim]

        # Cross attention for query 1 with key2 and value2
        cross_attn_output1, _ = self.multihead_attn(q1, k2, v2)

        # Cross attention for query 2 with key1 and value1
        cross_attn_output2, _ = self.multihead_attn(q2, k1, v1)

        return cross_attn_output1, cross_attn_output2


class DanceTransformer(nn.Module):
    def __init__(self):
        super(DanceTransformer, self).__init__()
        self.preprocessing = nn.Conv1d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.self_attn_1 = SelfAttentionModule(3, 1)
        self.self_attn_2 = SelfAttentionModule(3, 1)
        self.cross_attention = CrossAttentionModule(3, 1)
        self.decoder1 = SelfAttentionModule(3, 1)
        self.decoder2 = SelfAttentionModule(3, 1)

    def forward(self, dancer1_data, dancer2_data):
        # [batch_size, 3, 29]
        dancer1_data = torch.tensor(dancer1_data, dtype=torch.float32).transpose(1, 2)
        dancer2_data = torch.tensor(dancer2_data, dtype=torch.float32).transpose(1, 2)

        dancer1_data = self.preprocessing(dancer1_data)
        dancer2_data = self.preprocessing(dancer2_data)

        # [29, batch_size, 3]
        dancer1_data = dancer1_data.permute(2, 0, 1)
        dancer2_data = dancer2_data.permute(2, 0, 1)

        q1 = self.self_attn_1(dancer1_data)
        q2 = self.self_attn_2(dancer2_data)

        cross_attn_output1, cross_attn_output2 = self.cross_attention(q1, q1, q1, q2, q2, q2)

        output1 = self.decoder1(cross_attn_output1)
        output2 = self.decoder2(cross_attn_output2)

        # Transpose back to original shape [batch_size, 29, 3]
        output1 = output1.permute(1, 0, 2)
        output2 = output2.permute(1, 0, 2)

        return output1, output2
