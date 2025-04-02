from torch import nn
import math


class MultiHeadAttentionBlock(nn.Module):
    super().__init__()

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv

        self.w_o = nn.Linear(d_model, d_model)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (Batch, h, Seq_Len, d_k) --> (Batch, h, Seq_Len, Seq_Len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        key = self.w_k(k)  # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        value = self.w_v(v)  # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)

        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, h, d_k) --> (Batch, h, Seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = query.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, Seq_Len, d_k) --> (Batch, Seq_Len, h, d_K) --> (Batch, Seq_Len, d_model)
        x = x.transpose(1, 2).contiguous().view(-1, self.d_h * self.d_k)

        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        return self.w_o(x)