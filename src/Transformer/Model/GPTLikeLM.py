import torch.nn as nn
from src.Transformer.Model.InputEmbeddings import InputEmbeddings
from src.Transformer.Model.PositionalEncoding import PositionalEncoding
from src.Transformer.Model.Decoder import Decoder
from src.Transformer.Model.DecoderBlock import DecoderBlock
from src.Transformer.Model.MultiHeadAttentionBlock import MultiHeadAttentionBlock
from src.Transformer.Model.FeedForwardBlock import FeedForwardBlock
from src.Transformer.Model.ProjectionLayer import ProjectionLayer


class GPTLikeLM(nn.Module):
    def __init__(self, vocab_size, seq_length, d_model=512, N=6, h=8, dropout=0.1, d_ff=2048):
        super().__init__()

        self.embed = InputEmbeddings(d_model, vocab_size)
        self.pos = PositionalEncoding(d_model, seq_length, dropout)

        decoder_blocks = []
        for _ in range(N):
            self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
            feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
            dummy_attention = MultiHeadAttentionBlock(d_model, h, dropout)
            decoder_blocks.append(
                DecoderBlock(self_attention, dummy_attention, feed_forward, dropout)
            )
        self.decoder = Decoder(nn.ModuleList(decoder_blocks))
        self.proj = ProjectionLayer(d_model, vocab_size)

    def forward(self, x, mask):
        x = self.embed(x)
        x = self.pos(x)
        x = self.decoder(x, encoder_output=None, src_mask=None, tgt_mask=mask)
        return self.proj(x)
