from torch import nn

from Model.InputEmbeddings import InputEmbeddings
from Model.Encoder import Encoder
from Model.Decoder import Decoder
from Model.PositionalEncoding import PositionalEncoding
from Model.ProjectionLayer import ProjectionLayer


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

