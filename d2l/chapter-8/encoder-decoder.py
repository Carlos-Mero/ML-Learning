import torch
from torch import nn

class Encode(nn.Module):
    """The base encoder interface for the encoder-decoder architecture"""
    def __init__(self):
        super().__init__()

    def forward(self, X, *args):
        pass

class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture"""
    def __init__(self):
        super().__init__()

    def init_state(self, enc_all_outputs, *args):
        pass
    def forward(self, X, state):
        pass

class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        return self.decoder(dec_X, dec_state)[0]
