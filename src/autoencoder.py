import torch

class AutoEncoder(torch.nn.Module):
    """
    Initialize an AutoEncoder with encoder layers defined by
    `encoder` and decoder layers defined by `decoder`.
    """
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    """
    Run a forward pass on the AutoEncoder with input `x`.
    """
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
