import torch
import problem
import autoencoder
import utils
import torch

class SimpleProblem(problem.ProblemBase):
    def __init__(self, inp_size = 16_000, enc_size = 4000):
        encoder = torch.nn.Sequential(
            torch.nn.Linear(inp_size, (inp_size + enc_size) // 2),
            torch.nn.Linear((inp_size + enc_size) // 2, enc_size)
        )
            
        decoder = torch.nn.Sequential(
            torch.nn.Linear(enc_size, (inp_size + enc_size) // 2),
            torch.nn.Linear((inp_size + enc_size) // 2, inp_size),
        )

        model = autoencoder.AutoEncoder(encoder, decoder)
        mse_loss = torch.nn.MSELoss()

        normalized_loss = lambda reconstructed, target: mse_loss(self.normalize(reconstructed), self.normalize(target))
        optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-4,
                             weight_decay = 1e-8)
        super().__init__(model, normalized_loss, optimizer, inp_size, enc_size)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        # reshape decoded output to match original input size / erase padding
        return decoded[:len(x)]

    def encode(self, x):
        # split into chunks of length self.inp_size
        chunks = utils.split_and_pad(x, self.inp_size)
        res = torch.Tensor()
        for chunk in chunks:
            # scale to 0-1 range before inputting
            res = torch.cat((res, self.autoencoder.encoder(self.normalize(chunk))))
        
        return res
        
    def decode(self, x):
        chunks = utils.split_and_pad(x, self.enc_size)
        res = torch.Tensor()
        for chunk in chunks:
            # scale final layer back into output range
            res = torch.cat((res, self.unpack(self.autoencoder.decoder(chunk))))
        
        return res

    """
    Normalizes an audio input into the [-1, 1] range.
    """
    def normalize(self, x):
        return x / 32768

    """
    Unpack an audio input into the standard [-32767, 32767] range again.
    """
    def unpack(self, x):
        return x * 32768