import torch
import problem
import autoencoder
import utils
import torch

class TimeDomain(problem.ProblemBase):
    def __init__(self, kernel_size = 400, stride = 2):
        encoder = torch.nn.Sequential(
            torch.nn.Conv1d(1, 1, kernel_size, stride),
            torch.nn.Conv1d(1, 1, kernel_size, stride),
        )
            
        decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(1, 1, kernel_size, stride),
            torch.nn.ConvTranspose1d(1, 1, kernel_size, stride),
        )

        model = autoencoder.AutoEncoder(encoder, decoder)
        mse_loss = torch.nn.MSELoss()

        # Due to the convolution/deconvolution, the reconstructed version might be a couple of samples short - 
        # to resolve this, pad it with 0's at the end. The difference should not be significant

        normalized_loss = lambda reconstructed, target: \
            mse_loss(utils.pad(self.normalize(reconstructed), len(target)), self.normalize(target))

        optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-4,
                             weight_decay = 1e-8)
        super().__init__(model, normalized_loss, optimizer)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        # reshape decoded output to match original input size / erase padding
        return decoded[:len(x)]

    def encode(self, x):
        return self.autoencoder.encoder(self.normalize(x).unsqueeze(0).unsqueeze(0))
        
    def decode(self, x):
        return self.unpack(self.autoencoder.decoder(x)).squeeze(0).squeeze(0)