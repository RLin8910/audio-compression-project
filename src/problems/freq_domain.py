import torch
import problem
import autoencoder
import utils
import torch

class FreqDomain(problem.ProblemBase):
    save_shape = True

    def __init__(self, kernel_sizes = (4,2), strides = (2,2), n_fft = 256):
        encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_sizes[0], strides[0], padding=1).to(torch.cfloat),
            torch.nn.Conv2d(1, 1, kernel_sizes[1], strides[1], padding=1).to(torch.cfloat),
        )
            
        decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1, 1, kernel_sizes[1], strides[1]).to(torch.cfloat),
            torch.nn.ConvTranspose2d(1, 1, kernel_sizes[0], strides[0]).to(torch.cfloat),
        )

        model = autoencoder.AutoEncoder(encoder, decoder)

        self.n_fft = n_fft

        # Calculate loss on frequency domain
        normalized_loss = lambda reconstructed, target: \
            (0.5*(torch.abs(reconstructed - target))**2).mean(dtype=torch.float32)

        optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-2,
                             weight_decay = 1e-8)
        super().__init__(model, normalized_loss, optimizer)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        # reshape decoded output to match original input size / erase padding
        return decoded[:x.shape[0], :x.shape[1]]

    def encode(self, x):
        return self.autoencoder.encoder(x.unsqueeze(0))
        
    def decode(self, x):
        return self.autoencoder.decoder(x).squeeze(0)

    def preprocess(self, x):
        return torch.stft(self.normalize(x), self.n_fft, return_complex=True)
    
    def postprocess(self, x, shape):
        return self.unpack(torch.istft(x[:shape[0],:shape[1]], self.n_fft))