import torch
import problem
import autoencoder
import utils
import torch

class FreqDomain(problem.ProblemBase):
    def __init__(self, kernel_sizes = (4,2), strides = (2,2), n_fft = 512):
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
        def normalized_loss(reconstructed, target):

            # Due to the convolution/deconvolution, the reconstructed version might be a couple of samples short - 
            # to resolve this, pad it with 0's at the end. The difference should not be significant
            reconstructed_padded = torch.nn.functional.pad(reconstructed, \
                (0, target.shape[1] - reconstructed.shape[1], 0, target.shape[0] - reconstructed.shape[0]), \
                mode = "constant", value = 0.0)

            # take norm as abs of complex value difference
            return (0.5*(torch.abs(reconstructed_padded - target))**2).mean(dtype=torch.float64)

        optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)
        super().__init__(model, normalized_loss, optimizer)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        # reshape decoded output to match original input size / erase padding
        return decoded[:len(x)]

    def encode(self, x):
        return self.autoencoder.encoder(x.unsqueeze(0))
        
    def decode(self, x):
        unpacked = self.unpack(self.autoencoder.decoder(x)).squeeze(0)

        # Due to the convolution/deconvolution, the reconstructed version might be a couple of samples short - 
        # to resolve this, pad it with 0's at the end. The difference should not be significant

        padded = torch.nn.functional.pad(unpacked, \
            (0,0,0,self.n_fft // 2 + 1 - unpacked.shape[0]), mode="constant", value = 0.0)
        return padded

    def preprocess(self, x):
        return torch.stft(self.normalize(x), self.n_fft, return_complex=True)
    
    def postprocess(self, x):
        return self.unpack(torch.istft(x, self.n_fft))