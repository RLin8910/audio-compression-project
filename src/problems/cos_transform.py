import torch
import problem
import autoencoder
import utils
import scipy.fft
import numpy
import torch

class CosTransform(problem.ProblemBase):
    save_shape = True

    def __init__(self, kernel_size = 400, stride = 2, samprate = 16000):
        encoder = torch.nn.Sequential(
            torch.nn.Conv1d(1, 1, kernel_size, stride),
            torch.nn.Conv1d(1, 1, kernel_size, stride),
        )
            
        decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(1, 1, kernel_size, stride),
            torch.nn.ConvTranspose1d(1, 1, kernel_size, stride),
        )

        model = autoencoder.AutoEncoder(encoder, decoder)
        """
        ITU-R 468 noise weighting
        source: https://en.m.wikipedia.org/wiki/ITU-R_468_noise_weighting#Summary_of_specification
        """
        def RITU(f,**kwargs):
            pars = {"h16" : -4.7373E-24,
                    "h14" :  2.0438E-15,
                    "h12" : -1.3638E-7,
                    "h25" :  1.3066E-19,
                    "h23" : -2.1181E-11,
                    "h21" :  5.5594E-4,
                    "R0"  :  1.2463E-4
                    }
            pars.update(**kwargs)
            
            def h1(f):
                return pars["h16"]*f**6 + pars["h14"]*f**4 + pars["h12"]*f**2 + 1.
            def h2(f):
                return pars["h25"]*f**5 + pars["h23"]*f**3 + pars["h21"]*f
            return f * pars["R0"] / numpy.sqrt( h1(f)**2 + h2(f)**2 )

        """
        Calculate frequency domain loss with supplied metric
        """
        def loss_freq(orig, comp, fs, metric = lambda f:1): #original & compressed wav
            l = len(orig)

            freq  = torch.tensor([metric((fs/l)*i) for i in range (l)])

            dif   = freq * torch.square(orig-comp)
            ref   = freq * torch.square(orig)
            return torch.sum(dif)/torch.sum(ref)

        # Due to the convolution/deconvolution, the reconstructed version might be a couple of samples short - 
        # to resolve this, pad it with 0's at the end. The difference should not be significant

        # Calculate loss on frequency domain
        normalized_loss = lambda reconstructed, target: \
            loss_freq(target, utils.pad(reconstructed, len(target)), samprate, RITU)

        optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-2,
                             weight_decay = 1e-8)
        super().__init__(model, normalized_loss, optimizer)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        # reshape decoded output to match original input size / erase padding
        return decoded[:x.shape[0]]

    def encode(self, x):
        return self.autoencoder.encoder(x.unsqueeze(0).unsqueeze(0))
        
    def decode(self, x):
        return self.autoencoder.decoder(x).squeeze(0).squeeze(0)

    def preprocess(self, x):
        dct = torch.from_numpy(scipy.fft.dct(self.normalize(x).detach().numpy()))
        return dct
    
    def postprocess(self, x, shape):
        out = torch.from_numpy(scipy.fft.idct(x[:shape[0]].detach().numpy()))
        return self.unpack(out)