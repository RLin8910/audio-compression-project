from abc import ABC, abstractmethod
import os
import utils
import torch

"""
Parent class of all trainable autoencoder problems.
"""
class ProblemBase(ABC):
    def __init__(self, autoencoder, loss, optimizer, inp_size = 16_000, enc_size = 1_000):
        self.autoencoder = autoencoder
        self.loss = loss
        self.optimizer = optimizer

        self.inp_size = inp_size
        self.enc_size = enc_size

    """
    Saves the model to the folder at path `path`.
    """
    def save_model(self, path):
        path = utils.path_slash(path)

        if not os.path.exists(path):
            os.makedirs(path)
        
        torch.save(self.autoencoder.encoder.state_dict(), path + 'encoder.pth')
        torch.save(self.autoencoder.decoder.state_dict(), path + 'decoder.pth')

    """
    Loads the model with the root folder at path `path`.
    """
    def load_model(self, path):
        path = utils.path_slash(path)

        self.autoencoder.encoder.load_state_dict(torch.load(path + 'encoder.pth'))
        self.autoencoder.decoder.load_state_dict(torch.load(path + 'decoder.pth'))

        self.autoencoder.encoder.eval()
        self.autoencoder.decoder.eval()

    """
    Train this model on the given data.

    `data` should be an array of arrays, with each row being the frames of an audio file.
    `epochs` is the number of training epochs.

    Returns an array of the losses over time.

    Adapted from: https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/
    """
    def train(self, data, epochs):
        losses = []
        for epoch in range(epochs):
            for samp in data:       
                # Output of Autoencoder
                reconstructed = self.forward(samp)
                print(max(reconstructed))
                print(min(reconstructed))
                
                # Calculate loss
                loss = self.loss(reconstructed, samp)
                
                # Back propagate
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Store loss
                losses.append(loss)
        
        return losses

    """
    Perform an encode/decode step. Defined here separately from AutoEncoder.forward() in case the data needs to be 
    processed and/or reshaped before it can be used as an input to the AutoEncoder or output for comparison with
    the original sample.
    """
    @abstractmethod
    def forward(self, x):
        return self.decode(self.encode(x))

    """
    Encode the data.

    Defined separately from AutoEncoder.encoder in case the data needs to be processed/reshaped before it can be decoded.
    """
    @abstractmethod
    def encode(self, x):
        raise NotImplementedError("Must override forward function")
    
    """
    Decode the data.

    Defined separately from AutoEncoder.decoder in case the data needs to be processed/reshaped before it can be decoded.
    """
    @abstractmethod
    def decode(self, x):
        raise NotImplementedError("Must override forward function")
