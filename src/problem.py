from abc import ABC, abstractmethod

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
