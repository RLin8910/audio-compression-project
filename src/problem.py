from abc import ABC, abstractmethod
import os
import utils
import wave_helpers
import pickle
import torch

"""
Parent class of all trainable autoencoder problems.
"""
class ProblemBase(ABC):
    save_shape = False

    def __init__(self, autoencoder, loss, optimizer):
        self.autoencoder = autoencoder
        self.loss = loss
        self.optimizer = optimizer

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
                samp = self.preprocess(samp)
                # Output of Autoencoder
                reconstructed = self.forward(samp)
                
                # Calculate loss
                loss = self.loss(reconstructed, samp)
                print(float(loss))
                
                # Back propagate
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Store loss
                losses.append(loss)
        
        return losses

    """
    Given an uncompressed input file, compress it and export it to the desired path.
    """
    def compress_and_export(self, in_path, out_path):
        # load original
        orig = wave_helpers.import_to_array(in_path)
        frames = self.preprocess(orig['frames'])

        # Allow saving original shape in case postprocessing needs it.
        if self.save_shape:
            orig['shape'] = frames.shape
        # compress frames
        # detach because we don't care about gradient if we aren't training
        orig['frames'] = self.encode(frames).detach()
        # save
        with open(out_path, 'wb') as output:
            pickle.dump(orig, output)

    """
    Given a compressed file, decompress it and export it to the desired path
    """
    def decompress_and_export(self, in_path, out_path):
        # load original
        with open(in_path, 'rb') as input:
            orig = pickle.load(input)
            shape = None
            if self.save_shape:
                shape = orig['shape']
            frames = self.postprocess(self.decode(orig['frames']), shape).detach().numpy()
            # export
            wave_helpers.export_to_file(frames, orig['framerate'], orig['channels'], orig['sampwidth'], out_path)


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

    """
    Preprocess data before encoding. Can be overwritten.
    """
    def preprocess(self, x):
        return x
    
    """
    Postprocess data after decoding for export - not used in calculating loss. Can be overwritten.
    """
    def postprocess(self, x, shape = None):
        return x

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
