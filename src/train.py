import wave_helpers
import utils
import torch
import os

"""
Train to solve the problem `problem` over all files found in the path `data_root` for `epochs` epochs.
"""
def train(problem, epochs, data_root):
    data = get_data(data_root)
    problem.train(data, epochs)

def get_data(data_root):
    data_root = utils.path_slash(data_root)

    for filename in os.listdir(data_root):
        yield wave_helpers.import_to_array(data_root + filename)['frames']