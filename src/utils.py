import torch

"""
Split a torch tensor x into tensors of length n, with zero padding to fill size as needed.
"""
def split_and_pad(x, n):
    chunks = []
    # split into chunks of size n
    for i in range(0, len(x), n):
        chunks.append(x[i:i+n])
    
    # pad last element
    padded = torch.zeros((n,))
    padded[:len(chunks[-1])] = chunks[-1]
    chunks[-1] = padded
    return chunks

"""
Makes sure the last character of a path is a slash so that items inside the folder 
can be accessed.
"""
def path_slash(path):
    if path[-1] != '/' or path[-1] != '\\':
        path += '/'
    return path