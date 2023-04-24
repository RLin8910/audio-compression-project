"""
Split a list x into lists of length n, with zero padding to fill size as needed.
"""
def split_and_pad(x, n):
    chunks = []
    # split into chunks of size n
    for i in range(0, len(x), n):
        chunks.append(x[i:i+n])
    
    # pad last element
    chunks[-1] += (n - len(chunks[-1]))
    return chunks