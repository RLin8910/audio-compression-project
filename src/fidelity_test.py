import sys
from problems.freq_domain import FreqDomain
from problems.time_domain import TimeDomain
import wave_helpers
import os
import utils
import numpy as np

def fidelity(orig, comp): #original & compressed wav
    l   = min(len(orig),len(comp)) 
    dif = np.array([(orig[i]-comp[i])**2 for i in range(l)])
    ref = np.square(orig[:l])
    return 1-sum(dif)/sum(ref)

def test_fidelity(prob, data_root):
    data_root = utils.path_slash(data_root)

    fids = []

    for filename in os.listdir(data_root):
        wav = wave_helpers.import_to_array(data_root + filename)

        inp = wav['frames']
        preprocessed = prob.preprocess(inp)

        out = prob.postprocess(prob.forward(preprocessed).detach(), preprocessed.shape) 
        fids.append(float(fidelity(inp, out)))
        print(fids[-1])

    return fids

if __name__ == '__main__':
    if sys.argv[1] == 'freq':
        prob = FreqDomain()
        prob.load_model('../examples/models/freq_domain')
    elif sys.argv[1] == 'time':
        prob = TimeDomain()
        prob.load_model('../examples/models/time_domain')
    else:
        assert False, 'Invalid model type'

    data_root = sys.argv[2]
    test_fidelity(prob, data_root)