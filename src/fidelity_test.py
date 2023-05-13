import sys
from problems.freq_domain import FreqDomain
from problems.time_domain import TimeDomain
import wave_helpers
import os
import utils
import numpy as np
import scipy.fft

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
    return f * pars["R0"] / np.sqrt( h1(f)**2 + h2(f)**2 )

"""
Calclulate frequency domain fidelity.
"""
def fidelity_freq(orig, comp, fs, metric = lambda f:1): #original & compressed wav
    l     = min(len(orig),len(comp)) 
    Torig = orig[:l] #truncated to min length
    Tcomp = comp[:l] #truncated to min length
    Forig = scipy.fft.dct(Torig)
    Fcomp = scipy.fft.dct(Tcomp)
    freq  = [(fs/l)*i for i in range (l)]
    dif   = np.array([((Forig[i]-Fcomp[i]))**2*metric(freq[i]) for i in range(l)])
    ref   = np.array([((Forig[i]))**2*metric(freq[i]) for i in range(l)])
    return 1-sum(dif)/sum(ref)

"""
Calculate time-domain fidelity.
"""
def fidelity(orig, comp): #original & compressed wav
    l   = min(len(orig),len(comp)) 
    dif = np.array([(orig[i]-comp[i])**2 for i in range(l)])
    ref = np.square(orig[:l])
    return 1-sum(dif)/sum(ref)

def test_fidelity(prob, data_root, fid_func):
    data_root = utils.path_slash(data_root)

    fids = []

    for filename in os.listdir(data_root):
        wav = wave_helpers.import_to_array(data_root + filename)

        inp = wav['frames']
        preprocessed = prob.preprocess(inp)

        out = prob.postprocess(prob.forward(preprocessed).detach(), preprocessed.shape) 
        fids.append(float(fid_func(inp.detach().numpy(), out.detach().numpy())))
        print(fids[-1])

    return fids

def test_args(args):
    if args[0] == 'freq':
        prob = FreqDomain()
        prob.load_model('../examples/models/freq_domain')
    elif args[0] == 'time':
        prob = TimeDomain()
        prob.load_model('../examples/models/time_domain')
    else:
        assert False, 'Invalid model type'

    if args[1] == 'time':
        fid_func = fidelity
    elif args[1] == 'freq':
        fid_func = lambda orig, comp: fidelity_freq(orig, comp, 16000, RITU)
    else:
        assert False, 'Invalid fidelity function'

    data_root = args[2]
    return test_fidelity(prob, data_root, fid_func)

if __name__ == '__main__':
    test_args(sys.argv[1:])