"""
dataset.py

import datasets for examples.
"""

import re
import numpy as np
import matplotlib.pyplot as plt


def example1(template, loopn=2, sig=2/50, quite=False):
    
    theta = np.repeat(template[:, 0], loopn)
    R = np.repeat(template[:, 1], loopn)
    noise = np.random.randn(np.size(R)) * sig
    R = R + noise
    
    if not quite:
        plt.figure(figsize=(8,8))
        plt.polar(np.hstack((template[:,0], template[:,0][0:1])), np.hstack((template[:,1], template[:,1][0:1])), label='template')
        plt.scatter(np.hstack((theta, theta[0:1])), np.hstack((R, R[0:1])), marker='+', c='r', label='noised data')
        plt.legend()
        plt.show()

    return theta, R

def example2( stim_len, fs, n=20, gap_len=(3, 7), noise_sig=1, quite=False):

    tspec = np.linspace(0, stim_len, stim_len*fs)

    taper = np.cos(2 / stim_len * np.pi * (tspec + stim_len/2)) * 0.5 + 0.5
    template = np.cos(5 * np.pi * tspec) * taper

    signal = np.random.randn(fs * np.random.randint(gap_len[0], gap_len[1])) * noise_sig
    marker = np.array([])
    for i in range(n):
        noise = np.random.randn(stim_len*fs) * noise_sig
        gap = np.random.randn(fs*np.random.randint(gap_len[0], gap_len[1])) * noise_sig

        marker = np.hstack((marker, np.size(signal) / fs))
        signal = np.hstack((signal, template  + noise, gap))


    if not quite:
        ttspec = np.linspace(0, np.size(signal)/fs, np.size(signal))

        plt.figure(figsize=(18,3))

        plt.plot(ttspec, signal, label='')
        plt.vlines(marker, np.min(signal), np.min(signal)-1, label='marker')

        plt.xlim([0, ttspec[-1]])
        plt.xlabel('time')
        # plt.ylabel('signal')
        # plt.legend()
        plt.show()
    
    return (signal, marker), template

class Dataset(object):
    
    def __init__(self):
        self.template_1 = np.array([
            [np.pi * 0 / 8, 25/50],
            [np.pi * 1 / 8, 35/50],
            [np.pi * 2 / 8, 50/50],
            [np.pi * 3 / 8, 35/50],
            [np.pi * 4 / 8, 25/50],
            [np.pi * 5 / 8, 20/50],
            [np.pi * 6 / 8, 16/50],
            [np.pi * 7 / 8, 14/50],
            [np.pi * 8 / 8, 13/50],
            [np.pi * 9 / 8, 13/50],
            [np.pi * 10/ 8, 15/50],
            [np.pi * 11/ 8, 13/50],
            [np.pi * 12/ 8, 13/50],
            [np.pi * 13/ 8, 14/50],
            [np.pi * 14/ 8, 16/50],
            [np.pi * 15/ 8, 20/50],
        ])
        
        self.template_2 = None
    
    def load(self, string, **kwargs):
        if re.match(r'[Ee]xample\s?1', string):
            data = example1(self.template_1, **kwargs)
        elif re.match(r'[Ee]xample\s?2', string):
            data, self.template_2 = example2(**kwargs)
        elif re.match(r'[Ee]xample\s?3', string):
            data = plt.imread('./Rick_and_Morty_characters.jpg', format='jpeg')
        else:
            raise ValueError('unknown token: \"%s\"'%string)
            
        return data
    
    def show_gDSI(self, theta, R, gDSI):
        template = self.template_1
        plt.figure(figsize=(8,8))
        plt.polar(np.hstack((template[:,0], template[:,0][0:1])), np.hstack((template[:,1], template[:,1][0:1])), label='template')
        plt.scatter(np.hstack((theta, theta[0:1])), np.hstack((R, R[0:1])), marker='+', c='r', label='noised data')
        plt.polar((0, np.angle(gDSI)), (0, np.abs(gDSI)), linewidth=5, c='g', label='gDSI')
        plt.legend()
        plt.show()
        
    def show_ERP(self, erp):
        template = self.template_2
        plt.figure(figsize=(8,6))
        plt.plot(template, linewidth=3)
        plt.plot(erp, alpha=0.5)
        plt.show()
        
    def show_mono(self, mono):
        plt.imshow(mono, cmap=plt.get_cmap('gray'))
        plt.show()
        
    def show_histogram(self, *hist):
        x = range(256)
        
        plt.figure(figsize = (9,6))
        if len(hist) == 1:
            # mono hist
            plt.bar(x, hist[0], width=1, color='k', alpha=0.5)
            
        elif len(hist) == 3:
            # rgb hist
            plt.bar(x, hist[0], width=1, color='r', alpha=0.5)
            plt.bar(x, hist[1], width=1, color='g', alpha=0.5)
            plt.bar(x, hist[2], width=1, color='b', alpha=0.5)
            
        else:
            raise ValueError('unknown token')
            
        plt.xlim((0, 255))
        plt.show()
            
            
            