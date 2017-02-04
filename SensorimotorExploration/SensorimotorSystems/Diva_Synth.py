'''
Created on Feb 3, 2017

@author: Juan Manuel Acevedo Valle
'''
import os

import numpy.linalg  as linalg
import numpy as np      
from numpy import tanh, matrix, array
from scipy.io import loadmat as loadmat
    
class Diva(object):
    '''
    Python implementation of the Diva Synthesizer
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.vt = None
        self.fmfit = None
        
        abs_path = os.path.dirname(os.path.abspath(__file__))

        self.diva_synth_vt_file = abs_path + '/DIVA/vt_py.mat'
        self.diva_synth_fmfit_file = abs_path + '/DIVA/fmfit_py.mat'
        vt = loadmat(self.diva_synth_vt_file)
        fmfit = loadmat(self.diva_synth_fmfit_file)
        
        keys = ['vt_scale', 'vt_base', 'vt_average', 'vt_box']
        for key in keys:
            vt[key] = matrix(vt[key])
        keys = ['fmfit_beta_fmt', 'fmfit_p', 'fmfit_beta_som', 'fmfit_mu', 'fmfit_isigma']
        for key in keys:
            fmfit[key] = matrix(fmfit[key])
        self.vt = vt
        self.fm_fit = fmfit
        
    
    def get_audsom(self, Art):
        '''
             Art n_samples x n_articulators(13)
        '''
        Art = tanh(Art)
        n_samples = Art.shape[0]
        if n_samples == 1:
            pass
        else:
            Aud, Som, Outline, af = self.get_sample(Art);
        
        
    def get_sample(self, Art):
        '''
            Art matrix 1 x n_articulators(13) 
            % computes auditory/somatosensory representations
            % Art(1:10) vocaltract shape params
            % Art(11:13) F0/P/V params
            % Aud(1:4) F0-F3 pitch&formants
            % Som(1:6) place of articulation (~ from pharyngeal to labial closure)
            % Som(7:8) P/V params (pressure,voicing)
        '''
        
        #======================================================================
        # computes vocal tract configuration
        #======================================================================
        idx = range(10)
        
        if Art.shape[0] > 1:        
            Art = Art.reshape([13,1])
            
        x = matrix(np.multiply(self.vt['vt_scale'][idx,0].transpose(), Art[0,idx]))
        Outline = self.vt['vt_average'] + self.vt['vt_base'][:,idx].reshape([396,10]) * x.transpose()
        
        #=======================================================================
        # % computes somatosensory output (explicitly from vocal tract configuration)        
        #=======================================================================
        Som = matrix(np.zeros(8,1))
        a, b, sc, af, d = xy2ab(Outline)
        
        
        
        
        
        return 0, 0, Outline, 0
         
        

    def xy2ab(self, Outline):
        if not hasattr(self, 'ab_alpha'):
            amax = 220
            alpha = matrix([1, 1, 1, 1, 1, 1, 1])
            beta = matrix([.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25])
            idx = [range(60), range(60,70), range(70,80), range(80,120), range(120,150), range(150,190), range(190,amax)]
            ab_alpha=matrix(np.zeros(amax,1))
            ab_beta=matrix(np.zeros(amax,1))
            for n1 in range(len(idx)):
                ab_alpha[idx[n1]]=alpha(n1)
                ab_beta[idx[n1]]=beta(n1)
            
            h=hanning(51)/sum(hanning(51));
            ab_alpha=convn(ab_alpha([ones(1,25),1:end,end+zeros(1,25)]),h,'valid');
            ab_beta=convn(ab_beta([ones(1,25),1:end,end+zeros(1,25)]),h,'valid');
        
        return a. b, sc, af, d        
            

        
         