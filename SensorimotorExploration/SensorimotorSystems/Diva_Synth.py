'''
Created on Feb 3, 2017

@author: Juan Manuel Acevedo Valle
'''
import os

import numpy.linalg  as linalg
import numpy as np      
from numpy import tanh, matrix, array
from scipy.io import loadmat as loadmat
from numpy_groupies.aggregate_weave import aggregate 
    
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
            Art = Art.reshape((13,1),order = 'F')
            
        x = matrix(np.multiply(self.vt['vt_scale'][idx,0].transpose(), Art[0,idx]))
        Outline = self.vt['vt_average'] + self.vt['vt_base'][:,idx].reshape([396,10]) * x.transpose()
        
        #=======================================================================
        # % computes somatosensory output (explicitly from vocal tract configuration)        
        #=======================================================================
        
        Som = np.zeros((8,))
        a, b, sc, af, d = self.xy2ab(Outline)
        Som[0:6] = np.maximum(-1*np.ones((len(sc),)),np.minimum(np.ones((len(sc),)), -tanh(1*sc) ))
        Som[6:7] = Art[-2:]
        
        Aud = np.zeros((4,))
        Aud[1] = 100 + 50 * Art[-3]
        dx = Art[idx] - self.fmfit['fmfit_mu'];
        p = -1 * np.sum(dx * np.multiply(self.fmfit['fmfit_iSigma'],dx),axis=1)/2;
        
        '''
        p = fmfit.p.*exp(p-max(p));
        p = p/sum(p);
        px = p*[Art(idx)',1];
        Aud(2:4) = fmfit.beta_fmt*px(:);
        '''
        ####################### Not implemente yet (nargout = 2 or 3)
        #---------------------------- if ~isempty(fmfit)&&nargout>1&&nargout<=3,
            #------------------------------------ Som(1:6)=fmfit.beta_som*px(:);
            #------------------------------------------ Som(7:8)=Art(end-1:end);
        #------------------------------------------------------------------- end
        
        return Aud, Som, Outline, af, d 
         
        

    def xy2ab(self, x, y = False):  #x -> Outline (column matrix)
        x = np.asarray(x).flatten()
        if not hasattr(self, 'ab_alpha'):
            amax = 220
            alpha = array([1, 1, 1, 1, 1, 1, 1])
            beta = array([.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25])
            idx = [range(60), range(60,70), range(70,80), range(80,120), range(120,150), range(150,190), range(190,amax)]
            ab_alpha = np.zeros((amax,))
            ab_beta = np.zeros((amax,))
            for n1 in range(len(idx)):
                ab_alpha[idx[n1]] = alpha[n1]
                ab_beta[idx[n1]] = beta[n1]
            
            h = np.hanning(51)/sum(np.hanning(51)); #Not same result as in hanning
            idx_2 = np.zeros((25,))
            idx_2 = np.concatenate((idx_2, np.array(range(amax))))
            idx_2 = np.concatenate((idx_2, (amax-1)*np.ones((25,))))
            idx_2 = idx_2.tolist()
            
            ab_alpha = np.convolve(ab_alpha[idx_2],h,'valid')
            ab_beta = np.convolve(ab_beta[idx_2],h,'valid')
            self.ab_alpha = ab_alpha
            self.ab_beta = ab_beta
            
        if not y:
            i = np.array([0 + 1j])[0]
            x=np.exp(-i*np.pi/12) * x
            y=np.imag(x)
            x=np.real(x)
        #=======================================================================
        # % Grid
        #=======================================================================
        x0 = 45       #%90
        y0 = -100     #%-60
        r = 60        #%30
        k = np.pi * (r/2)
        d = 0.75/10    #%unitstocm
        
        a = np.zeros(x.shape)
        b = np.zeros(x.shape)
        i1 = np.where(np.less(y,y0))
        i2 = np.where(np.less(x,x0))
        i3 = np.where(np.logical_and(np.greater_equal(y, y0), np.greater_equal(x, x0)))
        
        #=======================================================================
        # % a,b: "linearized" coordinates along vocal tract
        #======================================================================= 
        
        a[i1] = y[i1]-y0
        b[i1] = x[i1]-x0
        a[i2] = k+x0-x[i2]
        b[i2] = y[i2]-y0
        z = x[i3]-x0 + i*(y[i3]-y0)
        a[i3] = r*np.angle(z)
        b[i3] = np.abs(z)
        #=======================================================================
        # % tube area
        #=======================================================================
         
        olips = range(29,45) 
        ilips = range(256,302)
        #------------------------------------------------- owall = range(44,164)  #Not used in Matlab
        iwall = range(163+10,257)
        oall = range(29,164)
        iall = range(163,302)
        xmin = -20; 
        ymin = -160;
        amin = ymin-y0
        amax=np.ceil((x0-xmin+k-amin))  #here is the variability of the af vector
        
        fact = 3
        
        # wallab1 = accumarray(max(1,min(fact*9, ceil(fact*9*(a(oall)-amin)/amax))),b(oall),[fact*9,1],@min,nan)
        # wallab2 = accumarray(max(1,min(fact*9, ceil(fact*9*(a(iwall)-amin)/amax))),b(iwall),[fact*9,1],@max,nan)
        
        idx_wallab1 = np.int_(np.maximum(np.zeros((len(oall),)),np.minimum(((fact*9)-1)*np.ones((len(oall),)), np.ceil(fact*9*(a[oall]-amin)/amax)-1)))
        idx_wallab2 = np.int_(np.maximum(np.zeros((len(iwall),)),np.minimum(((fact*9)-1)*np.ones((len(iwall),)), np.ceil(fact*9*(a[iwall]-amin)/amax)-1)))
        wallab1 = aggregate(idx_wallab1,b[oall], size = fact*9, func = 'min', fill_value=None)
        wallab2 = aggregate(idx_wallab2,b[iwall], size = fact*9, func = 'max', fill_value=None)
        
        lipsab1 = np.nanmin(b[olips])
        lipsab2 = np.max(b[ilips])
        
        mind_precursor = wallab1[range(fact*2,fact*8)]-wallab2[range(fact*2,fact*8)]
        mind = np.nanmin(mind_precursor.reshape((fact,6), order = 'F'), axis = 0)
        sc = mind[range(4)]    
        sc = np.append(sc,np.nanmin(mind[range(4,6)]))
        sc = np.append(sc,lipsab1-lipsab2); 
        sc = d * sc #In Matlab this is a column vector
        
        w = 2
        
        #ab1 = aggregate(max(1,min(amax, round((a(oall)-amin)))),b(oall),[amax,1],min,nan);
        #ab2 = aggregate(max(1,min(amax, round((a(iall)-amin)))),b(iall),[amax,1],max,nan);
        
        idx_ab1 = np.int_(np.maximum(np.zeros((len(oall),)),np.minimum((amax-1)*np.ones((len(oall),)), np.round(a[oall]-amin-1))))
        idx_ab2 = np.int_(np.maximum(np.zeros((len(iall),)),np.minimum((amax-1)*np.ones((len(iall),)), np.round(a[iall]-amin-1))))
        
        ab1 = aggregate(idx_ab1,b[oall],size = amax, func= 'min', fill_value=None);
        ab2 = aggregate(idx_ab2,b[iall],size = amax, func = 'max', fill_value=None);
        
        ab1[np.isnan(ab1)] = np.inf
        ab2[np.isnan(ab2)] = -np.inf
        for n1 in range(w):
            ab1[1:-1] = np.minimum(np.minimum(ab1[1:-1], ab1[0:-2]), ab1[2:])
        for n1 in range(w):
            ab2[1:-1] = np.maximum(np.maximum(ab2[1:-1], ab2[0:-2]), ab2[2:])
        

        idx_af = np.logical_and(np.greater(ab1,0),np.greater(ab2,0))
        af = d*(ab1[idx_af]-ab2[idx_af])
        idx = None
        for ii in range(len(af)):
            if af[ii] > 0: 
                idx =  ii
                break
       
        #=======================================================================
        # % af: area function
        #=======================================================================
        af_tmp = np.minimum(np.zeros((len(af),)),af)  
        af = af_tmp + np.multiply(ab_alpha[idx_af], np.power(np.maximum(np.zeros((len(af),)),af), ab_beta[idx_af]))
        af = af[idx:]
        for ii in range(len(af)-1,-1,-1):
            if np.isinf(af[ii]):
                af = np.delete(af, -1) 
            else:
                break
        return a, b, sc, af, d
       
        