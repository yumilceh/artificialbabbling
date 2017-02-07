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
from array import array
    
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
            vt[key] = array(vt[key])
        keys = ['fmfit_beta_fmt', 'fmfit_p', 'fmfit_beta_som', 'fmfit_mu', 'fmfit_isigma']
        for key in keys:
            fmfit[key] = array(fmfit[key])
        self.vt = vt
        self.fmfit = fmfit
        
    
    def get_audsom(self, Art):
        '''
             Art n_samples x n_articulators(13)
        '''
        Art = tanh(Art)
        
        if len(Art.shape)>1:
            n_samples = Art.shape[0]    
        else:
            Aud, Som, Outline, af, d = self.get_sample(Art);
        return Aud, Som, Outline, af
    
    def get_sound(self, art):
        synth = object()
        synth.fs = 11025
        synth.update_fs = 200 # Modify sample frequency
        synth.f0 = 120;
        synth.samplesperperiod = np.ceil(synth.fs/synth.f0);
        
        glt_in = array(range(0,1/synth.samplesperperiod,1-1/synth.samplesperperiod))
        
        synth.glottalsource = glotlf(0, );
        
        '''
        synth.f=[0,1];
        synth.filt=[0,0];
        synth.pressure=0;
        %synth.modulation=1;
        synth.voicing=1;
        synth.pressurebuildup=0;
        synth.pressure0=0;
        synth.sample=zeros(synth.samplesperperiod,1);
        synth.k1=1;
        synth.numberofperiods=1;
        synth.samplesoutput=0;
        
        vt.idx=1:10;
        vt.pressure=0;
        vt.f0=120;
        vt.closed=0;
        vt.closure_time=0;
        vt.closure_position=0;
        vt.opening_time=0;
        
        voices=struct('F0',{120,340},'size',{1,.7});
        opt.voices=1;
        
        ndata=size(Art,2);
        dt=.005;
        time=0;
        s=zeros(ceil((ndata+1)*dt*synth.fs),1);
        
        
        Aud = None
        af = None
        '''
    def get_sample(self, Art):
        '''
            Art numpy array 1 x n_articulators(13) 
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
                    
        x = np.multiply(self.vt['vt_scale'][idx].flatten(), Art[idx])
        Outline = self.vt['vt_average'] + np.dot(self.vt['vt_base'][:,idx].reshape((396,10), order = 'F'), x.reshape((10,1))) #Keep in mind reshaping order
        Outline = Outline.flatten()
        
        #=======================================================================
        # % computes somatosensory output (explicitly from vocal tract configuration)        
        #=======================================================================
        
        Som = np.zeros((8,))
        a, b, sc, af, d = self.xy2ab(Outline)
        Som[0:6] = np.maximum(-1*np.ones((len(sc),)),np.minimum(np.ones((len(sc),)), -tanh(1*sc) ))
        Som[6:] = Art[-2:]
        
        Aud = np.zeros((4,))
        Aud[0] = 100 + 50 * Art[-3]
        dx = Art[idx] - self.fmfit['fmfit_mu']
        p = -1 * np.sum(np.multiply(np.dot(dx, self.fmfit['fmfit_isigma']), dx),axis=1)/2
        p = np.multiply(self.fmfit['fmfit_p'].flatten(),np.exp(p-(np.max(p)*np.ones(p.shape))))
        p = p / np.sum(p)
        px = np.dot(p.reshape((len(p),1)) , np.append(Art[idx],1).reshape((1,len(idx)+1)))
        Aud[1:4] = np.dot(self.fmfit['fmfit_beta_fmt'], px.flatten(1))
        
        ####################### Not implemente yet (nargout = 2 or 3)
        #---------------------------- if ~isempty(fmfit)&&nargout>1&&nargout<=3,
            #------------------------------------ Som(1:6)=fmfit.beta_som*px(:);
            #------------------------------------------ Som(7:8)=Art(end-1:end);
        #------------------------------------------------------------------- end
        
        return Aud, Som, Outline, af, d 

    def xy2ab(self, x, y = False):  #x -> Outline (column matrix)
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
        
        ab_alpha = self.ab_alpha
        ab_beta = self.ab_beta
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
    
    
def glotlf(d, t, p=None):
    if p == None:
        tt = array(range(99))/100
    else:
        tt = t-np.floor(t)
    
    u = np.zeros((len(tt),));
    de = array([0.6, 0.1, 0.2])
    
    p = de #Only implemented for one/two input arguments
    
    te = p[0]
    mtc = t*(10.0**(-1))
    e0 = 1
    wa = np.pi / (te * ( 1 - p[2] ))
    a = -np.log(-p[1]* np.sin(wa * te)) / te
    inta = e0 * ((wa / np.tan(wa*te)-a)/p[1]+wa) / (a**2.0 + wa**2.0)
    
    #----------------------------------------- % if inta<0 we should reduce p(2)
    #------------------------- % if inta>0.5*p(2)*(1-te) we should increase p(2)
    
    rb0 = p[1] * inta
    rb = rb0
    
    #--------------------------- % Use Newton to determine closure time constant
    #----------------------------------- % so that flow starts and ends at zero.
    
    for i in range(4):
        kk = 1 - np.exp(mtc/rb)
        err = rb + mtc * (1 /kk - 1 ) - rb0
        derr = 1 - (1 - kk) * (mtc/ rb / kk)**2.
        rb = rb - err/derr
    e1 = 1 /(p[1]*(1 - np.exp( mtc / rb ) ))
    
    
    ta = np.lower(tt, te)
    tb = ~ta
    
    if d == 0:
        u(ta) = e0 * (np.multiply(np.exp(a*tt[np.where(ta)]),(a*np.sin(wa*tt[np.where(ta)])-wa*cos(wa*tt[np.where(ta)])))+wa)/(a**2.+wa**2.)
        u(tb) = e1 * (np.exp(mtc/rb)*(tt[np.where(tb)]-1-rb)+np.exp((te-tt[np.where(tb)])/rb)*rb)
    elif d==1:
        u(ta) = e0 * np.multiply(np.exp(a*tt[np.where(ta)]),np.sin(wa*ttnp.where[(ta)]))
        u(tb) = e1 * (np.exp(mtc/rb)-np.exp((te-tt[np.where(tb)])/rb))
    elif d==2:
        u(ta) = e0 * np.multiply(np.exp(a*tt[np.where(ta)]),(a*np.sin(wa*tt[np.where(ta)])+wa*np.cos(wa*tt[np.where(ta)])))
        u(tb) = e1 * np.exp((te-tt[np.where(tb)])/rb)/rb
    else:
        print('Derivative must be 0,1 or 2')
        rise ValueError
    
    return 
    doc = '''
        %GLOTLF   Liljencrants-Fant glottal model U=(D,T,P)
        % d is derivative of flow waveform: must be 0, 1 or 2
        % t is in fractions of a cycle
        % p has one row per output point
        %    p(:,1)=open phase [0.6]
        %    p(:,2)=+ve/-ve slope ratio [0.1]
        %    p(:,3)=closure time constant/closed phase [0.2]
        % Note: this signal has not been low-pass filtered
        % and will therefore be aliased
        %
        % Usage example:    ncyc=5;
        %            period=80;
        %            t=0:1/period:ncyc;
        %            ug=glotlf(0,t);
        %            plot(t,ug)
        
        
        %      Copyright (C) Mike Brookes 1998
        %
        %      Last modified Thu Apr 30 17:22:00 1998
        %
        %   VOICEBOX home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %   This program is free software; you can redistribute it and/or modify
        %   it under the terms of the GNU General Public License as published by
        %   the Free Software Foundation; either version 2 of the License, or
        %   (at your option) any later version.
        %
        %   This program is distributed in the hope that it will be useful,
        %   but WITHOUT ANY WARRANTY; without even the implied warranty of
        %   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        %   GNU General Public License for more details.
        %
        %   You can obtain a copy of the GNU General Public License from
        %   ftp://prep.ai.mit.edu/pub/gnu/COPYING-2.0 or by writing to
        %   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        '''
            
        