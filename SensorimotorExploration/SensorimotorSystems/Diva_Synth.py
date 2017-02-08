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
    
class Object(object):
    pass


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
            Aud, Som, Outline, af, d = self.get_sample(Art)
        return Aud, Som, Outline, af
    
    def get_sound(self, art):
        synth = Object()
        synth.fs = 11025.
        synth.update_fs = 200. # Modify sample frequency
        synth.f0 = 120.
        synth.samplesperperiod = np.ceil(synth.fs/synth.f0)
        
        glt_in = array(np.arange(0,1-1/synth.samplesperperiod + 1/synth.samplesperperiod ,1/synth.samplesperperiod))
        synth.glottalsource = glotlf(0, glt_in)
        
        synth.f = array([0,1])
        synth.filt = array([0,0])
        synth.pressure = 0.
        #-------------------------------------------------- %synth.modulation=1;
        synth.voicing = 1.
        synth.pressurebuildup = 0.
        synth.pressure0 = 0.
        synth.sample = np.zeros((synth.samplesperperiod,))
        synth.k1 = 1 
        synth.numberofperiods = 1
        synth.samplesoutput = 0

        self.vt['idx'] = range(10)
        self.vt['pressure'] = 0.
        self.vt['f0'] = 120
        self.vt['closed'] = 0
        self.vt['closure_time'] = 0.
        self.vt['closure_position'] = 0.
        self.vt['opening_time'] = 0.
        
        voices = {'F0': array([120, 340]),'size':array([1,.7])}
        opt = Object()
        opt.voices = 1
        
        
        ndata = art.shape[0]
        dt = 0.005
        time = 0.
        s = np.zeros((np.ceil((ndata+1)*dt*synth.fs),1))
        
        while time<(ndata+1)*dt:
            #---------------------------------- % sample articulatory parameters
            t0 = np.floor(time/dt)
            t1 = (time-t0*dt)/dt
            [nill,nill,nill,af1,d]=diva_synth_sample(Art(:,min(ndata,1+t0)))
            [nill,nill,nill,af2,d]=diva_synth_sample(Art(:,min(ndata,2+t0)))
            naf1=numel(af1)
            naf2=numel(af2)
            if naf2<naf1,af2(end+(1:naf1-naf2))=af2(end); end
            if naf1<naf2,af1(end+(1:naf2-naf1))=af1(end); end
            af=af1*(1-t1)+af2*t1
            FPV=max(-1,min(1, Art(end-2:end,min(ndata,1+t0))*(1-t1)+Art(end-2:end,min(ndata,2+t0))*t1 ))
            vt.voicing=(1+tanh(3*FPV(3)))/2
            vt.pressure=FPV(2)
            vt.pressure0=vt.pressure>.01
            vt.f0=100+20*FPV(1)
            
            af0=max(0,af)
            k=.025;af0(af0>0&af0<k)=k;
            minaf=min(af)
            minaf0=min(af0)
            vt.af=af
        %      display(af)
        %      DIVA_x.af_sample=DIVA_x.af_sample+1
        %      DIVA_x.af(:,DIVA_x.af_sample)=af
        %    tracks place of articulation
            if minaf0==0, 
                release=0
                vt.opening_time=0; vt.closure_time=vt.closure_time+1
                vt.closure_position=find(af0==0,1,'last')
                if ~vt.closed, closure=vt.closure_position; else closure=0; end
                vt.closed=1
            else
                if vt.closed, release=vt.closure_position; release_closure_time=vt.closure_time; else release=0; end
                if (vt.pressure0&&~synth.pressure0) vt.opening_time=0; end
                vt.opening_time=vt.opening_time+1
                vt.closure_time=0
                [nill,vt.closure_position]=min(af)
                closure=0
                vt.closed=0
            end
        %     display(vt.closed)
            if release>0  af=max(k,af);minaf=max(k,minaf);minaf0=max(k,minaf0); end
            
            if release>0, 
                            vt.f0=(.95+.1*rand)*voices(opt.voices).F0
                            synth.pressure=0;%modulation=0 
            elseif  (vt.pressure0&&~synth.pressure0) 
                            vt.f0=(.95+.1*rand)*voices(opt.voices).F0
                            synth.pressure=vt.pressure; synth.f0=1.25*vt.f0 
                            synth.pressure=1;%synth.modulation=1 
            elseif  (~vt.pressure0&&synth.pressure0&&~vt.closed), synth.pressure=synth.pressure/10
            end
            
            % computes glottal source
            synth.samplesperperiod=ceil(synth.fs/synth.f0)
            pp=[.6,.2-.1*synth.voicing,.25];%10+.15*max(0,min(1,1-vt.opening_time/100))]
            synth.glottalsource=10*.25*glotlf(0,(0:1/synth.samplesperperiod:1-1/synth.samplesperperiod)',pp)+10*.025*synth.k1*glotlf(1,(0:1/synth.samplesperperiod:1-1/synth.samplesperperiod)',pp);
            numberofperiods=synth.numberofperiods
                
            % computes vocal tract filter
            [synth.filt,synth.f,synth.filt_closure]=a2h(af0,d,synth.samplesperperiod,synth.fs,vt.closure_position,minaf0)
            synth.filt=2*synth.filt/max(eps,synth.filt(1))
            synth.filt(1)=0
            synth.filt_closure=2*synth.filt_closure/max(eps,synth.filt_closure(1))
            synth.filt_closure(1)=0
            
            % computes sound signal
            w=linspace(0,1,synth.samplesperperiod)'
            if release>0,%&&synth.pressure>.01,
                u=synth.voicing*1*.010*(synth.pressure+20*synth.pressurebuildup)*synth.glottalsource + (1-synth.voicing)*1*.010*(synth.pressure+20*synth.pressurebuildup)*randn(synth.samplesperperiod,1)
        %         if release_closure_time<40
        %             u=1*.010*synth.pressure*synth.glottalsource;%.*(0.25+.025*randn(synth.samplesperperiod,1)); % vocal tract filter
        %         else
        %             u=1*.010*(synth.pressure+synth.pressurebuildup)*randn(synth.samplesperperiod,1)
        %         end
                v0=real(ifft(fft(u).*synth.filt_closure))
                numberofperiods=numberofperiods-1
                synth.pressure=synth.pressure/10
                vnew=v0(1:synth.samplesperperiod)
                v0=(1-w).*synth.sample(ceil(numel(synth.sample)*(1:synth.samplesperperiod)/synth.samplesperperiod))+w.*vnew
                synth.sample=vnew        
            else v0=[]; end
            if numberofperiods>0,
                %u=0.25*synth.modulation*synth.pressure*synth.glottalsource.*(1+.1*randn(synth.samplesperperiod,1)) % vocal tract filter
                u=0.25*synth.pressure*synth.glottalsource.*(1+.1*randn(synth.samplesperperiod,1)) % vocal tract filter
                u=(synth.voicing*u+(1-synth.voicing)*.025*synth.pressure*randn(synth.samplesperperiod,1))
                if minaf0>0&&minaf0<=k, u=minaf/k*u+(1-minaf/k)*.02*synth.pressure*randn(synth.samplesperperiod,1); end
                v=real(ifft(fft(u).*synth.filt))
                
                vnew=v(1:synth.samplesperperiod)
                v=(1-w).*synth.sample(ceil(numel(synth.sample)*(1:synth.samplesperperiod)'/synth.samplesperperiod))+w.*vnew
                synth.sample=vnew
                
                if numberofperiods>1
                    v=cat(1,v,repmat(vnew,[numberofperiods-1,1]))
                end
            else v=[]; end
            v=cat(1,v0,v)
            v=v+.0001*randn(size(v))
            v=(1-exp(-v))./(1+exp(-v))
            s(synth.samplesoutput+(1:numel(v)))=v
            time=time+numel(v)/synth.fs
            synth.samplesoutput=synth.samplesoutput+numel(v)
            
            % computes f0/amp/voicing/pressurebuildup modulation
            synth.pressure0=vt.pressure0
            alpha=min(1,(.1)*synth.numberofperiods);beta=100/synth.numberofperiods
            synth.pressure=synth.pressure+alpha*(vt.pressure*(max(1,1.5-vt.opening_time/beta))-synth.pressure)
            alpha=min(1,.5*synth.numberofperiods);beta=100/synth.numberofperiods
            synth.f0=synth.f0+2*sqrt(alpha)*randn+alpha*(vt.f0*max(1,1.25-vt.opening_time/beta)-synth.f0)%147;%120;
            synth.voicing=max(0,min(1, synth.voicing+.5*(vt.voicing-synth.voicing) ))
            %synth.modulation=max(0,min(1, synth.modulation+.1*(2*(vt.pressure>0&&minaf>-k)-1) ))
            alpha=min(1,.1*synth.numberofperiods)
            synth.pressurebuildup=max(0,min(1, synth.pressurebuildup+alpha*(2*(vt.pressure>0&minaf<0)-1) ))
            synth.numberofperiods=max(1,numberofperiods)
        end
        s=s(1:ceil(synth.fs*ndata*dt))
        end

        
        
        
        
        
         
        
        af = 0
        
        return s af
    
        
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
            #------------------------------------ Som(1:6)=fmfit.beta_som*px(:)
            #------------------------------------------ Som(7:8)=Art(end-1:end)
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
            
            h = np.hanning(51)/sum(np.hanning(51)) #Not same result as in hanning
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
        xmin = -20 
        ymin = -160
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
        sc = np.append(sc,lipsab1-lipsab2) 
        sc = d * sc #In Matlab this is a column vector
        
        w = 2
        
        #ab1 = aggregate(max(1,min(amax, round((a(oall)-amin)))),b(oall),[amax,1],min,nan)
        #ab2 = aggregate(max(1,min(amax, round((a(iall)-amin)))),b(iall),[amax,1],max,nan)
        
        idx_ab1 = np.int_(np.maximum(np.zeros((len(oall),)),np.minimum((amax-1)*np.ones((len(oall),)), np.round(a[oall]-amin-1))))
        idx_ab2 = np.int_(np.maximum(np.zeros((len(iall),)),np.minimum((amax-1)*np.ones((len(iall),)), np.round(a[iall]-amin-1))))
        
        ab1 = aggregate(idx_ab1,b[oall],size = amax, func= 'min', fill_value=None)
        ab2 = aggregate(idx_ab2,b[iall],size = amax, func = 'max', fill_value=None)
        
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
    
    
def glotlf(d, t = None, p = None):
    if t is None:
        tt = array(range(99))/100
    else:
        tt = t-np.floor(t)
    
    u = np.zeros((len(tt),))
    de = array([0.6, 0.1, 0.2])
    
    p = de #Only implemented for one/two input arguments
    
    te = p[0]
    mtc = te-1
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
    
    pre_ta_tb = np.less(tt, te)
    ta = np.where(pre_ta_tb)
    tb = np.where(np.logical_not(pre_ta_tb))
    
    if d == 0:
        u[ta] = e0 * (np.multiply(np.exp(a*tt[ta]),(a*np.sin(wa*tt[ta])-wa*np.cos(wa*tt[ta])))+wa)/(a**2.+wa**2.)
        u[tb] = e1 * (np.exp(mtc/rb)*(tt[tb]-1-rb)+np.exp((te-tt[tb])/rb)*rb)
    elif d==1:
        u[ta] = e0 * np.multiply(np.exp(a*tt[ta]),np.sin(wa*tt[ta]))
        u[tb] = e1 * (np.exp(mtc/rb)-np.exp((te-tt[tb])/rb))
    elif d==2:
        u[ta] = e0 * np.multiply(np.exp(a*tt[ta]),(a*np.sin(wa*tt[ta])+wa*np.cos(wa*tt[ta])))
        u[tb] = e1 * np.exp((te-tt[tb])/rb)/rb
    else:
        print('Derivative must be 0,1 or 2')
        raise ValueError
    
    return u
     
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
            
        