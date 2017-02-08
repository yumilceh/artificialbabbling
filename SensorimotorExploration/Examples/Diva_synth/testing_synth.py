'''
Created on Feb 4, 2017

@author: Juan Manuel Acevedo Valle
'''
from SensorimotorSystems.Diva_Synth import Diva
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    art = [0.1]*13
    
    art = np.array(art)
    
    diva_synth = Diva()
    
    Aud, Som, Outline, af = diva_synth.get_audsom(art)
    
    
    arts = [0.1]*13
    arts[10:] = ([1]*3)
    arts = 80 * [arts]
    arts = np.array(arts)
    Aud, af = diva_synth.get_sound(arts) 
    
    pass
    #plt.plot(np.real(Outline),np.imag(Outline))
    #plt.show()