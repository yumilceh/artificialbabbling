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
    
    
    arts = [[0.1]*13, [0.7]*13]
    arts[0][11:] = ([1]*3)
    arts[1][11:] = ([1]*3)
    
    arts = np.array(arts).transpose()
    arts = np.tile(arts, 40).transpose()
    Aud, af = diva_synth.get_sound(arts) 
    
    welktjv = 0
    
    pass
    #plt.plot(np.real(Outline),np.imag(Outline))
    #plt.show()