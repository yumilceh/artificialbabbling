'''
Created on Feb 4, 2017

@author: Juan Manuel Acevedo Valle
'''
from SensorimotorSystems.Diva_Synth import Diva
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    art = [0.8]*13
    art = np.matrix(art)
    
    diva_synth = Diva()
    
    a,b,Outline,d = diva_synth.get_sample(art)
    
   

    plt.plot(np.real(Outline),np.imag(Outline))
    plt.show()