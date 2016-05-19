'''
Created on Feb 5, 2016

@author: yumilceh
'''
from Agent.SensorimotorSystems.Diva_Proprio2015a import Diva_Proprio2015a
import numpy as np



if __name__ == '__main__':
    divaAgent=Diva_Proprio2015a()
    divaAgent.setMotorCommand([2,0,2,0,0,0,0,0,0,0,0,0.7,0.7,-3,0,2,0,0,0,0,0,0,0,0,0.7,0.7])    
    divaAgent.getMotorDynamics()
    divaAgent.vocalize()
    
    #divaAgent.getOutline()
    #divaAgent.plotOutline(0.1)
    
    #divaAgent.getSoundWave(play=1,save=0)
    #data=np.asarray(divaAgent.soundWave)
    
    divaAgent.getVocalizationVideo()
    #divaAgent.stop()
    #del(divaAgent)

    