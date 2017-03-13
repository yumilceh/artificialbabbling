'''
Created on Feb 5, 2016

@author: yumilceh
'''

if __name__ == '__main__':
    import os,sys,random
    sys.path.append(os.getcwd())
    from SensorimotorExploration.Systems.Diva2015a import DivaProprio2015a

    divaAgent=DivaProprio2015a()
    divaAgent.set_action([2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0.7, 0.7, -3, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0.7, 0.7])
    divaAgent.getMotorDynamics()
    divaAgent.vocalize()
    
    divaAgent.plotVocalTractShape(0.1)
    
    divaAgent.getSoundWave(play=1,save=0)
    #===========================================================================
    # data=np.asarray(divaAgent.soundWave)
    #===========================================================================
    
    divaAgent.getVocalizationVideo()
    #divaAgent.stop()
    #del(divaAgent)

    