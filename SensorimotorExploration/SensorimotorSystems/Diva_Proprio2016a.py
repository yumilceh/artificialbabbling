'''
Created on Feb 5, 2016
This sensorimor system defines the DIVA agent used for the CCIA 2015's paper
@author: Juan Manuel Acevedo Valle
'''

#import sys
#import wave
import subprocess as sp
import math
import numpy as np
import pymatlab as ml
import matplotlib.pyplot as plt
from matplotlib import animation 
from scipy.integrate import odeint
from scipy import linspace
from scipy.io.wavfile import write
from SensorimotorSystems.Diva_Proprio2015a import Diva_Proprio2015a
#from matplotlib.pyplot import autoscale
#from matplotlib.animation import Animation
#from scipy.interpolate.interpolate_wrapper import block

class Diva_Proprio2016a(Diva_Proprio2015a):
            
    def getMotorDynamics(self,sound=0):
        if sound:
            ts=0.005;
        else:
            ts=self.ts
            
        durationM1=0.4
        durationM2=0.4
        nSamples=int(0.8/ts+1)
        nSamples1=int(durationM1/ts)+1
        nSamples2=int(durationM2/ts)+1
        y_neutral=[0.0]*13
        y_neutral[11]=-0.25
        y_neutral[12]=-0.25
        y0=[0.0]*26
        y0[:13]=y_neutral
        m1=self.motor_command[:13]
        t1=linspace(0.0,durationM1,nSamples1)
        artStates1=odeint(motorDynamics,y0,t1,args=(self,m1))
        t2=linspace(0.0,durationM2,nSamples2)
        m2=self.motor_command[13:]
        artStates2=odeint(motorDynamics,artStates1[-1,:],t2,args=(self,m2))
        if sound:
            return np.concatenate((artStates1,artStates2))
        else:
            self.artStates= np.zeros((nSamples, 26))
            self.artStates[:nSamples1,:]=artStates1
            self.artStates[nSamples1-1:,:]=artStates2
    
    def vocalize(self):
        ts=self.ts;
        perceptionWindowDuration = 0.4;
        perceptionTime = linspace(ts,perceptionWindowDuration, int(perceptionWindowDuration/ts))
        nPerceptionSamples = (len(perceptionTime))
        self.auditoryResult = [0.0]*6
        proprioceptiveAv = [0.0]*2
        self.matlabSession.putvalue('artStates',self.artStates)
        #self.matlabSession.run('save artStates.mat artStates')
        self.matlabSession.run('mscript_Aud_Proprio')
        auditoryStates = self.matlabSession.getvalue('auditoryStates')
        self.auditoryStates = auditoryStates;
        minaf = self.matlabSession.getvalue('minaf')
        self.somatoOutput = minaf
        '''print('audStates')
        print(auditoryStates)
        print('minaf')
        print(minaf)'''
        
        #First perception time window
        for index in range(nPerceptionSamples):
            #print(index)
            #print(nPerceptionSamples)
            if (self.artStates[index,11]>0) and (self.artStates[index,12]>0) and (minaf[index]>0):
                self.auditoryResult[0]=self.auditoryResult[0]+auditoryStates[index,1]
                self.auditoryResult[1]=self.auditoryResult[1]+auditoryStates[index,2]
                self.auditoryResult[2]=self.auditoryResult[2]+1.0
            proprioceptiveAv[0]=proprioceptiveAv[0]+(minaf[index]/nPerceptionSamples)
        self.auditoryResult[0]=self.auditoryResult[0]/nPerceptionSamples
        self.auditoryResult[1]=self.auditoryResult[1]/nPerceptionSamples
        self.auditoryResult[2]=self.auditoryResult[2]/nPerceptionSamples   
            
        #Second perception time window
        for index in range(nPerceptionSamples):
            #print(index)
            if (self.artStates[index+41,11]>0) and (self.artStates[index+41,12]>0) and (minaf[index+41]>0):
                self.auditoryResult[3]=self.auditoryResult[3]+auditoryStates[index+41,1]
                self.auditoryResult[4]=self.auditoryResult[4]+auditoryStates[index+41,2]
                self.auditoryResult[5]=self.auditoryResult[5]+1.0
            proprioceptiveAv[1]=proprioceptiveAv[1]+(minaf[index+41]/nPerceptionSamples)
        self.auditoryResult[3]=self.auditoryResult[3]/nPerceptionSamples
        self.auditoryResult[4]=self.auditoryResult[4]/nPerceptionSamples
        self.auditoryResult[5]=self.auditoryResult[5]/nPerceptionSamples
                
        self.somatoOutput=0.0
        if((proprioceptiveAv[0]<0.0) or (proprioceptiveAv[1]<0.0)):
            self.somatoOutput=1.0
        self.sensorOutput=self.auditoryResult;     
            
    
def motorDynamics(y,t,self,m):
    dumpingFactor=1.01
    w0=2*math.pi/0.2
    
    dy1=y[13]
    dy2=y[14]
    dy3=y[15]
    dy4=y[16]
    dy5=y[17]
    dy6=y[18]
    dy7=y[19]
    dy8=y[20]
    dy9=y[21]
    dy10=y[22]
    dy11=y[23]
    dy12=y[24]
    dy13=y[25]
        
    dy14=-2*dumpingFactor*w0*y[13]-(pow(w0,2))*y[0]+(pow(w0,2))*m[0]
    dy15=-2*dumpingFactor*w0*y[14]-(pow(w0,2))*y[1]+(pow(w0,2))*m[1]
    dy16=-2*dumpingFactor*w0*y[15]-(pow(w0,2))*y[2]+(pow(w0,2))*m[2]
    dy17=-2*dumpingFactor*w0*y[16]-(pow(w0,2))*y[3]+(pow(w0,2))*m[3]
    dy18=-2*dumpingFactor*w0*y[17]-(pow(w0,2))*y[4]+(pow(w0,2))*m[4]
    dy19=-2*dumpingFactor*w0*y[18]-(pow(w0,2))*y[5]+(pow(w0,2))*m[5]
    dy20=-2*dumpingFactor*w0*y[19]-(pow(w0,2))*y[6]+(pow(w0,2))*m[6]
    dy21=-2*dumpingFactor*w0*y[20]-(pow(w0,2))*y[7]+(pow(w0,2))*m[7]
    dy22=-2*dumpingFactor*w0*y[21]-(pow(w0,2))*y[8]+(pow(w0,2))*m[8]
    dy23=-2*dumpingFactor*w0*y[22]-(pow(w0,2))*y[9]+(pow(w0,2))*m[9]
    dy24=-2*dumpingFactor*w0*y[23]-(pow(w0,2))*y[10]+(pow(w0,2))*m[10]
    dy25=-2*dumpingFactor*w0*y[24]-(pow(w0,2))*y[11]+(pow(w0,2))*m[11]
    dy26=-2*dumpingFactor*w0*y[25]-(pow(w0,2))*y[12]+(pow(w0,2))*m[12]
    
    return [dy1,dy2,dy3,dy4,dy5,dy6,dy7,dy8,dy9,dy10,dy11,dy12,dy13,dy14,dy15,dy16,dy17,dy18,dy19,dy20,dy21,dy22,dy23,dy24,dy25,dy26]
    
