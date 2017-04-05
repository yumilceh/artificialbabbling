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
import os

#from matplotlib.pyplot import autoscale
#from matplotlib.animation import Animation
#from scipy.interpolate.interpolate_wrapper import block
english_vowels = {'i': [296.0, 2241.0, 1.0], 'I': [396.0, 1839.0, 1.0], 'e': [532.0, 1656.0, 1.0],
                  'ae': [667.0, 1565.0, 1.0], 'A': [661.0, 1296.0, 1.0], 'a': [680.0, 1193.0, 1.0],
                  'b': [643.0, 1019.0, 1.0], 'c': [480.0, 857.0, 1.0], 'U': [395.0, 1408.0, 1.0],
                  'u': [386.0, 1587.0, 1.0], 'E': [519.0, 1408.0, 1.0]}

diva_output_scale = [100.0, 500.0, 1500.0, 3000.0]

class DivaStatic(object):
    
    def __init__(self):
        motor_names=['M1','M2','M3','M4','M5','M6','M7','M8','M9','M10','M11','M12','M13']
        sensor_names=['S1','S2','S3']
        somato_names=['P1']
        n_motor=13
        n_sensor=3
        n_somato=1
        outputScale=[100.0,500.0,1500.0,3000.0];
        min_motor_values=np.array([-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,0.,0.,0.])
        max_motor_values=np.array([3.3,3,3,3,3,3,3,3,3,3,1.,1.,1.])
        
        min_sensor_values=np.array([0.0, 0.0, 0.0])
        max_sensor_values=np.array([2.0, 2.0, 1.0])

        name = 'DivaStatic'
        self.name = name

        self.n_motor=n_motor
        self.n_sensor=n_sensor
        self.n_somato=n_somato
        self.motor_names=motor_names
        self.sensor_names=sensor_names
        self.somato_names=somato_names
        
        self.min_motor_values=min_motor_values
        self.max_motor_values=max_motor_values
        self.min_sensor_values=min_sensor_values
        self.max_sensor_values=max_sensor_values
        
        self.motor_command=[0.0] * n_motor
        self.sensor_out=[0.0] * n_sensor
        self.sensor_goal=[0.0] * n_sensor
        self.somato_out=[0.0] * n_somato
        self.competence_result=0.0;
        self.matlabSession=ml.session_factory()   

        abs_path = os.path.dirname(os.path.abspath(__file__))
        
        command_ = 'cd ' + abs_path + '/DivaMatlab/'
        self.matlabSession.run('x=3')    

        self.matlabSession.run(command_) #Path to DivaMatlab functions
        self.matlabSession.putvalue('outputScale', outputScale)
        
    def setMotorCommand(self,motor_command):
        self.motor_command=motor_command    
          
    def vocalize(self):
        self.matlabSession.putvalue('art',self.motor_command)
        self.matlabSession.run('[aud, Som, outline, af] = diva_synth(art)')
        s_ml = self.matlabSession.getvalue('aud').flatten()
        af = self.matlabSession.getvalue('af').flatten()
        
        self.vt_shape = self.matlabSession.getvalue('outline').flatten()
        
        self.sensor_out = s_ml
        minaf = min(af)
        
        self.somato_out = 0
        
        if (minaf < 0.0):
            self.sensor_out = 0. * self.sensor_out
            self.somato_out = 1.
             
    def executeMotorCommand(self):
        self.vocalize()
                    
    def plotVocalTractShape(self, ax = None):
        
        if type(ax) is type(None):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(np.real(self.vt_shape), np.imag(self.vt_shape))
            return fig, ax
        else:
            ax.plot(np.real(self.vt_shape), np.imag(self.vt_shape))
               
    def getSoundWave(self, play=0, save=0, returnArtStates=0, file_name='vt'): #based on explauto
        time = 0.6
        ts = 0.005
        n_samples = int(time/ts)+1
        soundArtStates = np.tile(self.motor_command,(n_samples,1))
        #print(soundArtStates)
        #print('ts=0.005')
        #print(soundArtStates.shape)
        #print('ts=0.01')
        #print(self.artStates.shape)
        self.matlabSession.putvalue('artStates',soundArtStates[:,0:13])
        self.matlabSession.run('save test_art_shape.mat artStates')

        #self.matlabSession.run('save artStates.mat artStates')
        self.matlabSession.run('sound_wave = diva_synth(artStates\', \'sound\')')
        self.soundWave = self.matlabSession.getvalue('sound_wave')
        if(play):
            self.playSoundWave()
        if(save):
            scaled = np.int16(self.soundWave/np.max(np.abs(self.soundWave)) * 32767)
            write(file_name + '.wav', 11025, scaled)
        if(returnArtStates):
            return soundArtStates  

    def plotSoundWave(self, ax = None):
        
        if type(ax) is type(None):
            fig, ax = plt.plot(np.float128(xrange(0,len(self.soundWave)))* self.ts, self.soundWave)
            return fig, ax
        else:
            ax.plot(np.float128(xrange(0,len(self.soundWave)))* self.ts, self.soundWave)
        
        
    
    def playSoundWave(self): #keep in mind that DivaMatlab works with ts=0.005
        import pyaudio 
        self.pa = pyaudio.PyAudio() #If pa and stream are not elements of the self object then sound does not play
        self.stream = self.pa.open(format=pyaudio.paFloat32,
                         channels=1,
                         rate=11025,
                         output=True)
        self.stream.start_stream()
        self.stream.write(self.soundWave.astype(np.float32).tostring())
        
    
    def releaseAudioDevice(self): #any sound in the buffer will be removed
        try:
            self.stream.close()
            self.pa.terminate()
        except:
            pass
        
    def stop(self):
        del self.matlabSession