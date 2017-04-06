"""
Created on Abr 6, 2016
This sensorimor system defines the DIVA agent used for the Epirob 2017's paper
but using divapy
@author: Juan Manuel Acevedo Valle
"""

from matplotlib import animation
import subprocess as sp
from scipy.io.wavfile import write
import math
import numpy as np
from scipy.integrate import odeint
from scipy import linspace
from divapy import Diva
import matplotlib.pyplot as plt

import Tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

english_vowels = {'i': [296.0, 2241.0, 1.0], 'I': [396.0, 1839.0, 1.0], 'e': [532.0, 1656.0, 1.0],
                  'ae': [667.0, 1565.0, 1.0], 'A': [661.0, 1296.0, 1.0], 'a': [680.0, 1193.0, 1.0],
                  'b': [643.0, 1019.0, 1.0], 'c': [480.0, 857.0, 1.0], 'U': [395.0, 1408.0, 1.0],
                  'u': [386.0, 1587.0, 1.0], 'E': [519.0, 1408.0, 1.0]}

# Write down german vowels here
diva_output_scale = [100.0, 500.0, 1500.0, 3000.0]

class Diva2016b(object):
    def __init__(self):
        motor_names = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13', 'M14', 'M15',
                       'M16', 'M17', 'M18', 'M19', 'M20', 'M21', 'M22', 'M23', 'M24', 'M25', 'M26']
        sensor_names = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
        somato_names = ['P1']
        n_motor = 26
        n_sensor = 6
        n_somato = 1
        outputScale = [100.0, 500.0, 1500.0, 3000.0]
        min_motor_values = np.array([-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -0.25, -0.25, -0.25] * 2)
        max_motor_values = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1] * 2)

        min_motor_values_init = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0] * 2)
        max_motor_values_init = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 2)

        min_sensor_values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_sensor_values = np.array([2.0, 2.0, 1.0, 2.0, 2.0, 1.0])

        min_somato_values = np.array([0.])
        max_somato_values = np.array([1.])
        ts = 0.01

        name = 'Diva2016b_py'
        self.name = name

        self.ts = ts
        self.time = linspace(0, .8, int(.8 / ts) + 1)
        self.n_motor = n_motor
        self.n_sensor = n_sensor
        self.n_somato = n_somato
        self.motor_names = motor_names
        self.sensor_names = sensor_names
        self.somato_names = somato_names

        self.min_motor_values = min_motor_values
        self.max_motor_values = max_motor_values
        self.min_motor_values_init = min_motor_values_init
        self.max_motor_values_init = max_motor_values_init
        self.min_sensor_values = min_sensor_values
        self.max_sensor_values = max_sensor_values
        self.min_somato_values = min_somato_values
        self.max_somato_values = max_somato_values

        self.motor_command = [0.0] * n_motor
        self.sensor_out = [0.0] * n_sensor
        self.sensor_goal = [0.0] * n_sensor
        self.somato_out = [0.0] * n_somato
        self.somato_threshold = 0.6
        self.competence_result = 0.0
        self.sensor_instructor = np.empty((self.n_sensor,))
        self.sensor_instructor.fill(np.nan)
        self.synth = Diva()

    def set_action(self, motor_command):
        self.motor_command = motor_command

    def getMotorDynamics(self, sound =0):
        if sound:
            ts = 0.005
        else:
            ts = self.ts

        durationM1 = 0.4
        durationM2 = 0.4
        nSamples = int(0.8 / ts + 1)
        nSamples1 = int(durationM1 / ts) + 1
        nSamples2 = int(durationM2 / ts) + 1
        y_neutral = [0.0] * 13
        y_neutral[11] = 0
        y_neutral[12] = 0
        y0 = [0.0] * 26
        y0[:13] = y_neutral
        m1 = self.motor_command[:13]
        t1 = linspace(0.0, durationM1, nSamples1)
        art_states1 = odeint(motorDynamics, y0, t1, args=(self, m1))
        t2 = linspace(0.0, durationM2, nSamples2)
        m2 = self.motor_command[13:]
        art_states2 = odeint(motorDynamics, art_states1[-1, :], t2, args=(self, m2))
        if sound:
            return np.concatenate((art_states1, art_states2))
        else:
            self.art_states = np.zeros((nSamples, 26))
            self.art_states[:nSamples1, :] = art_states1
            self.art_states[nSamples1 - 1:, :] = art_states2

    def vocalize(self):
        ts = self.ts
        perceptionWindowDuration = 0.38
        perceptionTime = linspace(ts, perceptionWindowDuration, int(perceptionWindowDuration / ts))
        nPerceptionSamples = (len(perceptionTime))
        self.auditoryResult = [0.0] * 6
        proprioceptiveAv = [0.0] * 2

        auditory_states, tr_, tr__, af = self.synth.get_audsom(self.art_states,scale=True)
        self.auditory_states = auditory_states
        minaf = np.array([np.min(x) for x in af])

        # First perception time window
        for index in range(nPerceptionSamples):
            # print(index)
            # print(nPerceptionSamples)
            if (self.art_states[index + 2, 11] > 0) and (self.art_states[index + 2, 12] > 0) and (minaf[index + 2] > 0):
                self.auditoryResult[0] = self.auditoryResult[0] + auditory_states[index + 2, 1]
                self.auditoryResult[1] = self.auditoryResult[1] + auditory_states[index + 2, 2]
                self.auditoryResult[2] = self.auditoryResult[2] + 1.0
            proprioceptiveAv[0] = proprioceptiveAv[0] + (minaf[index + 2] / nPerceptionSamples)
        self.auditoryResult[0] = self.auditoryResult[0] / nPerceptionSamples
        self.auditoryResult[1] = self.auditoryResult[1] / nPerceptionSamples
        self.auditoryResult[2] = self.auditoryResult[2] / nPerceptionSamples

        # Second perception time window
        for index in range(nPerceptionSamples):
            # print(index)
            if (self.art_states[index + 43, 11] > 0) and (self.art_states[index + 43, 12] > 0) and (
                        minaf[index + 43] > 0):
                self.auditoryResult[3] = self.auditoryResult[3] + auditory_states[index + 43, 1]
                self.auditoryResult[4] = self.auditoryResult[4] + auditory_states[index + 43, 2]
                self.auditoryResult[5] = self.auditoryResult[5] + 1.0
            proprioceptiveAv[1] = proprioceptiveAv[1] + (minaf[index + 43] / nPerceptionSamples)
        self.auditoryResult[3] = self.auditoryResult[3] / nPerceptionSamples
        self.auditoryResult[4] = self.auditoryResult[4] / nPerceptionSamples
        self.auditoryResult[5] = self.auditoryResult[5] / nPerceptionSamples

        self.somato_out = 0.0
        if ((proprioceptiveAv[0] < 0.0) or (proprioceptiveAv[1] < 0.0)):
            self.somato_out = 1.0
        self.sensor_out = self.auditoryResult

    def executeMotorCommand(self):
        self.getMotorDynamics()
        self.vocalize()

    def plotArticulatoryEvolution(self, arts):
        for index in range(len(arts)):
            plt.plot(self.time, self.art_states[:, arts[index] - 1])
            plt.hold(True)
        plt.show()

    def plotAuditoryOutput(self, audOut):
        for index in range(len(audOut)):
            plt.plot(self.time, self.auditory_states[:, audOut[index] - 1])
            plt.hold(True)
        plt.show()

    def plotsomato_out(self):
        plt.plot(self.time, self.somato_out)
        plt.show

    def getVocaltractShape(self, art_states, returnShape=0):
        auditory_states = self.synth.get_audsom(self.art_states,scale=True)
        tr_,tr__,outline,tr___ = auditory_states
        if returnShape:
            return outline
        else:
            self.vocalTractshape = outline

    def plotVocalTractShape(self, time, plot=1):
        self.getVocaltractShape(self.art_states)
        ts = self.ts
        index = np.round(time / ts)
        figVocalTract = plt.figure()
        axVocalTract = figVocalTract.add_subplot(111, autoscale_on=False, xlim=(-75, 225), ylim=(-200, 100))
        axVocalTract.plot(np.real(self.vocalTractshape[:, index]), np.imag(self.vocalTractshape[:, index]))
        figVocalTract.show()
        '''try:
            input('Press any key to continue...')
        except:
            pass'''

    def drawVocalTractShape_index(self, index, ax):
        ax.plot(np.real(self.vocalTractshape[:, index]), np.imag(self.vocalTractshape[:, index]))
        '''try:
            input('Press any key to continue...')
        except:
            pass'''

    def getVocalizationVideo(self, show=0, file_name='vt', keep_audio=0):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=1 / 0.005, metadata=dict(artist='Juan Manuel Acevedo Valle'))
        figVocalTract = plt.figure()

        art_states = self.getSoundWave(save=1, returnArtStates=1, file_name=file_name)

        outline = self.getVocaltractShape(art_states, returnShape=1)
        nSamples = art_states.shape[0]
        print(nSamples)
        sequence = []
        for index in range(nSamples):
            sequence.append((plt.plot(np.real(outline[:, index]), np.imag(outline[:, index]))))
        im_ani = animation.ArtistAnimation(figVocalTract, sequence, interval=0.005, repeat=False, blit=True)
        im_ani.save(file_name + '.mp4', writer=writer, codec="libx264")
        command = ["ffmpeg",
                   '-i', file_name + '.wav',
                   '-i', file_name + '.mp4',
                   '-strict', '-2',
                   '-c:v', "libx264", file_name + '_audio.mp4']
        sp.call(command)
        if keep_audio == 0:
            command = ["rm",
                       file_name + '.wav',
                       file_name + '.mp4']
            sp.call(command)
        if (show):
            figVocalTract.show()

    def getVideo(self, art_states, show=0, file_name='vt', keep_audio=0):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=1 / 0.005, metadata=dict(artist='Juan Manuel Acevedo Valle'))
        figVocalTract = plt.figure()

        self.get_sound_wave(art_states, save=1, file_name=file_name)

        outline = self.getVocaltractShape(art_states, returnShape=1)
        nSamples = art_states.shape[0]

        print(nSamples)
        sequence = []
        for index in range(nSamples):
            sequence.append((plt.plot(np.real(outline[:, index]), np.imag(outline[:, index]))))
        im_ani = animation.ArtistAnimation(figVocalTract, sequence, interval=0.005, repeat=False, blit=True)
        im_ani.save(file_name + '.mp4', writer=writer, codec="libx264")
        command = ["ffmpeg",
                   '-i', file_name + '.wav',
                   '-i', file_name + '.mp4',
                   '-c:v', "libx264", '-strict', '-2', file_name + '_audio.mp4']
        sp.call(command)
        if keep_audio == 0:
            command = ["rm",
                       file_name + '.wav',
                       file_name + '.mp4']
            sp.call(command)
        if (show):
            figVocalTract.show()
        else:
            del figVocalTract

    def get_sound_wave(self, art_states, play=0, save=0, file_name='vt'):  # based on explauto
        self.soundWave = self.synth.get_sound(art_states)
        if (play):
            self.playSoundWave()
        if (save):
            scaled = np.int16(self.soundWave / np.max(np.abs(self.soundWave)) * 32767)
            write(file_name + '.wav', 11025, scaled)

    def getSoundWave(self, play=0, save=0, returnArtStates=0, file_name='vt'):  # based on explauto
        art_states = self.getMotorDynamics(sound=1)
        self.soundWave = self.synth.get_sound(art_states)
        if (play):
            self.playSoundWave()
        if (save):
            scaled = np.int16(self.soundWave / np.max(np.abs(self.soundWave)) * 32767)
            write(file_name + '.wav', 11025, scaled)
        if (returnArtStates):
            return art_states

    def plotSoundWave(self, ax):
        ax.plot(np.float128(xrange(0, len(self.soundWave))) * self.ts, self.soundWave)

    def playSoundWave(self):  # keep in mind that DivaMatlab works with ts=0.005
        import pyaudio
        self.pa = pyaudio.PyAudio()  # If pa and stream are not elements of the self object then sound does not play
        self.stream = self.pa.open(format=pyaudio.paFloat32,
                                   channels=1,
                                   rate=11025,
                                   output=True)
        self.stream.start_stream()
        self.stream.write(self.soundWave.astype(np.float32).tostring())
        self.stream.close()

    def releaseAudioDevice(self):  # any sound in the buffer will be removed
        try:
            self.pa.terminate()
        except:
            pass

    def interactiveSystem(self):
        ### Main window container
        self.root_window = tk.Tk()
        self.root_window.geometry("800x800")
        self.root_window.title("Diva Agent")

        self.root_frame = tk.Frame(self.root_window, width=800, height=800, bg="green")
        self.root_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        ## Vocal tract and sound container
        self.guiVTPanel()

        ### Motor commands container   
        self.guiMotorPanel()
        self.guiMotorPanel_reset_callback()

        self.root_window.mainloop()

    def execute_callback(self):
        motor_command = self.motor_command
        motor_command[0] = np.float128(self.entry_m1.get())
        motor_command[1] = np.float128(self.entry_m2.get())
        motor_command[2] = np.float128(self.entry_m3.get())
        motor_command[3] = np.float128(self.entry_m4.get())
        motor_command[4] = np.float128(self.entry_m5.get())
        motor_command[5] = np.float128(self.entry_m6.get())
        motor_command[6] = np.float128(self.entry_m7.get())
        motor_command[7] = np.float128(self.entry_m8.get())
        motor_command[8] = np.float128(self.entry_m9.get())
        motor_command[9] = np.float128(self.entry_m10.get())
        motor_command[10] = np.float128(self.entry_m11.get())
        motor_command[11] = np.float128(self.entry_m12.get())
        motor_command[12] = np.float128(self.entry_m13.get())

        self.set_action(motor_command)
        self.executeMotorCommand()

        self.getVocaltractShape(self.art_states)
        self.vt_shape_index_max = self.vocalTractshape.shape[1]
        self.vt_shape_current = 0
        self.sv_current.set(str(self.vt_shape_current))

        self.drawVocalTractShape_index(self.vt_shape_current, self.ax_vt)
        self.canvas_vt.draw()

        self.getSoundWave(play=0, save=0, returnArtStates=0, file_name='vt')
        self.plotSoundWave(self.ax_sound)
        self.canvas_sound.draw()

        # --------------------------- self.prev_next_vt.config(state = tk.NORMAL)
        self.btn_next_vt.config(state="normal")
        self.btn_play_vt.config(state="normal")
        self.entry_current_vt.config(state="normal")
        self.entry_step_vt.config(state="normal")

    def vt_shape_prev_callback(self):
        self.btn_next_vt.config(state=tk.NORMAL)

        self.vt_shape_current = self.vt_shape_current - self.vt_shape_step
        self.drawVocalTractShape_index(self.vt_shape_current, self.ax_vt)
        self.canvas_vt.draw()
        self.sv_current.set(str(self.vt_shape_current))

        if self.vt_shape_current - self.vt_shape_step < 0:
            self.btn_prev_vt.config(state=tk.DISABLED)

    def vt_shape_next_callback(self):
        self.btn_prev_vt.config(state=tk.NORMAL)

        self.vt_shape_current = self.vt_shape_current + self.vt_shape_step
        self.drawVocalTractShape_index(self.vt_shape_current, self.ax_vt)
        self.canvas_vt.draw()
        self.sv_current.set(str(self.vt_shape_current))

        if self.vt_shape_current + self.vt_shape_step >= self.vt_shape_index_max:
            self.btn_next_vt.config(state=tk.DISABLED)

    def play_callback(self):
        self.playSoundWave()

    def set_vt_opts_callback(self):
        try:
            self.vt_shape_step = np.int(self.sv_step.get())
            if self.vt_shape_current + self.vt_shape_step >= self.vt_shape_index_max:
                self.btn_next_vt.config(state=tk.DISABLED)
            if self.vt_shape_current - self.vt_shape_step < 0:
                self.btn_prev_vt.config(state=tk.DISABLED)
        except:
            pass

    def set_vt_current_callback(self):
        try:
            self.vt_shape_current = np.int(self.sv_current.get())
            self.drawVocalTractShape_index(self.vt_shape_current, self.ax_vt)
            self.canvas_vt.draw()
        except:
            pass

    def guiMotorPanel_reset_callback(self):
        self.entry_m1.insert(0, "0.2")
        self.entry_m2.insert(0, "0.2")
        self.entry_m3.insert(0, "0.2")
        self.entry_m4.insert(0, "0.2")
        self.entry_m5.insert(0, "0.2")
        self.entry_m6.insert(0, "0.2")
        self.entry_m7.insert(0, "0.2")
        self.entry_m8.insert(0, "0.2")
        self.entry_m9.insert(0, "0.2")
        self.entry_m10.insert(0, "0.2")
        self.entry_m11.insert(0, "0.2")
        self.entry_m12.insert(0, "0.2")
        self.entry_m13.insert(0, "0.2")

    def guiVTPanel(self):
        self.vt_sound_frame = tk.Frame(self.root_frame, width=800, height=150, bg="white")
        self.vt_sound_frame.pack(side=tk.TOP, fill=tk.X, expand=1)

        self.vt_frame = tk.Frame(self.vt_sound_frame, width=120, height=150, bg="yellow")
        self.sound_frame = tk.Frame(self.vt_sound_frame, width=600, height=150, bg="black")
        self.vt_sound_opt_frame = tk.Frame(self.vt_sound_frame, bg="blue")
        self.vt_frame.pack(side=tk.LEFT, fill=tk.NONE, expand=0)
        self.sound_frame.pack(side=tk.LEFT, fill=tk.NONE, expand=0)
        self.vt_sound_opt_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        self.fig_vt = plt.Figure(figsize=(1.20, 1.5), dpi=100)
        self.fig_vt.patch.set_facecolor('white')
        self.canvas_vt = FigureCanvasTkAgg(self.fig_vt, master=self.vt_frame)
        self.canvas_vt.show()
        self.canvas_vt.get_tk_widget().pack(side="left", fill="none", expand=False)
        self.canvas_vt.get_tk_widget().configure(background='white', highlightcolor='white',
                                                 highlightbackground='white')
        self.ax_vt = self.fig_vt.add_subplot(111)
        self.ax_vt.spines['right'].set_visible(False)
        self.ax_vt.spines['top'].set_visible(False)
        self.ax_vt.spines['left'].set_visible(False)
        self.ax_vt.spines['bottom'].set_visible(False)
        self.ax_vt.xaxis.set_ticks_position('none')
        self.ax_vt.yaxis.set_ticks_position('none')
        self.ax_vt.xaxis.set_ticks([])
        self.ax_vt.yaxis.set_ticks([])

        self.canvas_vt.draw()

        self.fig_sound = plt.Figure(figsize=(6.00, 1.5), dpi=100)
        self.canvas_sound = FigureCanvasTkAgg(self.fig_sound, master=self.sound_frame)
        self.canvas_sound.show()
        self.canvas_sound.get_tk_widget().pack(side="left", fill="none", expand=False)
        self.ax_sound = self.fig_sound.add_subplot(111)
        pos1 = self.ax_sound.get_position()  # get the original position
        pos2 = [pos1.x0 * 0.9, pos1.y0 * 2.0, pos1.width * 1.1, pos1.height * 0.9]
        self.ax_sound.set_position(pos2)  # set a new position
        self.ax_sound.autoscale(enable=True, axis='both', tight=None)
        self.canvas_sound.draw()

        self.btn_prev_vt = tk.Button(self.vt_sound_opt_frame, state=tk.DISABLED, text="<<",
                                     command=self.vt_shape_prev_callback)
        self.btn_next_vt = tk.Button(self.vt_sound_opt_frame, state=tk.DISABLED, text=">>",
                                     command=self.vt_shape_next_callback)
        self.btn_play_vt = tk.Button(self.vt_sound_opt_frame, state=tk.DISABLED, text="Play",
                                     command=self.play_callback)
        self.btn_prev_vt.pack(side=tk.TOP, fill=tk.X, expand=1)
        self.btn_next_vt.pack(side=tk.TOP, fill=tk.X, expand=1)
        self.btn_play_vt.pack(side=tk.TOP, fill=tk.X, expand=1)

        self.sv_step = tk.StringVar()
        self.sv_step.set("10")
        self.vt_shape_step = np.int(self.sv_step.get())
        self.sv_step.trace("w", lambda name, index, mode, sv=self.sv_step: self.set_vt_opts_callback())

        self.sv_current = tk.StringVar()
        self.sv_current.set("0")
        self.vt_shape_current = np.int(self.sv_current.get())
        self.sv_current.trace("w", lambda name, index, mode, sv=self.sv_current: self.set_vt_current_callback())

        self.lbl_step_vt = tk.Label(self.vt_sound_opt_frame, text="Step")
        self.lbl_current_vt = tk.Label(self.vt_sound_opt_frame, text="Current")
        self.entry_step_vt = tk.Entry(self.vt_sound_opt_frame, state=tk.DISABLED, textvariable=self.sv_step, width=8)
        self.entry_current_vt = tk.Entry(self.vt_sound_opt_frame, state=tk.DISABLED, textvariable=self.sv_current,
                                         width=8)
        self.lbl_step_vt.pack(side=tk.TOP, fill=tk.X, expand=1)
        self.entry_step_vt.pack(side=tk.TOP, fill=tk.X, expand=1)
        self.lbl_current_vt.pack(side=tk.TOP, fill=tk.X, expand=1)
        self.entry_current_vt.pack(side=tk.TOP, fill=tk.X, expand=1)

        self.vt_shape_step = np.int(self.sv_step.get())
        self.vt_shape_current = np.int(self.sv_current.get())

    def guiMotorPanel(self):
        self.motor_frame = tk.Frame(self.root_frame, width=800, height=150, bg="white")
        self.motor_frame.pack(side=tk.TOP, fill=tk.X, expand=1)

        self.motor_entries_frame = tk.Frame(self.motor_frame, width=800, height=150, bg="white")
        self.motor_entries_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        self.lbl_m1 = tk.Label(self.motor_entries_frame, text="M1")
        self.lbl_m1.grid(row=0, padx=5, pady=5)
        self.lbl_m2 = tk.Label(self.motor_entries_frame, text="M2")
        self.lbl_m2.grid(row=1, padx=5, pady=5)
        self.lbl_m3 = tk.Label(self.motor_entries_frame, text="M3")
        self.lbl_m3.grid(row=2, padx=5, pady=5)
        self.lbl_m4 = tk.Label(self.motor_entries_frame, text="M4")
        self.lbl_m4.grid(row=3, padx=5, pady=5)

        self.entry_m1 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m1.grid(row=0, column=1, padx=5, pady=5)
        self.entry_m2 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m2.grid(row=1, column=1, padx=5, pady=5)
        self.entry_m3 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m3.grid(row=2, column=1, padx=5, pady=5)
        self.entry_m4 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m4.grid(row=3, column=1, padx=5, pady=5)

        self.lbl_m5 = tk.Label(self.motor_entries_frame, text="M5")
        self.lbl_m5.grid(row=4, column=0, padx=5, pady=5)
        self.lbl_m6 = tk.Label(self.motor_entries_frame, text="M6")
        self.lbl_m6.grid(row=0, column=2, padx=5, pady=5)
        self.lbl_m7 = tk.Label(self.motor_entries_frame, text="M7")
        self.lbl_m7.grid(row=1, column=2, padx=5, pady=5)
        self.lbl_m8 = tk.Label(self.motor_entries_frame, text="M8")
        self.lbl_m8.grid(row=2, column=2, padx=5, pady=5)

        self.entry_m5 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m5.grid(row=4, column=1, padx=5, pady=5)
        self.entry_m6 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m6.grid(row=0, column=3, padx=5, pady=5)
        self.entry_m7 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m7.grid(row=1, column=3, padx=5, pady=5)
        self.entry_m8 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m8.grid(row=2, column=3, padx=5, pady=5)

        self.lbl_m9 = tk.Label(self.motor_entries_frame, text="M9")
        self.lbl_m9.grid(row=3, column=2, padx=5, pady=5)
        self.lbl_m10 = tk.Label(self.motor_entries_frame, text="M10")
        self.lbl_m10.grid(row=4, column=2, padx=5, pady=5)
        self.lbl_m11 = tk.Label(self.motor_entries_frame, text="M11")
        self.lbl_m11.grid(row=1, column=4, padx=5, pady=5)
        self.lbl_m12 = tk.Label(self.motor_entries_frame, text="M12")
        self.lbl_m12.grid(row=2, column=4, padx=5, pady=5)
        self.lbl_m13 = tk.Label(self.motor_entries_frame, text="M13")
        self.lbl_m13.grid(row=3, column=4, padx=5, pady=5)

        self.entry_m9 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m9.grid(row=3, column=3, padx=5, pady=5)
        self.entry_m10 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m10.grid(row=4, column=3, padx=5, pady=5)
        self.entry_m11 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m11.grid(row=1, column=5, padx=5, pady=5)
        self.entry_m12 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m12.grid(row=2, column=5, padx=5, pady=5)
        self.entry_m13 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m13.grid(row=3, column=5, padx=5, pady=5)

        self.btn_execute_m = tk.Button(self.motor_frame, text="Execute", command=self.execute_callback)
        self.btn_execute_m.pack(side=tk.LEFT, fill=tk.NONE, expand=0)


class Instructor_(Diva2016b):
    def __init__(self):
        import os
        from ..DataManager.SimulationData import load_sim_h5
        Diva2016b.__init__(self)
        abs_path = os.path.dirname(os.path.abspath(__file__))
        self.instructor_file = abs_path + '/datasets/german_dataset_1.h5'

        self.data = load_sim_h5(self.instructor_file)
        self.n_units = len(self.data.sensor.data.index)
        self.unit_threshold = 0.1

    def interaction(self, sensor):
        dist = np.array(self.get_distances(sensor))
        min_idx = np.argmin(dist)
        self.min_idx = min_idx
        if dist[min_idx] <= self.unit_threshold:
            self.set_action(self.data.motor.data.iloc[min_idx])
            self.executeMotorCommand()
            return 1, self.sensor_out   #Analize implications of return the object itself
        tmp_rtn = np.empty((self.n_sensor,))
        tmp_rtn.fill(np.nan)
        return 0, tmp_rtn

    def get_distances(self, sensor):
        dist = []
        s_data = self.data.sensor.data.as_matrix()
        for i in range(self.n_units):
            dist += [np.linalg.norm(sensor - s_data[i, :])]
        return dist

    def change_dataset(self, file):
        import os
        from ..DataManager.SimulationData import load_sim_h5

        self.instructor_file = file

        self.data = load_sim_h5(self.instructor_file)
        self.n_units = len(self.data.sensor.data.index)

class Instructor(object):
    def __init__(self):
        import os
        from ..DataManager.SimulationData import load_sim_h5
        # ss = [[3., 0.5], [1.9, 1.25], [4.15, 2],[2.3, 3.4], [5.27, 6.23], [0.15, 8.7], [2.36, 7.46], [5.2, 8.87]]
        #     for s in ss:
        #         system.set_action(infer_motor(0, s))
        #         system.executeMotorCommand()
        #         data.appendData(system)
        abs_path = os.path.dirname(os.path.abspath(__file__))
        # self.instructor_file = abs_path + '/datasets/vowels_dataset_1.h5'
        self.instructor_file = abs_path + '/datasets/german_dataset_3.h5'

        self.name = 'diva2017a-memorydata'
        self.data = load_sim_h5(self.instructor_file)
        self.sensor_out = self.data.sensor.data.iloc[0].as_matrix()
        self.n_sensor = len(self.sensor_out)
        self.n_units = len(self.data.sensor.data.index)
        self.unit_threshold = 0.5

    def interaction(self, sensor):
        dist = np.array(self.get_distances(sensor))
        min_idx = np.argmin(dist)
        self.min_idx = min_idx
        if dist[min_idx] <= self.unit_threshold:
            out_tmp = self.data.sensor.data.iloc[min_idx].as_matrix()
            out = out_tmp.copy()
            return 1, out  # Analize implications of return the object itself
        tmp_rtn = np.empty((self.n_sensor,))
        tmp_rtn.fill(np.nan)
        return 0, tmp_rtn

    def get_distances(self, sensor):
        dist = []
        s_data = self.data.sensor.data.as_matrix()
        for i in range(self.n_units):
            dist += [np.linalg.norm(sensor - s_data[i, :])]
        return dist

    def change_dataset(self, file):
        import os
        from ..DataManager.SimulationData import load_sim_h5
        
        self.instructor_file = file

        self.data = load_sim_h5(self.instructor_file)
        self.n_units = len(self.data.sensor.data.index)


def motorDynamics(y, t, self, m):
    dumpingFactor = 1.01
    w0 = 2 * math.pi / 0.01

    dy1 = y[13]
    dy2 = y[14]
    dy3 = y[15]
    dy4 = y[16]
    dy5 = y[17]
    dy6 = y[18]
    dy7 = y[19]
    dy8 = y[20]
    dy9 = y[21]
    dy10 = y[22]
    dy11 = y[23]
    dy12 = y[24]
    dy13 = y[25]

    dy14 = -2 * dumpingFactor * w0 * y[13] - (pow(w0, 2)) * y[0] + (pow(w0, 2)) * m[0]
    dy15 = -2 * dumpingFactor * w0 * y[14] - (pow(w0, 2)) * y[1] + (pow(w0, 2)) * m[1]
    dy16 = -2 * dumpingFactor * w0 * y[15] - (pow(w0, 2)) * y[2] + (pow(w0, 2)) * m[2]
    dy17 = -2 * dumpingFactor * w0 * y[16] - (pow(w0, 2)) * y[3] + (pow(w0, 2)) * m[3]
    dy18 = -2 * dumpingFactor * w0 * y[17] - (pow(w0, 2)) * y[4] + (pow(w0, 2)) * m[4]
    dy19 = -2 * dumpingFactor * w0 * y[18] - (pow(w0, 2)) * y[5] + (pow(w0, 2)) * m[5]
    dy20 = -2 * dumpingFactor * w0 * y[19] - (pow(w0, 2)) * y[6] + (pow(w0, 2)) * m[6]
    dy21 = -2 * dumpingFactor * w0 * y[20] - (pow(w0, 2)) * y[7] + (pow(w0, 2)) * m[7]
    dy22 = -2 * dumpingFactor * w0 * y[21] - (pow(w0, 2)) * y[8] + (pow(w0, 2)) * m[8]
    dy23 = -2 * dumpingFactor * w0 * y[22] - (pow(w0, 2)) * y[9] + (pow(w0, 2)) * m[9]
    dy24 = -2 * dumpingFactor * w0 * y[23] - (pow(w0, 2)) * y[10] + (pow(w0, 2)) * m[10]
    dy25 = -2 * dumpingFactor * w0 * y[24] - (pow(w0, 2)) * y[11] + (pow(w0, 2)) * m[11]
    dy26 = -2 * dumpingFactor * w0 * y[25] - (pow(w0, 2)) * y[12] + (pow(w0, 2)) * m[12]

    return [dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8, dy9, dy10, dy11, dy12, dy13, dy14, dy15, dy16, dy17, dy18, dy19,
            dy20, dy21, dy22, dy23, dy24, dy25, dy26]
