"""
Created on March, 2017
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
import os, random
from exploration.data.data import load_sim_h5
import python_speech_features as psf

english_vowels = {'i': [296.0, 2241.0, 1.0], 'I': [396.0, 1839.0, 1.0], 'e': [532.0, 1656.0, 1.0],
                  'ae': [667.0, 1565.0, 1.0], 'A': [661.0, 1296.0, 1.0], 'a': [680.0, 1193.0, 1.0],
                  'b': [643.0, 1019.0, 1.0], 'c': [480.0, 857.0, 1.0], 'U': [395.0, 1408.0, 1.0],
                  'u': [386.0, 1587.0, 1.0], 'E': [519.0, 1408.0, 1.0]}

# Write down german vowels here
diva_output_scale = [100.0, 500.0, 1500.0, 3000.0]


class Diva2018():
    def __init__(self, sensori_out='formant', **kargs):

        motor_names = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13', 'M14', 'M15',
                       'M16', 'M17', 'M18', 'M19', 'M20', 'M21', 'M22', 'M23', 'M24', 'M25', 'M26']

        if sensori_out == 'formant':
            sensor_names = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
            n_sensor = len(sensor_names)

        somato_names = ['SS1', 'SS2', 'SS3', 'SS4', 'SS5', 'SS6', 'SS7', 'SS8', 'SS9', 'SS10', 'SS11', 'SS12', 'SS13',
                        'SS14', 'SS15',
                        'SS16']
        cons_names = ['P1']

        n_motor = 26

        n_somato = 16
        n_cons = 1

        outputScale = [100.0, 500.0, 1500.0, 3000.0]

        min_motor_values = np.array([-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -0.25, -0.25, -0.25] * 2)
        max_motor_values = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1] * 2)

        min_motor_values_init = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0] * 2)
        max_motor_values_init = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 2)

        if sensori_out == 'formant':
            min_sensor_values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            max_sensor_values = np.array([2.0, 2.0, 1.0, 2.0, 2.0, 1.0])

        min_somato_values = np.array([-1.0] * n_somato)
        max_somato_values = np.array([1.0] * n_somato)

        min_cons_values = np.array([0])
        max_cons_values = np.array([1])

        ts = 0.01

        name = 'Diva2018'
        self.name = name

        self.params = {'duration_m1': 0.4,
                       'duration_m2': 0.4,
                       'duration_s1': [0., 0.4],
                       'duration_s2': [0.4, 0.8],
                       'sensori_out': sensori_out,
                       'damping_factor': 1.01,
                       'w0': 2 * math.pi / 0.01,
                       'mfcc': {'winlen': 0.025, 'winstep': 0.01, 'samplerate': 11025, 'n_feat': 13}}

        self.ts = ts
        total_time = self.params['duration_m1'] + self.params['duration_m2']

        if sensori_out == 'mfcc':
            # n_sensor = int((total_time + self.params['mfcc']['winstep'])/(self.params['mfcc']['winlen']-self.params['mfcc']['winstep'])* self.params['mfcc']['n_feat'])
            n_sensor = int((total_time / self.params['mfcc']['winstep']) * self.params['mfcc']['n_feat'])
            # print(n_sensor)
            sensor_names = ['mffc_' + str(x) for x in range(n_sensor)]
            min_sensor_values = np.array([-10.0] * self.params['mfcc']['n_feat'])
            max_sensor_values = np.array([10.0] * self.params['mfcc']['n_feat'])

        self.time = linspace(0, total_time, int(total_time / ts) + 1)
        self.n_motor = n_motor
        self.n_sensor = n_sensor
        self.n_somato = n_somato
        self.n_cons = n_cons

        self.motor_names = motor_names
        self.sensor_names = sensor_names
        self.somato_names = somato_names
        self.cons_names = cons_names

        self.min_motor_values = min_motor_values
        self.max_motor_values = max_motor_values
        self.min_motor_values_init = min_motor_values_init
        self.max_motor_values_init = max_motor_values_init
        self.min_sensor_values = min_sensor_values
        self.max_sensor_values = max_sensor_values
        self.min_somato_values = min_somato_values
        self.max_somato_values = max_somato_values
        self.min_cons_values = min_cons_values
        self.max_cons_values = max_cons_values

        self.motor_command = np.array([0.0] * n_motor)
        self.sensor_out = np.array([0.0] * n_sensor)
        self.sensor_goal = np.array([0.0] * n_sensor)
        self.somato_out = np.array([0.0] * n_somato)
        self.somato_goal = np.array([0.0] * n_somato)
        self.cons_out = 0.0  # Constraint violations flag
        self.cons_threshold = 0.5
        self.competence_result = 0.0
        self.sensor_instructor = np.empty((self.n_sensor,))
        self.sensor_instructor.fill(np.nan)
        self.synth = Diva()

    def set_params(self, **kwargs):
        for key in kwargs.keys():
            self.params[key] = kwargs[key]
        total_time = self.params['duration_m1'] + self.params['duration_m2']
        self.time = linspace(0, total_time, int(total_time / self.ts) + 1)

    def set_action(self, motor_command):
        self.motor_command = motor_command

    def execute_action(self):
        if self.params['sensori_out'] == 'formant':
            sound = 0
            self.get_motor_dynamics(sound=sound)
            self.vocalize()
        elif self.params['sensori_out'] == 'mfcc':
            sound = 0
            self.get_motor_dynamics(sound=sound)
            self.vocalize()
            self.get_sound()
            sound_wave = self.sound_wave.copy()
            sound_mfcc = psf.mfcc(sound_wave, samplerate=self.params['mfcc']['samplerate'],
                                  winlen=self.params['mfcc']['winlen'],
                                  winstep=self.params['mfcc']['winstep'])
            self.sensor_out = np.array(sound_mfcc).flatten()
        else:
            raise ValueError

    def get_motor_dynamics(self,
                           sound=0):

        duration_m1 = self.params['duration_m1']
        duration_m2 = self.params['duration_m2']

        damping_factor = self.params['damping_factor']
        w0 = self.params['w0']

        if sound:
            ts = 0.005
        else:
            ts = self.ts

        duration = duration_m1 + duration_m2
        nSamples = int(duration / ts + 1)
        nSamples1 = int(duration_m1 / ts) + 1
        nSamples2 = int(duration_m2 / ts) + 1
        y_neutral = [0.0] * 13
        y_neutral[11] = 0
        y_neutral[12] = 0
        y0 = [0.0] * 26
        y0[:13] = y_neutral
        m1 = self.motor_command[:13]
        t1 = linspace(0.0, duration_m1, nSamples1)
        art_states1 = odeint(motor_dynamics, y0, t1, args=(self, m1,
                                                           damping_factor,
                                                           w0))
        t2 = linspace(0.0, duration_m2, nSamples2)
        m2 = self.motor_command[13:]
        art_states2 = odeint(motor_dynamics, art_states1[-1, :], t2, args=(self, m2,
                                                                           damping_factor,
                                                                           w0))

        self.art_states = np.zeros((nSamples, 26))
        self.art_states[:nSamples1, :] = art_states1
        self.art_states[nSamples1 - 1:, :] = art_states2

    def vocalize(self):
        ts = self.ts
        # perceptionWindowDuration = np.min([self.params['duration_m1'],self.params['duration_m1']])-0.02
        # perceptionTime = linspace(ts, perceptionWindowDuration, int(perceptionWindowDuration / ts))
        # n_percep_samples = (len(perceptionTime))

        cons_av = [0.0] * 2

        auditory_states, somato_states, tr__, af = self.synth.get_audsom(self.art_states[:, 0:13], scale=True)

        self.auditory_states = auditory_states
        self.somato_states = somato_states
        self.af = af
        minaf = np.array([np.min(x) for x in af])
        self.cons_states = minaf

        self.sensor_out = 0.0 * self.sensor_out
        self.somato_out = 0.0 * self.somato_out

        # time0_pw1 = self.params['duration_s1'][0]
        # time1_pw1 = self.params['duration_s1'][1]
        # time0_pw2 = self.params['duration_s2'][0]
        # time1_pw2 = self.params['duration_s2'][1]
        # time_pw1 = linspace(time0_pw1, time1_pw1, int((time1_pw1 - time0_pw1)/ ts))
        # n_pw1 = len(time_pw1)
        # time_pw2 = linspace(time0_pw2, time1_pw2, int((time1_pw2 - time0_pw2) / ts))
        # n_pw2 = len(time_pw2)

        sample0_pw1 = int(np.floor(self.params['duration_s1'][0] / ts))
        sample1_pw1 = int(np.floor(self.params['duration_s1'][1] / ts))
        sample0_pw2 = int(np.floor(self.params['duration_s2'][0] / ts))
        sample1_pw2 = int(np.floor(self.params['duration_s2'][1] / ts))
        samples_pw1 = range(sample0_pw1, sample1_pw1)
        n_pw1 = len(samples_pw1)
        samples_pw2 = range(sample0_pw2, sample1_pw2)
        n_pw2 = len(samples_pw2)

        # idx_pw1 = 2
        # idx_pw2 = int(self.params['duration_m1']/ts) + 3   # +2 or +3???
        # First perception time window
        for index in range(len(minaf)):
            if not ((self.art_states[index, 11] > 0) and (self.art_states[index, 12] > 0) and (minaf[index] > 0)):
                auditory_states[index, 0] *= 0
                auditory_states[index, 1] *= 0
                auditory_states[index, 2] *= 0
                auditory_states[index, 3] *= 0

        for index in samples_pw1:
            if (self.art_states[index, 11] > 0) and (self.art_states[index, 12] > 0) and (minaf[index] > 0):
                self.sensor_out[0] = self.sensor_out[0] + auditory_states[index, 1]
                self.sensor_out[1] = self.sensor_out[1] + auditory_states[index, 2]
                self.sensor_out[2] = self.sensor_out[2] + 1.0
            for jj in range(8):
                self.somato_out[jj] += somato_states[index + 2, jj] / n_pw1
            cons_av[0] = cons_av[0] + (minaf[index] / n_pw1)
        self.sensor_out[0] = self.sensor_out[0] / n_pw1
        self.sensor_out[1] = self.sensor_out[1] / n_pw1
        self.sensor_out[2] = self.sensor_out[2] / n_pw1

        # Second perception time window
        for index in samples_pw2:
            # print(index)
            if (self.art_states[index, 11] > 0) and (self.art_states[index, 12] > 0) and (
                        minaf[index] > 0):
                self.sensor_out[3] = self.sensor_out[3] + auditory_states[index, 1]
                self.sensor_out[4] = self.sensor_out[4] + auditory_states[index, 2]
                self.sensor_out[5] = self.sensor_out[5] + 1.0
            for jj in range(8):
                self.somato_out[jj + 8] += somato_states[index, jj] / n_pw2
            cons_av[1] = cons_av[1] + (minaf[index] / n_pw2)
        self.sensor_out[3] = self.sensor_out[3] / n_pw2
        self.sensor_out[4] = self.sensor_out[4] / n_pw2
        self.sensor_out[5] = self.sensor_out[5] / n_pw2

        self.cons_out = 0.0
        if ((cons_av[0] < 0.0) or (cons_av[1] < 0.0)):
            self.cons_out = 1.0

    def plot_arts_evo(self, arts_idx, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1, 1)
        plt.sca(axes)
        for index in range(len(arts_idx)):
            plt.plot(self.time, self.art_states[:, arts_idx[index]])

    def plot_aud_out(self, aud_out, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1, 1)
        plt.sca(axes)
        for index in range(len(aud_out)):
            plt.plot(self.time, self.auditory_states[:, aud_out[index]])

    def plot_som_out(self, som_idx, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1, 1)
        plt.sca(axes)
        for index in range(len(som_idx)):
            plt.plot(self.time, self.somato_states[:, som_idx[index]])

    def plot_cons_out(self, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1, 1)
        plt.sca(axes)
        plt.plot(self.time, self.cons_states)

    def get_vt_shape(self, sound=False):
        self.get_motor_dynamics(sound=sound)
        a, b, outline, af = self.synth.get_audsom(self.art_states[:, 0:13])
        self.vt_shape = outline
        self.af = af

    def plot_vt_shape(self, index=-1, time=None, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1, 1)
        plt.sca(axes)
        self.get_vt_shape()
        if time is not None:
            ts = self.ts
            index = int(np.floor(time / ts))
        plt.plot(np.real(self.vt_shape[index]), np.imag(self.vt_shape[index]),lw=2.0)
        plt.axis('off')


    def plot_af_shape(self, index=-1, time=None, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1, 1)
        plt.sca(axes)
        self.get_vt_shape()
        if time is not None:
            ts = self.ts
            index = int(np.floor(time / ts))
        plt.plot(self.af[index],lw=2.0)
        plt.xlabel("Index",weight='bold')
        plt.ylabel('Area Function ($a_f$)',weight='bold')
        axes.xaxis.set_ticks(np.arange(0,len(self.af[index])+29,30))

    def get_video(self, show=0, file_name='vt', keep_audio=0):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=1 / 0.005, metadata=dict(artist='Juan Manuel Acevedo Valle'))
        figVocalTract = plt.figure()

        self.get_sound(save=1, file_name=file_name)
        self.get_vt_shape()
        nSamples = self.art_states.shape[0]
        # print(nSamples)
        sequence = []
        for index in range(nSamples):
            sequence += [plt.plot(np.real(self.vt_shape[index]), np.imag(self.vt_shape[index]))]
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
        else:
            plt.close(figVocalTract)

    def get_sound(self, play=0, save=0, file_name='vt'):  # based on explauto
        self.get_motor_dynamics(sound=True)
        self.sound_wave = self.synth.get_sound(self.art_states[:, 0:13])
        if (play):
            self.playSoundWave()
        if (save):
            scaled = np.int16(self.sound_wave / np.max(np.abs(self.sound_wave)) * 32767)
            write(file_name + '.wav', 11025, scaled)

    def plot_sound(self, axes=None):
        self.get_sound()
        if axes is None:
            fig, axes = plt.subplots(1, 1)
        plt.sca(axes)
        duration = self.params['duration_m1'] + self.params['duration_m2']
        plt.plot(np.array(range(0, len(self.sound_wave))).astype(np.float) * duration / len(self.sound_wave),
                 self.sound_wave)

    def playSoundWave(self):  # keep in mind that DivaMatlab works with ts=0.005
        import pyaudio
        self.pa = pyaudio.PyAudio()  # If pa and stream are not elements of the self object then sound does not play
        self.stream = self.pa.open(format=pyaudio.paFloat32,
                                   channels=1,
                                   rate=11025,
                                   output=True)
        self.stream.start_stream()
        self.stream.write(self.sound_wave.astype(np.float32).tostring())
        self.stream.close()

    def releaseAudioDevice(self):  # any sound in the buffer will be removed
        try:
            self.pa.terminate()
        except:
            pass

    def generate_log(self):
        log = 'system: ' + self.name + '\n'
        for key in self.params.keys():
            try:
                attr_log = self.params[key].generate_log()
                log += key + ': {\n'
                log += attr_log
                log += '}\n'
                log = log.replace('\n}', '}')
            except IndexError:
                print("INDEX ERROR in Divapy2017a log generation")
            except AttributeError:
                if isinstance(self.params[key], dict):
                    log += key + ': {\n'
                    for key_ in self.params[key].keys():
                        log += key_ + ': ' + str(self.params[key][key_]) + ','
                    log += ('}\n')
                    log = log.replace(',}', '}')
                else:
                    log += key + ': ' + str(self.params[key]) + '\n'
        return log


class Instructor():
    def __init__(self, n_su=None, slope=1.):  # n_su -> n_sensor_units
        abs_path = os.path.dirname(os.path.abspath(__file__))
        # self.instructor_file = abs_path + '/datasets/vowels_dataset_1.h5'
        self.instructor_file = abs_path + '/datasets/german_dataset_somato.h5'
        self.instructor_file = self.instructor_file.replace('/', os.sep)

        self.name = 'diva2017a-memorydata'
        self.data, sys = load_sim_h5(self.instructor_file)
        n_samples = len(self.data.sensor.data.iloc[:])
        self.idx_sensor = range(n_samples)
        self.slope = slope
        self.n_su = n_samples
        if n_su is not None:
            self.n_su = n_su
            random.seed(1234)  # To gurantee that the first samples are going to be equal in any subset
            self.idx_sensor = random.sample(range(n_samples), n_su)
            self.data = self.data.get_samples(sys, self.idx_sensor)
        self.sensor_out = 0. * self.data.sensor.data.iloc[0].as_matrix()
        self.n_sensor = len(self.sensor_out)
        self.n_units = len(self.data.sensor.data.index)

        self.threshold = 0.5
        self.unit_threshold = self.threshold * np.ones((self.n_su,))

    def generate_log(self):
        params_to_logs = ['instructo_file', 'n_units', 'idx_sensor', 'slope', 'threshold']
        log = 'instructor: ' + self.name + '\n'
        for attr_ in params_to_logs:
            if hasattr(self, attr_):
                try:
                    attr_log = getattr(self, attr_).generate_log()
                    log += attr_ + ': {\n'
                    log += attr_log
                    log += '}\n'
                    log = log.replace('\n}', '}')
                except IndexError:
                    print("INDEX ERROR in Diva2017aInstructor log generation")
                except AttributeError:
                    if isinstance(getattr(self, attr_), dict):
                        log += attr_ + ': {\n'
                        for key in attr_.keys():
                            log += key + ': ' + str(attr_[key]) + ','
                        log += ('}\n')
                        log = log.replace(',}', '}')
                    else:
                        log += attr_ + ': ' + str(getattr(self, attr_)) + '\n'
        return log

    def interaction(self, sensor):
        dist = np.array(self.get_distances(sensor))
        min_idx = np.argmin(dist)
        self.min_idx = min_idx
        if dist[min_idx] <= self.unit_threshold[min_idx]:
            out_tmp = self.data.sensor.data.iloc[min_idx].as_matrix()
            out = out_tmp.copy()
            self.unit_threshold[min_idx] = self.slope * self.unit_threshold[min_idx]
            return 1, out  # Analize implications of return the object itself
        tmp_rtn = np.empty((self.n_sensor,))
        tmp_rtn.fill(np.nan)
        return 0, tmp_rtn

    def get_distances(self, sensor):
        dist = []
        s_data = self.data.sensor.data.as_matrix()  # .iloc[self.idx_sensor].as_matrix()
        for i in range(self.n_su):
            dist += [np.linalg.norm(sensor - s_data[i, :])]
        return dist

    def change_dataset(self, file):
        import os
        from ..data.data import load_sim_h5

        self.instructor_file = file

        self.data = load_sim_h5(self.instructor_file)
        self.n_units = len(self.data.sensor.data.index)


def motor_dynamics(y, t, self, m, damping_factor, w0):
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

    dy14 = -2 * damping_factor * w0 * y[13] - (pow(w0, 2)) * y[0] + (pow(w0, 2)) * m[0]
    dy15 = -2 * damping_factor * w0 * y[14] - (pow(w0, 2)) * y[1] + (pow(w0, 2)) * m[1]
    dy16 = -2 * damping_factor * w0 * y[15] - (pow(w0, 2)) * y[2] + (pow(w0, 2)) * m[2]
    dy17 = -2 * damping_factor * w0 * y[16] - (pow(w0, 2)) * y[3] + (pow(w0, 2)) * m[3]
    dy18 = -2 * damping_factor * w0 * y[17] - (pow(w0, 2)) * y[4] + (pow(w0, 2)) * m[4]
    dy19 = -2 * damping_factor * w0 * y[18] - (pow(w0, 2)) * y[5] + (pow(w0, 2)) * m[5]
    dy20 = -2 * damping_factor * w0 * y[19] - (pow(w0, 2)) * y[6] + (pow(w0, 2)) * m[6]
    dy21 = -2 * damping_factor * w0 * y[20] - (pow(w0, 2)) * y[7] + (pow(w0, 2)) * m[7]
    dy22 = -2 * damping_factor * w0 * y[21] - (pow(w0, 2)) * y[8] + (pow(w0, 2)) * m[8]
    dy23 = -2 * damping_factor * w0 * y[22] - (pow(w0, 2)) * y[9] + (pow(w0, 2)) * m[9]
    dy24 = -2 * damping_factor * w0 * y[23] - (pow(w0, 2)) * y[10] + (pow(w0, 2)) * m[10]
    dy25 = -2 * damping_factor * w0 * y[24] - (pow(w0, 2)) * y[11] + (pow(w0, 2)) * m[11]
    dy26 = -2 * damping_factor * w0 * y[25] - (pow(w0, 2)) * y[12] + (pow(w0, 2)) * m[12]

    return [dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8, dy9, dy10, dy11, dy12, dy13, dy14, dy15, dy16, dy17, dy18, dy19,
            dy20, dy21, dy22, dy23, dy24, dy25, dy26]
