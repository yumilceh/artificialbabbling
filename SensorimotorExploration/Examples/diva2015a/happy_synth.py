"""
Created on March 3, 2017

@author: Juan Manuel Acevedo Valle

This script could be sused to obtain audio and video of a custom series of articulatory configurations
"""

if __name__ == '__main__':
    import os
    import pymatlab
    import matplotlib.pyplot as plt
    from numpy import array as arr
    from SensorimotorExploration.Systems.Diva2015a import Diva_Proprio2015a as Diva
    from scipy.io import loadmat as loadmat

    diva_system = Diva()
    abs_path = os.path.dirname(os.path.abspath(__file__))
    arts_file = abs_path + '/../../Systems/DIVA/hello.mat'
    file_name = 'vt_happy'

    arts = loadmat(arts_file)['arts']

    diva_system.getVideo(arts, file_name=file_name)

    diva_system.releaseAudioDevice()


    # abs_path = os.path.dirname(os.path.abspath(__file__))
    #
    # self.diva_synth_vt_file = abs_path + '/DIVA/vt_py.mat'
    # self.diva_synth_fmfit_file = abs_path + '/DIVA/fmfit_py.mat'
    #
    # self.diva_hanning_file = abs_path + '/DIVA/hanning.mat'
    # self.hanning = loadmat(self.diva_hanning_file)['h']
    #
    # vt = loadmat(self.diva_synth_vt_file)
    # fmfit = loadmat(self.diva_synth_fmfit_file)
    #
    # keys = ['vt_scale', 'vt_base', 'vt_average', 'vt_box']
    # for key in keys:
    #     vt[key] = array(vt[key])
    # keys = ['fmfit_beta_fmt', 'fmfit_p', 'fmfit_beta_som', 'fmfit_mu', 'fmfit_isigma']
    # for key in keys:
    #     fmfit[key] = array(fmfit[key])
    # self.vt = vt