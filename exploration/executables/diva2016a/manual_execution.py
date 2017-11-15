"""
Created on Feb 5, 2016

@author: yumilceh
"""
from numpy import random as np_rnd
import os, sys, random

if __name__ == '__main__':
    # Adding libraries##
    from exploration.systems.Diva2016a import Diva2016a as System

    random_seed = 1234
    n_experiments = 200

    # To guarantee reproductible experiments##
    random.seed(random_seed)
    np_rnd.seed(random_seed)

    # Creating Agent ##
    system = System()

    # Running interactive simulation
    file_prefix = 'Manual_Simulation'

    system.interactiveSystem()
