'''
Created on May 23, 2016

@author: Juan Manuel Acevedo Valle
'''
import shutil
import gzip
import tarfile
import os

def saveSimulationData(in_file_names,out_file_name):
    ''' Files to be storaged must be gzip'''
    tar=tarfile.open(out_file_name,"w:gz")
    for data_file in in_file_names:
        tar.add(data_file)
        os.remove(data_file)
    tar.close()
