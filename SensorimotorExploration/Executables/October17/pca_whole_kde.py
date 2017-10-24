'''
Created on Oct 17, 2016

@author: Juan Manuel Acevedo Valle
''' 
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import h5py, os, sys, random
from scipy.stats import gaussian_kde
from scipy import stats
from scipy.stats.distributions import norm    
from scipy.io import savemat

if __name__ == '__main__':
    
    this_dir =  os.getcwd()
    sys.path.append("../../")
    from DataManager.PlotTools import initializeFigure
    #------------------------- from Models.GeneralModels.ILGMM_GREC import ILGMM
    
    dist_used = 'euclidean'
    n_rnd_samples = 5000
    n_permutations = 30
    n_iterations = 10
    random.seed(1234)
    np.random.seed(1234)
    os.chdir('../ExperimentsIEEETCDS2016/')
    
    ####-------------------------------------------------####
    ####                    PCA
    ####-------------------------------------------------####
    proprio_criteria = 2   # 1 for considering only non contacts and 2 to consider all
    directories = ['EVD_no_Proprio_0/',
                   'EVD_no_Proprio_1/',
                   'EVD_no_Proprio_2/',
                   'EVD_no_Proprio_3/',
                   'EVD_no_Proprio_4/',
                   'EVD_no_Proprio_6/',
                   'EVD_no_Proprio_7/',
                   'EVD_no_Proprio_8/',
                   'EVD_no_Proprio_9/',
                   'EVD_Proprio_0/',
                   'EVD_Proprio_1/',
                   'EVD_Proprio_2/',
                   'EVD_Proprio_3/',
                   'EVD_Proprio_4/',
                   'EVD_Proprio_6/',
                   'EVD_Proprio_7/',
                   'EVD_Proprio_8/',
                   'EVD_Proprio_9/',
                   'Special_EVD_Proprio_5/EVD_no_Proprio_5/',
                   'Special_EVD_Proprio_5/EVD_Proprio_5/']

    print("Loading data for PCA...")
    sensor_data = None
    for i in range(len(directories)): 
        directory = directories[i]
        mat = h5py.File(directory + 'SMdata.mat','r')
        data = np.array(mat.get('SMdata'))
        mat = h5py.File(directory + 'PRdata.mat','r')
        proprio_data = np.array(mat.get('PRdata'))
        proprio_data = proprio_data[[-1],:]
        try:
            sensor_data_tmp = np.transpose(data[[0,1,3,4],:])
            sensor_data_tmp = sensor_data_tmp[np.where(proprio_data  < proprio_criteria)[1],:]
            sensor_data = np.append(sensor_data,sensor_data_tmp,axis=0)
        except:
            sensor_data = np.transpose(data[[0,1,3,4],:])
            sensor_data = sensor_data[np.where(proprio_data < proprio_criteria)[1],:]

    pca =PCA(n_components=1)      
    pca.fit(sensor_data)
    print("The shape of the data for PCA is:")
    print(sensor_data.shape)
    print("Variance contribution per per principal axes: ")
    print(str(pca.explained_variance_))    
    print("% Variance contribution per per principal axes: ")
    print( str(pca.explained_variance_ratio_))    
    print("Principal directions [n_components_n_features]:")
    print(pca.components_)
    
    del(data)
    del(sensor_data)
    del(sensor_data_tmp)
    
    ####-------------------------------------------------####
    ####             Plot non-proprio distributions
    ####-------------------------------------------------####  
       
    #------------- print("Loading Data to compute Gaussian KDE (No Porprio)...")
    #--------------------------------------- directories = ['EVD_no_Proprio_0/',
               #------------------------------------------- 'EVD_no_Proprio_1/',
               #------------------------------------------- 'EVD_no_Proprio_2/',
               #------------------------------------------- 'EVD_no_Proprio_3/',
               #------------------------------------------- 'EVD_no_Proprio_4/',
               #------------------------------------------- 'EVD_no_Proprio_6/',
               #------------------------------------------- 'EVD_no_Proprio_7/',
               #------------------------------------------- 'EVD_no_Proprio_8/',
               #------------------------------------------- 'EVD_no_Proprio_9/']
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
    #--------------------------------------------- sensor_data_no_proprio = None
    #----------------------------------------- for i in range(len(directories)):
        #-------------------------------------------- directory = directories[i]
        #------------------------- mat = h5py.File(directory + 'SMdata.mat','r')
        #------------------------------------ data = np.array(mat.get('SMdata'))
        #------------------------- mat = h5py.File(directory + 'PRdata.mat','r')
        #---------------------------- proprio_data = np.array(mat.get('PRdata'))
        #----------------------------------- proprio_data = proprio_data[[-1],:]
        #------------------------------------------------------------------ try:
            #----------------- sensor_data_tmp = np.transpose(data[[0,1,3,4],:])
            # sensor_data_tmp = sensor_data_tmp[np.where(proprio_data  < proprio_criteria)[1],:]
            # sensor_data_no_proprio = np.append(sensor_data_no_proprio,sensor_data_tmp,axis=0)
        #--------------------------------------------------------------- except:
            #---------- sensor_data_no_proprio = np.transpose(data[[0,1,3,4],:])
            # sensor_data_no_proprio = sensor_data_no_proprio[np.where(proprio_data < proprio_criteria)[1],:]
#------------------------------------------------------------------------------ 
    #---------------- print("Loading Data to compute Gaussian KDE (Porprio)...")
#------------------------------------------------------------------------------ 
    #------------------------------------------ directories = ['EVD_Proprio_0/',
               #---------------------------------------------- 'EVD_Proprio_1/',
               #---------------------------------------------- 'EVD_Proprio_2/',
               #---------------------------------------------- 'EVD_Proprio_3/',
               #---------------------------------------------- 'EVD_Proprio_4/',
               #---------------------------------------------- 'EVD_Proprio_6/',
               #---------------------------------------------- 'EVD_Proprio_7/',
               #---------------------------------------------- 'EVD_Proprio_8/',
               #---------------------------------------------- 'EVD_Proprio_9/']
    #------------------------------------------------ sensor_data_proprio = None
    #----------------------------------------- for i in range(len(directories)):
        #-------------------------------------------- directory = directories[i]
        #------------------------- mat = h5py.File(directory + 'SMdata.mat','r')
        #------------------------------------ data = np.array(mat.get('SMdata'))
        #------------------------- mat = h5py.File(directory + 'PRdata.mat','r')
        #---------------------------- proprio_data = np.array(mat.get('PRdata'))
        #----------------------------------- proprio_data = proprio_data[[-1],:]
        #------------------------------------------------------------------ try:
            #----------------- sensor_data_tmp = np.transpose(data[[0,1,3,4],:])
            # sensor_data_tmp = sensor_data_tmp[np.where(proprio_data  < proprio_criteria)[1],:]
            # sensor_data_proprio = np.append(sensor_data_proprio,sensor_data_tmp,axis=0)
        #--------------------------------------------------------------- except:
            #------------- sensor_data_proprio = np.transpose(data[[0,1,3,4],:])
            # sensor_data_proprio = sensor_data_proprio[np.where(proprio_data < proprio_criteria)[1],:]
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
    #--------------------------------------- # Plotting PCA main directions GKDE
    #------------------------------------------ fig1, axes1 = initializeFigure()
    #------------------------------------------ fig2, axes2 = initializeFigure()
    #---------------------------------------------------------- axes1.hold(True)
    #---------------------------------------------------------- axes2.hold(True)
    #-------------------------------------- x_grid = np.linspace(-1.0, 1.5, 500)
    #-------------------------------------- y_grid = np.linspace(-1.0, 1.5, 500)
#------------------------------------------------------------------------------ 
    # #-------- sensor_data_no_proprio_pca = pca.transform(sensor_data_no_proprio)
    # #--------------------- kde_x = gaussian_kde(sensor_data_no_proprio_pca[:,0])
    # #-------------------------------------------- pdf_x = kde_x.evaluate(x_grid)
    # #------------------- axes1.plot(x_grid, pdf_x, color="red", alpha=0.5, lw=3)
    # #------------------------------------------------- axes1.set_xlim(-1.0, 1.5)
# #------------------------------------------------------------------------------
    # #--------------------- kde_y = gaussian_kde(sensor_data_no_proprio_pca[:,1])
    # #-------------------------------------------- pdf_y = kde_y.evaluate(y_grid)
    # #------------------- axes2.plot(y_grid, pdf_y, color="red", alpha=0.5, lw=3)
    # #------------------------------------------------- axes2.set_xlim(-1.0, 1.5)
# #------------------------------------------------------------------------------
    # #-------------- sensor_data_proprio_pca = pca.transform(sensor_data_proprio)
    # #------------------------ kde_x = gaussian_kde(sensor_data_proprio_pca[:,0])
    # #-------------------------------------------- pdf_x = kde_x.evaluate(x_grid)
    # #------------------ axes1.plot(x_grid, pdf_x, color="blue", alpha=0.5, lw=3)
    # #------------------------------------------------- axes1.set_xlim(-1.0, 1.5)
# #------------------------------------------------------------------------------
    # #------------------------ kde_y = gaussian_kde(sensor_data_proprio_pca[:,1])
    # #-------------------------------------------- pdf_y = kde_y.evaluate(y_grid)
    # #------------------ axes2.plot(y_grid, pdf_y, color="blue", alpha=0.5, lw=3)
    # #------------------------------------------------- axes2.set_xlim(-1.0, 1.5)
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
# #------------------------------------------------------------------------------
    # #----------------- ####-------------------------------------------------####
    #- #----------------------------------------- ####             KL divergence
    # #----------------- ####-------------------------------------------------####
    # #--------------------------------------- directories = ['EVD_no_Proprio_0/',
                   # #--------------------------------------- 'EVD_no_Proprio_1/',
                   # #--------------------------------------- 'EVD_no_Proprio_2/',
                   # #--------------------------------------- 'EVD_no_Proprio_3/',
                   # #--------------------------------------- 'EVD_no_Proprio_4/',
                   # #--------------------------------------- 'EVD_no_Proprio_6/',
                   # #--------------------------------------- 'EVD_no_Proprio_7/',
                   # #--------------------------------------- 'EVD_no_Proprio_8/',
                   # #--------------------------------------- 'EVD_no_Proprio_9/',
                   # #------------------------------------------ 'EVD_Proprio_0/',
                   # #------------------------------------------ 'EVD_Proprio_1/',
                   # #------------------------------------------ 'EVD_Proprio_2/',
                   # #------------------------------------------ 'EVD_Proprio_3/',
                   # #------------------------------------------ 'EVD_Proprio_4/',
                   # #------------------------------------------ 'EVD_Proprio_6/',
                   # #------------------------------------------ 'EVD_Proprio_7/',
                   # #------------------------------------------ 'EVD_Proprio_8/',
                   # #------------------------------------------ 'EVD_Proprio_9/']
    # #------------------------------- models = {key: None for key in directories}
# #------------------------------------------------------------------------------
    # #------------------------------------------ n_directories = len(directories)
    # #-------------------------------- kde_x = {key: None for key in directories}
    # #-------------------------------- kde_y = {key: None for key in directories}
# #------------------------------------------------------------------------------
# #------------------------------------------------------------------------------
    # #-------------------------------------------- for i in range(n_directories):
        # #-------------------------------------------- directory = directories[i]
        # #----- print('Working on directory {} of {}'.format(i+1, n_directories))
        # #------------------------- mat = h5py.File(directory + 'SMdata.mat','r')
        # #------------------------------------ data = np.array(mat.get('SMdata'))
        # #------------------------- mat = h5py.File(directory + 'PRdata.mat','r')
        # #---------------------------- proprio_data = np.array(mat.get('PRdata'))
        # #----------------------------------- proprio_data = proprio_data[[-1],:]
        # #------------------------- sensor = np.transpose(data[[0,1,3,4],:])
        # # sensor = sensor[np.where(proprio_data < proprio_criteria)[1],:]
        # #------------------------------ sensor = pca.transform(sensor)
        # #--------------------- kde_x[directory] = gaussian_kde(sensor[:,0])
        # #--------------------- kde_y[directory] = gaussian_kde(sensor[:,1])
# #------------------------------------------------------------------------------
# #------------------------------------------------------------------------------
# #------------------------------------------------------------------------------
    # #----------------------- kl_div_x = np.zeros((n_directories, n_directories))
    # #----------------------- kl_div_y = np.zeros((n_directories, n_directories))
# #------------------------------------------------------------------------------
    # #-------------------------------------------- for i in range(n_directories):
        # #----- print('Working on directory {} of {}'.format(i+1, n_directories))
        # #---------------------------------------- for j in range(n_directories):
            # # kl_div_x[i,j] = stats.entropy(kde_x[directories[i]].evaluate(x_grid), kde_x[directories[j]].evaluate(x_grid))
            # # kl_div_y[i,j] = stats.entropy(kde_y[directories[i]].evaluate(y_grid), kde_y[directories[j]].evaluate(y_grid))
# #------------------------------------------------------------------------------
    # #---------------------------------------------------------------- plt.draw()
    # #-------------------------------------------------------- os.chdir(this_dir)
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
    #------------------------------------------------------ print('Plot 1 of 4')
    #-------- sensor_data_no_proprio_pca = pca.transform(sensor_data_no_proprio)
    #--------------------- kde_x = gaussian_kde(sensor_data_no_proprio_pca[:,0])
    #----------- pdf_x = kde_x.evaluate(x_grid)#/np.amax(kde_x.evaluate(x_grid))
    #------------------- axes1.plot(x_grid, pdf_x, color="red", alpha=0.5, lw=3)
    #------------------------------------------------- axes1.set_xlim(-1.0, 1.5)
    #------------------------------------------------------ print('Plot 2 of 4')
    #--------------------- kde_y = gaussian_kde(sensor_data_no_proprio_pca[:,1])
    #----------- pdf_y = kde_y.evaluate(y_grid)#/np.amax(kde_x.evaluate(y_grid))
    #------------------- axes2.plot(y_grid, pdf_y, color="red", alpha=0.5, lw=3)
    #------------------------------------------------- axes2.set_xlim(-1.0, 1.5)
    #- savemat('PDF_no_proprio_NCsamples.mat', mdict = {'x': pdf_x, 'y': pdf_y})
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
    #------------------------------------------------------ print('Plot 3 of 4')
    #-------------- sensor_data_proprio_pca = pca.transform(sensor_data_proprio)
    #------------------------ kde_x = gaussian_kde(sensor_data_proprio_pca[:,0])
    #----------- pdf_x = kde_x.evaluate(x_grid)#/np.amax(kde_x.evaluate(x_grid))
    #------------------ axes1.plot(x_grid, pdf_x, color="blue", alpha=0.5, lw=3)
    #------------------------------------------------- axes1.set_xlim(-1.0, 1.5)
    #------------------------------------------------------ print('Plot 4 of 4')
    #------------------------ kde_y = gaussian_kde(sensor_data_proprio_pca[:,1])
    #----------- pdf_y = kde_y.evaluate(y_grid)#/np.amax(kde_x.evaluate(y_grid))
    #------------------ axes2.plot(y_grid, pdf_y, color="blue", alpha=0.5, lw=3)
    #------------------------------------------------- axes2.set_xlim(-1.0, 1.5)
    #---- savemat('PDF_proprio_NCsamples.mat', mdict = {'x': pdf_x, 'y': pdf_y})
