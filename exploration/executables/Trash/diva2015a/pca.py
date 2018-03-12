'''
Created on Oct 10, 2016

@author: Juan Manuel Acevedo Valle
''' 
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import h5py, os, sys, random
from scipy.stats import gaussian_kde
from scipy.stats.distributions import norm    

def plotPDFx_y(figs, axes, directories, color):
    if len(directories)==0:
        return 
    
    x_grid = np.linspace(-1.0, 1.5, 500)
    y_grid = np.linspace(-1.0, 1.5, 500)
    
    directory = directories[0]
    directories.remove(directory) 
    print('Working on ' + directory)
    mat = h5py.File(directory + 'SMdata.mat','r')
    data = np.array(mat.get('SMdata'))
    mat = h5py.File(directory + 'PRdata.mat','r')
    proprio_data = np.array(mat.get('PRdata'))
    proprio_data = proprio_data[[-1],:]
    sensor_data = np.transpose(data[[0,1,3,4],:])
    sensor_data = sensor_data[np.where(proprio_data < proprio_criteria)[1],:]
    sensor_data = pca.transform(sensor_data)
    
    # The grid we'll use for plotting

    #------ kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs) 
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.

    # pdf = gaussian_kde(sensor_data1[:,0], bw_method=0.02/  sensor_data1[:,0].std(ddof=1)).evaluate(x_grid)
    kde_x = gaussian_kde(sensor_data[:,0])
    pdf_x = kde_x.evaluate(x_grid)
    axes[0].plot(x_grid, pdf_x, color=color, alpha=0.5, lw=3)
    axes[0].set_xlim(-1.0, 1.5)
    
    # pdf = gaussian_kde(sensor_data1[:,1], bw_method=0.02 / sensor_data1[:,1].std(ddof=1)).evaluate(y_grid)
    kde_y = gaussian_kde(sensor_data[:,1])
    pdf_y = kde_y.evaluate(y_grid)
    axes[1].plot(y_grid, pdf_y, color=color, alpha=0.5, lw=3)
    axes[1].set_xlim(-1.0, 1.5)
    
    del(mat,data,sensor_data,proprio_data)
    tmp_dict = {'KDEx_' + directory: kde_x, 'KDEy_' + directory: kde_y, 'PDFx_' + directory: pdf_x, 'PDFy_' + directory: pdf_y }
    plotPDFx_y(figs, axes, directories, color)
    ####################################################################################
  
        
if __name__ == '__main__':
    
    this_dir =  os.getcwd()
    sys.path.append("../../")
    from DataManager.PlotTools import initializeFigure

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
    proprio_criteria = 1   # 1 for considering only non contacts and 2 to consider all
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

    pca =PCA(n_components=2)      
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
    fig1, axes1 = initializeFigure()
    fig2, axes2 = initializeFigure()   
    axes1.hold(True) 
    axes2.hold(True)
    figs = [fig1, fig2]
    axes = [axes1, axes2]
    
    directories = ['EVD_no_Proprio_0/',
                   'EVD_no_Proprio_1/',
                   'EVD_no_Proprio_2/',
                   'EVD_no_Proprio_3/',
                   'EVD_no_Proprio_4/',
                   'EVD_no_Proprio_6/',
                   'EVD_no_Proprio_7/',
                   'EVD_no_Proprio_8/',
                   'EVD_no_Proprio_9/',
                   'Special_EVD_Proprio_5/EVD_no_Proprio_5/']
    
    plotPDFx_y(figs, axes, directories, 'blue')

    
    directories = ['EVD_Proprio_0/',
                   'EVD_Proprio_1/',
                   'EVD_Proprio_2/',
                   'EVD_Proprio_3/',
                   'EVD_Proprio_4/',
                   'EVD_Proprio_6/',
                   'EVD_Proprio_7/',
                   'EVD_Proprio_8/',
                   'EVD_Proprio_9/',
                   'Special_EVD_Proprio_5/EVD_Proprio_5/'] 
    
    plotPDFx_y(figs, axes, directories, 'red')

    plt.show()
    os.chdir(this_dir)
    
