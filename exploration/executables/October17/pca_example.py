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
def kl_montecarlo(f, g, nsamples=500000):
    return np.mean(f.score(f.sample(nsamples)) - g.score(g.sample(nsamples)))

def kl_unscented(f, g):
    d = f.means_[0].shape[0]
    ncomponents = f.means_.shape[0]
    
    log_ratios = np.empty(ncomponents)
    for i, (mu, cov) in enumerate(zip(f.means_, f.covars_)):
        u, s, v = np.linalg.svd(cov)
        sigma_points = np.empty((2*d, d))
        for k in range(s.shape[0]):
            cov_sqrt = np.sqrt(d*s[k])*u[:,k]
            sigma_points[[k, k+d], :] = [mu + cov_sqrt, mu - cov_sqrt]
        plt.plot(sigma_points, '.')
        log_ratios[i] = np.sum(f.score(sigma_points) - g.score(sigma_points))
    return (1./(2.*d))*f.weights_.dot(log_ratios).sum()
    
if __name__ == '__main__':
    
    this_dir =  os.getcwd()
    sys.path.append("../../")
    from DataManager.PlotTools import initializeFigure
    from Models.GeneralModels.ILGMM_GREC import ILGMM

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
    
    ####################################################################################
    ####-------------------------------------------------####
    ####              DATA TRANSFORMATION: Two directories
    ####-------------------------------------------------####
    directories = ['EVD_no_Proprio_9/',
                   'EVD_Proprio_9/']
    for i in range(len(directories)-1):
        directory = directories[i]
        mat = h5py.File(directory + 'SMdata.mat','r')
        data = np.array(mat.get('SMdata'))
        mat = h5py.File(directory + 'PRdata.mat','r')
        proprio_data = np.array(mat.get('PRdata'))
        proprio_data1 = proprio_data[[-1],:]
        sensor_data1 = np.transpose(data[[0,1,3,4],:])
        sensor_data1 = sensor_data1[np.where(proprio_data1 < proprio_criteria)[1],:]
        sensor_data1 = pca.transform(sensor_data1)
        for j in  range(i+1,len(directories)):
            directory = directories[j]
            mat = h5py.File(directory + 'SMdata.mat','r')
            data = np.array(mat.get('SMdata'))
            mat = h5py.File(directory + 'PRdata.mat','r')
            proprio_data = np.array(mat.get('PRdata'))
            proprio_data2 = proprio_data[[-1],:]
            sensor_data2 = np.transpose(data[[0,1,3,4],:])
            sensor_data2 = sensor_data2[np.where(proprio_data2 < proprio_criteria)[1]]
            sensor_data2 = pca.transform(sensor_data2)
    fig1, axes1 = initializeFigure()
    plt.figure(fig1.number)
    plt.sca(axes1)
    plt.plot(sensor_data1[:,0],sensor_data1[:,1],'.r')
    fig2, axes2 = initializeFigure()
    plt.figure(fig2.number)
    plt.sca(axes2)
    plt.plot(sensor_data2[:,0],sensor_data2[:,1],'.b')
    #---------------------------------------------------------------- plt.show()
    ####-------------------------------------------------####
    ####              DISTRIBUTION: Two directories
    ####-------------------------------------------------####
    # The grid we'll use for plotting
    xmin = -1.0
    ymin = -1.0
    xmax = 1.5
    ymax = 1.5
    x_grid = np.linspace(-1.0, 1.5, 500)
    y_grid = np.linspace(-1.0, 1.5, 500)
    #------ kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    # pdf = gaussian_kde(sensor_data1[:,0], bw_method=0.02/  sensor_data1[:,0].std(ddof=1)).evaluate(x_grid)
    pdf1 = gaussian_kde(sensor_data1[:,0]).evaluate(x_grid)
    pdf2 = gaussian_kde(sensor_data2[:,0]).evaluate(x_grid)
    fig3, axes3= initializeFigure()
    axes3.plot(x_grid, pdf1, color='blue', alpha=0.5, lw=3)
    axes3.plot(x_grid, pdf2, color='red', alpha=0.5, lw=3)
    axes3.set_xlim(-1.0, 1.5)
    # pdf = gaussian_kde(sensor_data1[:,1], bw_method=0.02 / sensor_data1[:,1].std(ddof=1)).evaluate(y_grid)
    pdf1 = gaussian_kde(sensor_data1[:,1]).evaluate(y_grid)
    pdf2 = gaussian_kde(sensor_data2[:,1]).evaluate(y_grid)
    fig4, axes4= initializeFigure()
    axes4.plot(y_grid, pdf1, color='blue', alpha=0.5, lw=3)
    axes4.plot(y_grid, pdf2, color='red', alpha=0.5, lw=3)
    axes4.set_xlim(-1.0, 1.5)

    ####################################################################################
    
    
    ####-------------------------------------------------####
    ####    KL_divergence GMM
    ####-------------------------------------------------####
    model1 = ILGMM(min_components=25)
    model1.params['max_components'] = 35
    model1.train(sensor_data1)
    
    model2 = ILGMM(min_components=25)
    model2.params['max_components'] = 35
    model2.train(sensor_data2)
    
    fig5, ax5 = initializeFigure()
    ax5.set_xlim(-1.0, 1.5)
    ax5.set_ylim(-1.0, 1.5)
    fig5, ax5 = model1.plot_gmm_projection(fig5, ax5, 0, 1)
    
    fig6, ax6 = initializeFigure()
    ax6.set_xlim(-1.0, 1.5)
    ax6.set_ylim(-1.0, 1.5)
    fig6, ax6 = model2.plot_gmm_projection(fig6, ax6, 0, 1)
    plt.show()
    
    kl_unscented(model1.model, model1.model)
    kl_unscented(model1.model, model2.model)
    kl_unscented(model2.model, model2.model)
    kl_unscented(model2.model, model1.model)
    
    kl_montecarlo(model1.model, model1.model)
    kl_montecarlo(model1.model, model2.model)
    kl_montecarlo(model2.model, model2.model)
    kl_montecarlo(model2.model, model1.model)
    
    os.chdir(this_dir)