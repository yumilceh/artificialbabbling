'''
Created on Oct 17, 2016

@author: Juan Manuel Acevedo Valle
''' 
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import h5py, os, sys, random
from scipy.stats import gaussian_kde
from scipy.stats.distributions import norm    

def kl_montecarlo1(f, g, nsamples=500000): #https://github.com/pierrelux/notebooks/blob/master/KL%20Estimation%20for%20GMM.ipynb
    return np.mean(f.score(f.sample(nsamples)) - g.score(g.sample(nsamples)))


def kl_montecarlo2(gmm_p, gmm_q, n_samples=500000): #http://stackoverflow.com/questions/26079881/kl-divergence-of-two-gmms
    X = gmm_p.sample(n_samples).append(gmm_q.sample(n_samples),axis=0)
    log_p_X, _ = gmm_p.score(X)
    log_q_X, _ = gmm_q.score(X)
    return log_p_X.mean() - log_q_X.mean() 

def kl_unscented(f, g): #https://github.com/pierrelux/notebooks/blob/master/KL%20Estimation%20for%20GMM.ipynb
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
    
    ####-------------------------------------------------####
    ####             Plot non-proprio distributions
    ####-------------------------------------------------####
    fig1, axes1 = initializeFigure()
    fig2, axes2 = initializeFigure()   
    axes1.hold(True) 
    axes2.hold(True)
    x_grid = np.linspace(-1.0, 1.5, 500)
    y_grid = np.linspace(-1.0, 1.5, 500)
    
    directories_dict = {'non-Proprio':['EVD_no_Proprio_0/',
                                  'EVD_no_Proprio_1/',
                                  'EVD_no_Proprio_2/',
                                  'EVD_no_Proprio_3/',
                                  'EVD_no_Proprio_4/',
                                  'EVD_no_Proprio_6/',
                                  'EVD_no_Proprio_7/',
                                  'EVD_no_Proprio_8/',
                                  'EVD_no_Proprio_9/',
                                  'Special_EVD_Proprio_5/EVD_no_Proprio_5/'],
                       'Proprio':['EVD_Proprio_0/',
                                  'EVD_Proprio_1/',
                                  'EVD_Proprio_2/',
                                  'EVD_Proprio_3/',
                                  'EVD_Proprio_4/',
                                  'EVD_Proprio_6/',
                                  'EVD_Proprio_7/',
                                  'EVD_Proprio_8/',
                                  'EVD_Proprio_9/',
                                  'Special_EVD_Proprio_5/EVD_Proprio_5/']} 
    color_dict = {'non-Proprio': 'blue', 'Proprio': 'red'}
    for key in ['non-Proprio','Proprio']:
        directories = directories_dict[key]
        color = color_dict[key]
        for i in range(len(directories)):
            directory = directories[i]
            print('Working on directory {} of {} [In {} Agents]'.format(i+1, len(directories), key))
            mat = h5py.File(directory + 'SMdata.mat','r')
            data = np.array(mat.get('SMdata'))
            mat = h5py.File(directory + 'PRdata.mat','r')
            proprio_data = np.array(mat.get('PRdata'))
            proprio_data = proprio_data[[-1],:]
            sensor_data = np.transpose(data[[0,1,3,4],:])
            sensor_data = sensor_data[np.where(proprio_data < proprio_criteria)[1],:]
            sensor_data = pca.transform(sensor_data)
            
            # The grid we'll use for plotting
            kde_x = gaussian_kde(sensor_data[:,0])
            pdf_x = kde_x.evaluate(x_grid)
            axes1.plot(x_grid, pdf_x, color=color, alpha=0.5, lw=3)
            axes1.set_xlim(-1.0, 1.5)
            
            kde_y = gaussian_kde(sensor_data[:,1])
            pdf_y = kde_y.evaluate(y_grid)
            axes2.plot(y_grid, pdf_y, color=color, alpha=0.5, lw=3)
            axes2.set_xlim(-1.0, 1.5)
            
            
    ####-------------------------------------------------####
    ####             GMM Distribution
    ####-------------------------------------------------####
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
    models = {key: None for key in directories}
    
    for i in range(len(directories)):
        directory = directories[i]
        
        print('Working on directory {} of {}'.format(i+1, len(directories)))
        mat = h5py.File(directory + 'SMdata.mat','r')
        data = np.array(mat.get('SMdata'))
        mat = h5py.File(directory + 'PRdata.mat','r')
        proprio_data = np.array(mat.get('PRdata'))
        proprio_data = proprio_data[[-1],:]
        sensor_data = np.transpose(data[[0,1,3,4],:])
        sensor_data = sensor_data[np.where(proprio_data < proprio_criteria)[1],:]
        sensor_data = pca.transform(sensor_data)
        
        models[directory] = ILGMM(min_components=20)
        models[directory].params['max_components'] = 40
        models[directory].train(sensor_data)
        
                    
    n_directories = len(directories)
    kl_div_mc1 = np.zeros((n_directories, n_directories))
    kl_div_mc2 = np.zeros((n_directories, n_directories))
    kl_div_u = np.zeros((n_directories, n_directories))
    
    for i in range(n_directories):
        for j in range(n_directories):
            kl_div_mc1[i,j] = kl_montecarlo1(models[directories[i]].model, models[directories[j]].model)
            kl_div_mc2[i,j] = kl_montecarlo2(models[directories[i]].model, models[directories[j]].model)
            kl_div_u[i,j] = kl_unscented(models[directories[i]].model, models[directories[j]].model)
    
    plt.show()
    os.chdir(this_dir)
    
