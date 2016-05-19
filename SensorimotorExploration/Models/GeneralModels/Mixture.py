'''
Created on Feb 16, 2016

@author: Juan Manuel Acevedo Valle
'''
from sklearn import mixture as mix
import itertools
import numpy as np
from scipy import linalg 
import matplotlib.pyplot as plt
import matplotlib as mpl

class GMM(object):
    '''
    classdocs
    '''

    def __init__(self, n_components):
        self.type='GMM'
        GMMtmp=mix.GMM(n_components=n_components,
                       covariance_type='diag',
                       random_state=None,
                       thresh=None, 
                       min_covar=0.001, 
                       n_iter=100, 
                       n_init=1, 
                       params='wmc', 
                       init_params='wmc' 
                       )
        self.model=GMMtmp;

    def train(self,data):
        self.model.fit(data)
        
   
        
    def plotGMM_SMProjection(self,fig,axes,column1,column2):
        '''
            Display Gaussian distributions with a 95% interval of confidence
        '''
        # Number of samples per component
        gmm=self.model;
        color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])
        
        title='GMM'
        
        plt.figure(fig.number)
        plt.sca(axes)        
    
        
        for i,(mean, covar, color) in enumerate(zip(
            gmm.means_, gmm._get_covars(), color_iter)):
            covar_plt=np.zeros((2,2))
            print(covar_plt)
            covar_plt[0,0]=covar[column1,column1]
            covar_plt[1,1]=covar[column2,column2]
            covar_plt[0,1]=covar[column1,column2]
            covar_plt[1,0]=covar[column2,column1]
            print(covar_plt)
            
            mean_plt=[mean[column1], mean[column2]]
            print(mean_plt)
            
            v, w = linalg.eigh(covar_plt)
            u = w[0] / linalg.norm(w[0])
            v[0]=2*np.sqrt(5.991*v[0]);
            v[1]=2*np.sqrt(5.991*v[1]);
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
    
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean_plt, v[0], v[1], 180 + angle, color=color)
            ell.set_alpha(0.5)
            
            axes.add_patch(ell)
    
        axes.set_xlim(-1, 1)
        axes.set_ylim(-1, 1)
        axes.set_title(title)
        return fig,axes
    
        