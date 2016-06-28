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
import pandas as pd

class GMM(object):
    '''
    classdocs
    '''

    def __init__(self, n_components):
        self.type='GMM'
        GMMtmp=mix.GMM(n_components=n_components,
                       covariance_type='full',
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
        
    def trainIncrementalLearning(self,new_data,alpha):
        self.model.init_params='';
        n_new_samples=np.size(new_data,0)
        n_persistent_samples=np.round(((1-alpha)*n_new_samples)/alpha)
        persistent_data=self.model.sample(n_persistent_samples)
        data=np.concatenate((persistent_data,new_data),axis=0)
        self.model.fit(data)
        
    def train_K_means(self,data):
        pass
        
    def predict(self,x_dims, y_dims, y):
        '''
            This method returns the value of x that maximaze the probability P(x|y)
        '''
        y=np.mat(y)
        n_dimensions=np.amax(len(x_dims))+np.amax(len(y_dims))
        n_components=self.model.n_components
        gmm=self.model
        likely_x=np.mat(np.zeros((len(x_dims),n_components)))
        sm=np.mat(np.zeros((len(x_dims)+len(y_dims),n_components)))
        p_xy=np.mat(np.zeros((n_components,1)))
        
        for k,(Mu, Sigma) in enumerate(zip(gmm.means_, gmm._get_covars())):
            Mu=np.transpose(Mu)
            #----------------------------------------------- Sigma=np.mat(Sigma)
            Sigma_yy=Sigma[:,y_dims]
            Sigma_yy=Sigma_yy[y_dims,:]
            
            Sigma_xy=Sigma[x_dims,:]
            Sigma_xy=Sigma_xy[:,y_dims]
            tmp1=linalg.inv(Sigma_yy)*np.transpose(y-Mu[y_dims])
            tmp2=np.transpose(Sigma_xy*tmp1)
            likely_x[:,k]=np.transpose(Mu[x_dims]+tmp2)
            
            #----------- sm[:,k]=np.concatenate((likely_x[:,k],np.transpose(y)))
            likely_x_tmp=pd.DataFrame(likely_x[:,k],index=x_dims)
            y_tmp=pd.DataFrame(np.transpose(y),index=y_dims)
            tmp3=pd.concat([y_tmp, likely_x_tmp])
            tmp3=tmp3.sort_index()
            
            sm[:,k]=tmp3.as_matrix()
            
            tmp4=1/(np.sqrt(((2.0*np.pi)**n_dimensions)*np.abs(linalg.det(Sigma))))
            tmp5=np.transpose(sm[:,k])-(Mu)
            tmp6=linalg.inv(Sigma)
            tmp7=np.exp((-1.0/2.0)*(tmp5*tmp6*np.transpose(tmp5))) #Multiply time GMM.Priors????
            p_xy[k,:]=np.reshape(tmp4*tmp7,(1))
            #- print('Warning: Priors are not be considering to compute P(x,y)')
            
        k_ok=np.argmax(p_xy);
        x=likely_x[:,k_ok];
        
        return np.array(x.transpose())[0]
        
    def plotGMMProjection(self,fig,axes,column1,column2):
        '''
            Display Gaussian distributions with a 95% interval of confidence
        '''
        # Number of samples per component
        gmm=self.model;
        color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])
        
        title='GMM'
        
        plt.figure(fig.number)
        plt.sca(axes)        
    
        
        for i,(mean, covar, color) in enumerate(zip(gmm.means_, gmm._get_covars(), color_iter)):
            covar_plt=np.zeros((2,2))
            print(covar_plt)
            covar_plt[0,0]=covar[column1,column1]
            covar_plt[1,1]=covar[column2,column2]
            covar_plt[0,1]=covar[column1,column2]
            covar_plt[1,0]=covar[column2,column1]
            
            mean_plt=[mean[column1], mean[column2]]
            
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
    
        