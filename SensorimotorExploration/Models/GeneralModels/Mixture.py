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
                       tol = 0.001,
                       min_covar=0.0001,  
                       n_iter=100, 
                       n_init=1,      
                       params='wmc',   
                       init_params='wmc')
        self.model=GMMtmp;
        self.initialized=False

    def train(self,data):
        self.model.fit(data)
        if self.model.converged_:
            self.initialized=True
        else:
            print('The EM-algorithm did not converged...')
            
    def train_bestGMM(self,data):  #WRITE THIS FUNCTION
        self.model.fit(data)
        if self.model.converged_:
            self.initialized=True
        else:
            print('The EM-algorithm did not converged...')
     
    def getBestGMM(self, data, lims=[1,10]):         
        lowest_bic = np.infty
        bic = []
        aic= []
        minim = False
        minim_flag = 2
        
        n_components_range = range(lims[0],lims[1]+1,1)
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM, beware for cazes when te model is not found in any case
            gmm = mix.GMM(n_components=n_components,
                           covariance_type='full',
                           random_state=None,
                           thresh=None,
                           tol = 0.001,
                           min_covar=0.0001,  
                           n_iter=100, 
                           n_init=1,      
                           params='wmc',   
                           init_params='wmc')
            gmm.fit(data)
            bic.append(gmm.bic(data))
            aic.append(gmm.aic(data))
            
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = n_components
            try:    
                if (bic[-1] > bic[-2] and 
                    bic[-2] > bic[-3] and
                    bic[-3] < bic[-4] and
                    bic[-4] < bic[-5] and
                    bic[-5] < bic[-6]):
                    best_gmm = n_components - 2
                    break    
                
            except IndexError:
                pass

        gmm = mix.GMM(n_components=best_gmm,
                       covariance_type='full',
                       random_state=None,
                       thresh=None,
                       tol = 0.001,
                       min_covar=0.0001,  
                       n_iter=100, 
                       n_init=1,      
                       params='wmc',   
                       init_params='wmc')
        gmm.fit(data)        
        
        self.model.weights_ = gmm.weights_
        self.model.covars_ = gmm._get_covars()
        self.model.means_ = gmm.means_
        self.model.n_components = gmm.n_components 
        
    def trainIncrementalLearning(self,new_data,alpha):
        if self.initialized:
            self.model.init_params='';
            n_new_samples = np.size(new_data,0)
            n_persistent_samples = np.round(((1-alpha)*n_new_samples)/alpha)
            persistent_data = self.model.sample(n_persistent_samples)
            data = np.concatenate((persistent_data,new_data),axis=0)
            self.model.fit(data)
            if self.model.converged_==False:
                print('The EM-algorith did not converged...')
        else:
            self.train(new_data)
    
    
    def getBIC(self,data):
        return self.model.bic(data)        
        
    def predict(self, x_dims, y_dims, y):
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
            
            covar_plt[0,0] = covar[column1,column1]
            covar_plt[1,1] = covar[column2,column2]
            covar_plt[0,1] = covar[column1,column2]
            covar_plt[1,0] = covar[column2,column1]
            
            mean_plt = [mean[column1], mean[column2]]
            
            v, w = linalg.eigh(covar_plt)
            u = w[0] / linalg.norm(w[0])
            v[0] = 2.0*np.sqrt(2.0*v[0]);
            v[1] = 2.0*np.sqrt(2.0*v[1]);
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
    
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean_plt, v[0], v[1], 180 + angle, color=color)
            ell.set_alpha(0.5)
            
            axes.add_patch(ell)
            
        #=======================================================================
        # axes.set_xlim(-1, 1)
        # axes.set_ylim(-1, 1)
        #=======================================================================
        axes.set_title(title)
        return fig,axes
    
    def plotGMM3DProjection(self,fig,axes,column1,column2,column3):
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
            covar_plt=np.zeros((3,3))
            
            covar_plt[0,0] = covar[column1,column1]
            covar_plt[0,1] = covar[column1,column2]
            covar_plt[0,2] = covar[column1,column3]
            covar_plt[1,0] = covar[column2,column1]
            covar_plt[1,1] = covar[column2,column2]
            covar_plt[1,2] = covar[column2,column3]
            covar_plt[2,0] = covar[column3,column1]
            covar_plt[2,1] = covar[column3,column2]
            covar_plt[2,2] = covar[column3,column3]
             
             
            center = [mean[column1], mean[column2], mean[column3]]
             
            U, s, rotation = linalg.svd(covar_plt)
            radii = 1 / np.sqrt(s)
            
            # now carry on with EOL's answer
            u = np.linspace(0.0, 2.0 * np.pi, 100)
            v = np.linspace(0.0, np.pi, 100)
            x = radii[0] * np.outer(np.cos(u), np.sin(v))
            y = radii[1] * np.outer(np.sin(u), np.sin(v))
            z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
            for j in range(len(x)):
                for k in range(len(x)):
                    [x[j,k],y[j,k],z[j,k]] = np.dot([x[j,k],y[j,k],z[j,k]], rotation) + center
             
            axes.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
             
            axes.set_xlabel('x')
            axes.set_ylabel('y')
            axes.set_zlabel('z')
        #=======================================================================
        # axes.set_xlim(-1, 1)
        # axes.set_ylim(-1, 1)
        #=======================================================================
        axes.set_title(title)
        return fig,axes
     
         
    
  