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

import Tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from distutils import command

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
                    bic[-4] < bic[-5]):
                    best_gmm = n_components - 2
                    break    
                
            except IndexError:
                pass
        if best_gmm <= 6:
            best_gmm = np.array(bic).argmin() + lims[0]
             
        
        
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
         
    def returnCopy(self):
        '''If any trouble be sure that assignation of 
            means and weights is done copying through assignation        
        '''
        copy_tmp = GMM(n_components=self.model.n_components)
        
        copy_tmp.model.covars_ = self.model._get_covars()
        copy_tmp.model.means_ = self.model.means_
        copy_tmp.model.weights_ = self.model.weights_
        
        return copy_tmp
        
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
    
    def predict_all_gaussians(self, x_dims, y_dims, y):
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
        
        x=[0.0] * len(x_dims);
        
        for k,(Mu, Sigma, Weight) in enumerate(zip(gmm.means_, gmm._get_covars(), gmm.weights_)):
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
            
        p_xy = (1.0 / np.sum(p_xy)) * p_xy    
        for k in range(len(gmm.weights_)): 
            x = x + Weight * p_xy[k] * likely_x_tmp
        
        return np.array(x.transpose())   
        
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
    
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean_plt, v[0], v[1], 180 + angle, color=color)
            ell.set_alpha(0.5)
            
            axes.add_patch(ell)
            axes.autoscale_view()

        if axes.get_title() == '':
            axes.set_title(title)
        return fig,axes
    
    def plotGMMProjection_k(self,fig,axes,k,column1,column2):
        '''
            Display Gaussian distributions with a 95% interval of confidence
        '''
        # Number of samples per component
        gmm=self.model;
        
        
        plt.figure(fig.number)
        plt.sca(axes)        
        covar_plt=np.zeros((2,2))
        
        covar = gmm._get_covars()[k]
        covar_plt[0,0] = covar[column1,column1]
        covar_plt[1,1] = covar[column2,column2]
        covar_plt[0,1] = covar[column1,column2]
        covar_plt[1,0] = covar[column2,column1]
        
        mean_plt = [gmm.means_[k][column1], gmm.means_[k][column2]]
        

        v, w = linalg.eigh(covar_plt)
        u = w[0] / linalg.norm(w[0])
        v[0] = 2.0*np.sqrt(2.0*v[0]);
        v[1] = 2.0*np.sqrt(2.0*v[1]);

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean_plt, v[0], v[1], 180 + angle, color='r')
        ell.set_alpha(0.5)
        axes.add_patch(ell)
        axes.autoscale_view()
                   
        return axes
    
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
        if axes.get_title()=='':
            axes.set_title(title)
        return fig,axes
     
    def plot_callback(self):
        n_plots = np.int(self.n_proj_str.get())
        subplot_dim_x = self.proj_arrays[self.n_proj_str.get()][0]
        subplot_dim_y = self.proj_arrays[self.n_proj_str.get()][1]
        self.plots_fig.clf()
        
        current_gauss = self.current_gauss_str.get()
         
        self.plots_ax = []
        for i in range(n_plots): 
            self.plots_ax.append(self.plots_fig.add_subplot(subplot_dim_x,subplot_dim_y, i + 1))
            self.plots_ax[i] = self.plotGMMProjection_k(self.plots_fig,self.plots_ax[i],current_gauss,0,1)
        
        self.plots_canvas.draw()
        
        #=======================================================================
        # self.plots_ax = []
        # for i in range(self.n_proj):
        #     self.plots_ax.append(self.plots_fig.add_subplot(self.proj_arrays[n_proj_tmp][0],
        #                                                   self.proj_arrays[n_proj_tmp][1], i+1))
        #     self.plots_ax[i].spines['right'].set_visible(False)
        #     self.plots_ax[i].spines['top'].set_visible(False)
        #     self.plots_ax[i].spines['left'].set_visible(False)
        #     self.plots_ax[i].spines['bottom'].set_visible(False)
        #     self.plots_ax[i].xaxis.set_ticks_position('none')
        #     self.plots_ax[i].yaxis.set_ticks_position('none')
        #     self.plots_ax[i].xaxis.set_ticks([])
        #     self.plots_ax[i].yaxis.set_ticks([])
        # 
        # self.plots_canvas.draw()
        #=======================================================================
    def plot_array_callback(self):
        n_proj_old = self.n_proj
        n_proj_tmp = self.n_proj_str.get()
        self.n_proj = np.int(n_proj_tmp)
        
        if self.n_proj > 1:
            self.n_edit_proj_m.config(state=tk.NORMAL)
        else:
            self.n_edit_proj_m.config(state=tk.DISABLED)
        
        if self.n_proj < n_proj_old:
            pass
        else:
            pass
        
        self.plot_callback()
        
    def current_gauss_callback(self):
        self.current_gauss = np.int(self.current_gauss_str.get())
        if self.current_gauss == 0:
            self.prev_gauss_btn.config(state = tk.DISABLED)
        else:   
             self.prev_gauss_btn.config(state = tk.NORMAL)
             
        if self.current_gauss >= self.model.n_components - 1:
            self.next_gauss_btn.config(state = tk.DISABLED)
        else:
            self.next_gauss_btn.config(state = tk.NORMAL)
            
        self.plot_callback()
             
    def prev_gauss_callback(self):
        self.current_gauss = self.current_gauss - 1
        self.current_gauss_str.set(str(self.current_gauss))

    
    def next_gauss_callback(self):        
        self.current_gauss = self.current_gauss + 1
        self.current_gauss_str.set(str(self.current_gauss))

    def current_projection_callback(self):
        if np.int(self.n_edit_proj_str.get()) > self.n_proj -1:
            self.n_edit_proj_str.set(str(self.n_proj -1))
        self.n_edit_proj = np.int(self.n_edit_proj_str.get())
        self.edit_proj_dim1_str.set(str(self.proj_dim[self.n_edit_proj,0]))
        self.edit_proj_dim2_str.set(str(self.proj_dim[self.n_edit_proj,1]))
        
        
        
    def current_dim1_callback(self):
        self.proj_dim[self.n_edit_proj,0]=np.int(self.edit_proj_dim1_str.get())
        pass
                
    def current_dim2_callback(self):
        self.proj_dim[self.n_edit_proj,1]=np.int(self.edit_proj_dim2_str.get())
        pass
                
    def interactiveModel(self):
        if self.initialized:
            self.n_dims = self.model._get_covars()[0].shape[0]
            ### Main window container
            self.root_window = tk.Tk()
            self.root_window.geometry("800x800")
            self.root_window.title("Interactive Analysis of GMM")
            
            self.root_frame = tk.Frame(self.root_window, width=800, height=800, bg="green")
            self.root_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            
            self.guiPlotsPanel()
            self.guiControlPanel()
        
            self.plot_callback()
        #----------------------------------- self.guiMotorPanel_reset_callback() 
            self.root_window.mainloop()
        else:
            print("Interactive mode only can be used when the model has been initialized")    
    
    def guiPlotsPanel(self):
        self.plots_frame = tk.Frame(self.root_frame, width=800, height=600, bg="white")
        self.plots_frame.pack(side=tk.TOP, fill=tk.X, expand=1)
        self.plots_container_frame = tk.Frame(self.plots_frame, width=800, height=600, bg="black")
        self.plots_container_frame.pack(side=tk.LEFT, fill=tk.NONE, expand=0)
        
        self.plots_fig = plt.figure()
        self.plots_fig.set_dpi(100)
        self.plots_fig.set_figheight(6)
        self.plots_fig.set_figwidth(8)

        self.plots_fig.patch.set_facecolor('red')
        self.plots_canvas = FigureCanvasTkAgg(self.plots_fig, master=self.plots_container_frame) 
        self.plots_canvas.show()
        self.plots_canvas.get_tk_widget().pack(side="left",fill="none", expand=False)
        self.plots_canvas.get_tk_widget().configure(background='white',  highlightcolor='white', highlightbackground='white')
        
        self.plots_ax = self.plots_fig.add_subplot(111)
        self.plots_ax.spines['right'].set_visible(False)
        self.plots_ax.spines['top'].set_visible(False)
        self.plots_ax.spines['left'].set_visible(False)
        self.plots_ax.spines['bottom'].set_visible(False)
        self.plots_ax.xaxis.set_ticks_position('none')
        self.plots_ax.yaxis.set_ticks_position('none')
        self.plots_ax.xaxis.set_ticks([])
        self.plots_ax.yaxis.set_ticks([])
        
        self.plots_canvas.draw()
    
    def guiControlPanel(self):
        self.control_frame = tk.Frame(self.root_frame, width=800, height=200, bg="white")
        self.control_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.control_entries_frame = tk.Frame(self.control_frame, width=800, height=200, bg="white") 
        self.control_entries_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        
        #SELECT NUMBER OF PROJECTIONS
        self.n_proj_lbl = tk.Label(self.control_entries_frame, text="Number of projections:")
        self.n_proj_lbl.grid(row=0, padx=5, pady=5)
        self.n_proj_str = tk.StringVar()
        self.n_proj_str.set("1")     
        self.n_proj_str.trace("w", lambda name, index, mode, sv=self.n_proj_str: self.plot_array_callback())

           
        self.n_proj = 1   
           
        self.n_proj_m = tk.OptionMenu(self.control_entries_frame, self.n_proj_str, "1","2","3","4","6","8","9")
        self.proj_arrays={'1':[1,1], '2':[1,2], '3':[1,3], '4':[2,2], '6':[2,3], '8':[2,4], '9':[3,3]}
        
        self.n_proj_m.grid(row=0, column =1, columnspan=2, padx=5, pady=5)
        
        #SELECT NUMBER OF PROJECTION TO BE EDITED
        self.n_edit_proj_lbl = tk.Label(self.control_entries_frame, text="Projection to edit:")
        self.n_edit_proj_lbl.grid(row=3, column=0, padx=5, pady=5)
        self.n_edit_proj_str = tk.StringVar()
        self.n_edit_proj_str.set("1")
        posible_projectios = ["1","2", "3","4","5","6","7","8","9"]
        self.n_edit_proj_m = tk.OptionMenu(self.control_entries_frame, self.n_edit_proj_str, *posible_projectios)
        self.n_edit_proj_str.trace("w", lambda name, index, mode, sv=self.n_edit_proj_str: self.current_projection_callback())

        
        self.n_edit_proj_m.config(state=tk.DISABLED)
        self.n_edit_proj_m.grid(row=3, column=1, columnspan=2, padx=5, pady=5)
        self.n_edit_proj = np.int(self.n_edit_proj_str.get())
        
        #SELECT DIMENSIONS

        self.edit_proj_dim1_str = tk.StringVar()
        self.edit_proj_dim2_str = tk.StringVar()

        if self.n_dims <= 1:
            self.edit_proj_dim1_str.set("0")
            self.edit_proj_dim2_str.set("0")
            self.proj_dim=np.reshape(np.array([0, 0] * 9), (9,2))
        else:
            self.edit_proj_dim1_str.set("0")
            self.edit_proj_dim2_str.set("1")
            self.proj_dim=np.reshape(np.array([0, 1] * 9), (9,2))
              
           
        self.edit_proj_dim1_str.trace("w", lambda name, index, mode, sv=self.edit_proj_dim1_str: self.current_dim1_callback())
        self.edit_proj_dim2_str.trace("w", lambda name, index, mode, sv=self.edit_proj_dim2_str: self.current_dim2_callback())   
              
        self.edit_proj_dim1_lbl = tk.Label(self.control_entries_frame, text="Dimension 1:")
        self.edit_proj_dim1_lbl.grid(row=5, column=0, padx=5, pady=0)
        posible_dimensions = range(self.n_dims)
        self.edit_proj_dim1_m = tk.OptionMenu(self.control_entries_frame, self.edit_proj_dim1_str, *posible_dimensions)#, command = self.current_dim1_callback())
        self.edit_proj_dim1_m.config(state=tk.NORMAL)
        self.edit_proj_dim1_m.grid(row=5, column=1, columnspan=2, padx=5, pady=0)
        self.edit_proj_dim2_lbl = tk.Label(self.control_entries_frame, text="Dimension 2:")
        self.edit_proj_dim2_lbl.grid(row=6, column=0, padx=5, pady=0)

        self.edit_proj_dim2_m = tk.OptionMenu(self.control_entries_frame, self.edit_proj_dim2_str, *posible_dimensions)#, command = self.current_dim2_callback())
        self.edit_proj_dim2_m.config(state=tk.NORMAL)
        self.edit_proj_dim2_m.grid(row=6, column=1, columnspan=2, padx=5, pady=0)
        
        #Current Gaussian and sweeping 
        self.prev_gauss_btn = tk.Button(self.control_entries_frame, state=tk.DISABLED, text="<<", command = self.prev_gauss_callback)
        self.prev_gauss_btn.grid(row=3, column=3, padx=5, pady=0)
        self.current_gauss_str = tk.StringVar()
        self.current_gauss_str.set("0")
        self.entry_gauss = tk.Entry(self.control_entries_frame, state=tk.DISABLED, 
                                    textvariable=self.current_gauss_str, width=4)
        self.entry_gauss.grid(row=3, column=4, padx=5, pady=0)
        self.current_gauss_str.trace("w", lambda name, index, mode, sv=self.current_gauss_str: self.current_gauss_callback())
        self.current_gauss = 0
        self.next_gauss_btn = tk.Button(self.control_entries_frame, state=tk.DISABLED, text=">>", command = self.next_gauss_callback)
        self.next_gauss_btn.grid(row=3, column=5, padx=5, pady=0)
    
        if self.model.n_components > 1:
            self.next_gauss_btn.config(state=tk.NORMAL)
            self.entry_gauss.config(state=tk.NORMAL) 
    
    
        
    
    '''
        self.sv_step = tk.StringVar()
        self.sv_step.set("10")     
        self.vt_shape_step = np.int(self.sv_step.get())       
        self.sv_step.trace("w", lambda name, index, mode, sv=self.sv_step: self.set_vt_opts_callback())
        
        self.sv_current = tk.StringVar()
        self.sv_current.set("0")  
        self.vt_shape_current = np.int(self.sv_current.get())     
        self.sv_current.trace("w", lambda name, index, mode, sv=self.sv_current: self.set_vt_current_callback())

        
        self.lbl_step_vt = tk.Label(self.vt_sound_opt_frame, text="Step")
        self.lbl_current_vt = tk.Label(self.vt_sound_opt_frame, text="Current")
        self.entry_step_vt = tk.Entry(self.vt_sound_opt_frame, state=tk.DISABLED, textvariable=self.sv_step, width=8)
        self.entry_current_vt = tk.Entry(self.vt_sound_opt_frame, state=tk.DISABLED, textvariable=self.sv_current, width=8)
        self.lbl_step_vt.pack(side=tk.TOP, fill=tk.X, expand=1)
        self.entry_step_vt.pack(side=tk.TOP, fill=tk.X, expand=1)
        self.lbl_current_vt.pack(side=tk.TOP, fill=tk.X, expand=1)
        self.entry_current_vt.pack(side=tk.TOP, fill=tk.X, expand=1)
        
        self.vt_shape_step = np.int(self.sv_step.get())
        self.vt_shape_current = np.int(self.sv_current.get())
        , width=8
               
    def guiMotorPanel(self):
        self.motor_frame = tk.Frame(self.root_frame, width=800, height=150, bg="white")
        self.motor_frame.pack(side=tk.TOP, fill=tk.X, expand=1)
        
        self.motor_entries_frame = tk.Frame(self.motor_frame, width=800, height=150, bg="white") 
        self.motor_entries_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
          
        self.lbl_m1 = tk.Label(self.motor_entries_frame, text="M1")
        self.lbl_m1.grid(row=0, padx=5, pady=5)
        self.lbl_m2 = tk.Label(self.motor_entries_frame, text="M2")
        self.lbl_m2.grid(row=1, padx=5, pady=5)
        self.lbl_m3 = tk.Label(self.motor_entries_frame, text="M3")
        self.lbl_m3.grid(row=2, padx=5, pady=5)
        self.lbl_m4 = tk.Label(self.motor_entries_frame, text="M4")
        self.lbl_m4.grid(row=3, padx=5, pady=5)
        
        self.entry_m1 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m1.grid(row=0, column=1, padx=5, pady=5)
        self.entry_m2 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m2.grid(row=1, column=1, padx=5, pady=5)
        self.entry_m3 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m3.grid(row=2, column=1, padx=5, pady=5)
        self.entry_m4 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m4.grid(row=3, column=1, padx=5, pady=5)        
        
        self.lbl_m5 = tk.Label(self.motor_entries_frame, text="M5")
        self.lbl_m5.grid(row=4, column=0, padx=5, pady=5)
        self.lbl_m6 = tk.Label(self.motor_entries_frame, text="M6")
        self.lbl_m6.grid(row=0, column=2, padx=5, pady=5)
        self.lbl_m7 = tk.Label(self.motor_entries_frame, text="M7")
        self.lbl_m7.grid(row=1, column=2, padx=5, pady=5)
        self.lbl_m8 = tk.Label(self.motor_entries_frame, text="M8")
        self.lbl_m8.grid(row=2, column=2, padx=5, pady=5)
        
        self.entry_m5 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m5.grid(row=4, column=1, padx=5, pady=5)
        self.entry_m6 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m6.grid(row=0, column=2, padx=5, pady=5)
        self.entry_m7 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m7.grid(row=1, column=2, padx=5, pady=5)
        self.entry_m8 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m8.grid(row=2, column=2, padx=5, pady=5)

        self.lbl_m9 = tk.Label(self.motor_entries_frame, text="M9")
        self.lbl_m9.grid(row=3, column=2, padx=5, pady=5)
        self.lbl_m10 = tk.Label(self.motor_entries_frame, text="M10")
        self.lbl_m10.grid(row=4, column=2, padx=5, pady=5)
        self.lbl_m11 = tk.Label(self.motor_entries_frame, text="M11")
        self.lbl_m11.grid(row=1, column=4, padx=5, pady=5)
        self.lbl_m12 = tk.Label(self.motor_entries_frame, text="M12")
        self.lbl_m12.grid(row=2, column=4, padx=5, pady=5)
        self.lbl_m13 = tk.Label(self.motor_entries_frame, text="M13")
        self.lbl_m13.grid(row=3, column=4, padx=5, pady=5)
        
        self.entry_m9 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m9.grid(row=3, column=2, padx=5, pady=5)
        self.entry_m10 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m10.grid(row=4, column=2, padx=5, pady=5)
        self.entry_m11 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m11.grid(row=1, column=5, padx=5, pady=5)
        self.entry_m12 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m12.grid(row=2, column=5, padx=5, pady=5)
        self.entry_m13 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m13.grid(row=3, column=5, padx=5, pady=5)                
        
                
        self.btn_execute_m = tk.Button(self.motor_frame, text="Execute", command = self.execute_callback)
        self.btn_execute_m.pack(side=tk.LEFT, fill=tk.NONE, expand=0)   
    '''   
    
  