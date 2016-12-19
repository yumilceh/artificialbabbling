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
        if axes.get_title() == '':
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
        if axes.get_title()=='':
            axes.set_title(title)
        return fig,axes
     
    def interactiveSystem(self):
        ### Main window container
        self.root_window = tk.Tk()
        self.root_window.geometry("800x800")
        self.root_window.title("Interactive Analysis of GMM")
        
        self.root_frame = tk.Frame(self.root_window, width=800, height=800, bg="green")
        self.root_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.guiPlotsPanel()
        self.guiControlPanel()
        
        
        #----------------------------------- self.guiMotorPanel_reset_callback()
         
        self.root_window.mainloop()  
    
    def guiPlotsPanel(self):
        self.plots_frame = tk.Frame(self.root_frame, width=800, height=600, bg="white")
        self.plots_frame.pack(side=tk.TOP, fill=tk.X, expand=1)
        self.plots_container_frame = tk.Frame(self.plots_frame, width=800, height=600, bg="black")
        self.plots_container_frame.pack(side=tk.LEFT, fill=tk.NONE, expand=0)
        
        self.plots_fig = plt.Figure(figsize=(1.20,1.5), dpi=100)
        self.plots_fig.patch.set_facecolor('red')
        self.plots_canvas = FigureCanvasTkAgg(self.plots_fig, master=self.vt_frame) 
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
    
    '''

       

        
        self.fig_sound = plt.Figure(figsize=(6.00,1.5), dpi=100)
        self.canvas_sound = FigureCanvasTkAgg(self.fig_sound, master=self.sound_frame)
        self.canvas_sound.show()
        self.canvas_sound.get_tk_widget().pack(side="left",fill="none", expand=False)   
        self.ax_sound = self.fig_sound.add_subplot(111)
        pos1 = self.ax_sound.get_position() # get the original position 
        pos2 = [pos1.x0*0.9, pos1.y0*2.0,  pos1.width*1.1, pos1.height*0.9] 
        self.ax_sound.set_position(pos2) # set a new position
        self.ax_sound.autoscale(enable=True, axis='both', tight=None)
        self.canvas_sound.draw()

        self.btn_prev_vt = tk.Button(self.vt_sound_opt_frame, state=tk.DISABLED, text="<<", command = self.vt_shape_prev_callback)
        self.btn_next_vt = tk.Button(self.vtimport Tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler_sound_opt_frame, state=tk.DISABLED, text=">>", command = self.vt_shape_next_callback)
        self.btn_play_vt = tk.Button(self.vt_sound_opt_frame, state=tk.DISABLED, text="Play", command = self.play_callback)
        self.btn_prev_vt.pack(side=tk.TOP, fill=tk.X, expand=1)
        self.btn_next_vt.pack(side=tk.TOP, fill=tk.X, expand=1)
        self.btn_play_vt.pack(side=tk.TOP, fill=tk.X, expand=1)
        
        
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
        self.entry_m6.grid(row=0, column=3, padx=5, pady=5)
        self.entry_m7 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m7.grid(row=1, column=3, padx=5, pady=5)
        self.entry_m8 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m8.grid(row=2, column=3, padx=5, pady=5)

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
        self.entry_m9.grid(row=3, column=3, padx=5, pady=5)
        self.entry_m10 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m10.grid(row=4, column=3, padx=5, pady=5)
        self.entry_m11 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m11.grid(row=1, column=5, padx=5, pady=5)
        self.entry_m12 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m12.grid(row=2, column=5, padx=5, pady=5)
        self.entry_m13 = tk.Entry(self.motor_entries_frame, width=10)
        self.entry_m13.grid(row=3, column=5, padx=5, pady=5)                
        
                
        self.btn_execute_m = tk.Button(self.motor_frame, text="Execute", command = self.execute_callback)
        self.btn_execute_m.pack(side=tk.LEFT, fill=tk.NONE, expand=0)   
    '''   
    
  