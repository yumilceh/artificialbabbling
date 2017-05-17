'''
Created on Sep 13, 2016

@author: Juan Manuel Acevedo Valle
'''

if __name__ == '__main__':
    import sys
    import itertools
    
    sys.path.append("../../")
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot  import draw, show

    from SensorimotorExploration.Models.GeneralModels.Trash.ILGMM_GREC import ILGMM
    from SensorimotorExploration.DataManager.PlotTools import initializeFigure

    # Number of samples per component
    n_samples = 500
    colors = ['k', 'r', 'g', 'b', 'c', 'm', 'y']
    color_iter = itertools.cycle(colors)
    
    # Generate random sample, two components
    np.random.seed(0)
    C = np.array([[0., -0.1], [1.7, .4]])
    X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
              .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]


    model = ILGMM(min_components=2)
    model.train(X)
    
    Y_  = model.model.predict(X)
    
            
    #Model computed with three Gaussians
    fig1, ax1 = initializeFigure()
    fig1, ax1 = model.plotGMMProjection(fig1,ax1,0,1)
    
    #- for i, (mean,  color) in enumerate(zip(model.model.means_,  color_iter)):
        #----------- plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)
        
    plt.scatter(X[:, 0], X[:, 1], 0.8, color='k')
             
    n_samples2=150
    C2 = np.array([[0., -0.1], [0.2, 0.4]])
    X2 = np.r_[np.dot(np.random.randn(n_samples, 2), 0.5*C),
              .2 * np.random.randn(n_samples2, 2) + np.array([-2, 1])]
    plt.scatter(X2[:, 0], X2[:, 1], 0.8, color='r')
    
    model2 = ILGMM(min_components=2)
    model2.train(X2)
    
    fig1, ax1 = model2.plotGMMProjection(fig1,ax1,0,1)

    ax1.relim()
    ax1.autoscale_view()

    model.train(X2)
        
    #Model merging similar Gaussians
    fig2, ax2 = initializeFigure()
    fig2, ax2 = model.plotGMMProjection(fig2, ax2, 0, 1)
    ax2.relim()
    ax2.autoscale_view()   
    
    
    
    draw()
    show()
    
    