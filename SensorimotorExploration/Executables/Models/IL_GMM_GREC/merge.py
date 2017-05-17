'''
Created on Sep 13, 2016

@author: Juan Manuel Acevedo Valle
'''

if __name__ == '__main__':
    import os,sys
    sys.path.append(os.getcwd())
    
    import numpy as np
    from matplotlib.pyplot  import draw, show

    from SensorimotorExploration.Models.GeneralModels.Trash.ILGMM_GREC import ILGMM
    from SensorimotorExploration.DataManager.PlotTools import initializeFigure

    # Number of samples per component
    n_samples = 500
    
    # Generate random sample, two components
    np.random.seed(0)
    C = np.array([[0., -0.1], [1.7, .4]])
    X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
              .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]


    model = ILGMM(min_components=3)
    model.train(X)
    
    #Model computed with three Gaussians
    fig1, ax1 = initializeFigure()
    ax1 = model.plot_gmm_projection(0,1,axes=ax1)
    
    model.mergeGaussians(0,2)
    
    #Model merging similar Gaussians
    fig2, ax2 = initializeFigure()
    ax2 = model.plot_gmm_projection(0, 1, axes=ax2)
    
    #Model computed with two Gaussians
    model = ILGMM(min_components=2)
    model.train(X)
    fig3, ax3 = initializeFigure()
    ax3 = model.plot_gmm_projection(0, 1, axes=ax3)
    
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    ax3.relim()
    ax3.autoscale_view()

    draw()
    show()
    
    