'''
Created on Jun 3, 2016

@author: Juan Manuel Acevedo Valle
Original from:
https://github.com/flowersteam/explauto/blob/master/explauto/sensorimotor_model/imle.py
'''

'''
    In this script there are 4 classes:
            Normalizer: Normilize the data used to train the IMLE model using maximum and minimum values for the variables provided withing the conf structure.
                Methods: normalize, denormalize
                
            IMLE: Wraps the IMLE class, provides inference support, and conversion from IMLE to GMM
                Methods: infer, update, to_gmm
            
            ImleGmmModel: Uses GMM obtained from Imle for inference purposes
                Methods: update_gmm, infer
'''


from numpy import argmax, array, zeros, ones
import imle
from explauto.models.gmminf import GMM

class Normalizer(object):
    def __init__(self, params):
        self.params=params
        self.params.ranges=self.params.max - self.params.min
        
        
    def normalize(self, data, dims):
        return (data - self.params.min[dims]) / self.params.ranges[dims] 

    def denormalize(self, data, dims):
        return (data * self.params.ranges[dims]) + self.params.min[dims]

class IMLE(object):
    """
        This class wraps the IMLE model from Bruno Damas ( http://users.isr.ist.utl.pt/~bdamas/IMLE ) into a sensorimotor model class to be used by based on the wrapper of Explauto (Flowers)
        """
    # def __init__(self, m_dims, s_dims, sigma0, psi0, mode='explore'):
    def __init__(self, params, mode='exploit', **kwargs_imle):
        """ :param list m_dims: indices of motor dimensions
            :param list_ndims: indices of sensory dimensions
            :param float sigma0: a priori variance of the linear models on motor dimensions
            :param list psi0: a priori variance of the gaussian noise on each sensory dimensions
            :param string mode: either 'exploit' or 'explore' (default 'explore') to choose if the infer(.) method will return the most likely output or will sample according to the output probability.
            .. note::
            """
        self.in_dims = params.in_dims
        self.out_dims = params.out_dims
        if 'sigma0' not in kwargs_imle:  # sigma0 is None:
            kwargs_imle['sigma0'] = 1./30. # (conf.m_maxs[0] - conf.m_mins[0]) / 30.
        if 'Psi0' not in kwargs_imle:  # if psi0 is None:
            kwargs_imle['Psi0'] = array([1./30.] * params.out_dims) ** 2 # ((conf.s_maxs - conf.s_mins) / 30.)**2
        self.mode = mode
        self.t = 0
        self.imle = imle.Imle(d=len(self.m_dims), D=len(self.s_dims), **kwargs_imle) #sigma0=sigma0, Psi0=psi0)
        self.normalizer = Normalizer(params)

    def infer(self, in_dims, out_dims, x_):
        x = self.normalizer.normalize(x_, in_dims)
        if in_dims == self.s_dims and out_dims == self.m_dims:
            # try:
            res = self.imle.predict_inverse(x, var=True, weight=True)
            sols = res['prediction']
            covars = res['var']
            weights = res['weight']
            # sols, covars, weights = self.imle.predict_inverse(x)
            if self.mode == 'explore':
                gmm = GMM(n_components=len(sols), covariance_type='full')
                gmm.weights_ = weights / weights.sum()
                gmm.covars_ = covars
                gmm.means_ = sols
                return self.normalizer.denormalize(gmm.sample().flatten(), out_dims)
            elif self.mode == 'exploit':
                # pred, _, _, jacob = self.imle.predict(sols[0])
                sol = sols[argmax(weights)]  # .reshape(-1,1) + np.linalg.pinv(jacob[0]).dot(x - pred.reshape(-1,1))
                return self.normalizer.denormalize(sol, out_dims)

            # except Exception as e:
            #     print e
            #     return self.imle.to_gmm().inference(in_dims, out_dims, x).sample().flatten()

        # elif in_dims == self.m_dims and out_dims==self.s_dims:
        #     return self.imle.predict(x.flatten()).reshape(-1,1)
        else:
            return self.normalizer.denormalize(self.imle.to_gmm().inference(in_dims, out_dims, x).sample().flatten(), out_dims)

    def update(self, x_, y_):
        x = self.normalizer.normalize(x_, self.conf.in_dims)
        y = self.normalizer.normalize(y_, self.conf.out_dims)
        self.imle.update(x, y)
    
    def to_gmm(self):
        n = self.number_of_experts
        gmm = GMM(n_components=n, covariance_type='full')
        gmm.means_ = zeros((n, self.d+self.D))
        gmm.covars_ = zeros((n, self.d+self.D, self.d+self.D))

        for k in range(n):
            gmm.means_[k, :] = self.get_joint_mu(k)
            gmm.covars_[k, :, :] = self.get_joint_sigma(k)
        gmm.weights_ = (1.*ones((n,)))/n
        return gmm


class ImleGmmModel(IMLE):
    def update_gmm(self):
        self.gmm = self.imle.to_gmm()

    def infer(self, in_dims, out_dims, x):
        self.update_gmm()
        return self.gmm.inference(in_dims, out_dims, x).sample().T

def make_priors(prior_coef):
    priors = {}
    for prior in ['wsigma', 'wSigma', 'wNu', 'wLambda', 'wPsi']:
        priors[prior] = prior_coef
    return priors


'''I am not sure what for is the stuff below'''
configurations = {'default': {}, 'low_prior': make_priors(1.), 'hd_prior': make_priors(10.)}
sensorimotor_models = {'imle': (IMLE, configurations)}