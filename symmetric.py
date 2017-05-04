# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import itertools
from paramz.caching import Cache_this
from GPy.kern import Kern
from functools import reduce

class Symmetric(Kern):
    """
    Add given list of kernels together.
    propagates gradients through.

    This kernel will take over the active dims of it's subkernels passed in.
    """
    def __init__(self, basekernel, permutation):
        """     
        """
        assert isinstance(basekernel, Kern)
        self.basekernel = basekernel
        self.permutation = permutation
        input_dim = basekernel.input_dim
        active_dims = basekernel.active_dims
        # initialize the kernel with the full input_dim   
        super(Symmetric, self).__init__(input_dim, active_dims, name='Symmetric_'+basekernel.name)
        self.link_parameters(basekernel)
        self.number_permutations = permutation.shape[0]
        assert permutation.max()+1 <=  input_dim # don't allow permutations of columns that don't exist
        assert permutation.shape[1]==input_dim
        
    def Kdiag(self, X):
        # this defeats the point but will work for the moment.
        X2 = X
#        return reduce(np.add, (self.basekernel.K(X[:,self.permutation[i]], X2) for i in np\.arange(self.number_permutations)))
        return self.K(X).diagonal()
        # Problem: how can we do this efficiently? problem is that there is no Kdiag for basekernel coded that accepts arguments X and X2. This makes prediction very slow if N is large

    def K(self, X, X2=None):
        """       
                  Description                                                              
        """
        if X2 is None:  ## required in order to swap the columns - otherwise nothing changes
            X2 = X
        return reduce(np.add, (self.basekernel.K(X[:,self.permutation[i]], X2) for i in np.arange(self.number_permutations)))

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None: # required in order to swap the columns - otherwise nothing changes
            X2=X
        for i in np.arange(self.number_permutations):
            self.basekernel.update_gradients_full(dL_dK, X[:,self.permutation[i]], X2)            
            if i==0:
                G = self.basekernel.gradient
            else:
                G =+ self.basekernel.gradient 
        self.gradient = G


#    def update_gradients_diag(self, dL_dKdiag, X):
#        k = self.Kdiag(X)*dL_dKdiag
#        for p in self.parts:
#            p.update_gradients_diag(k/p.Kdiag(X),X)



# To do:
# update_gradients_diag
# Kdiag is currently just a diagonalization of K - this is very inefficient, but I can't think how else to do it.
# any other functions that need updating? sparse? gradient_x?
# add some checks that permutations are indeed permutations, and that all elements of the group are provided?
# For inputs that are symmetric, a covariance function with the same length-scales must be provided. I'm doing this by multiplying several isotropic kernels together at the moment. It may be possible to do it by tying parameters together instead
# I've only checked the results for the RBF kernel - check kernels work as well.
