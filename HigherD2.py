import GPy
import numpy as np
from symmetric import Symmetric

#k = GPy.kern.RBF(6, variance = 2, lengthscale=0.25)

k1 = GPy.kern.RBF(2, variance=2, lengthscale=0.8, active_dims=[0,1])
k2 = GPy.kern.RBF(4, variance=2, lengthscale=0.2, active_dims=[2,3,4,5])
k = k1*k2



permutation = np.array([[0,1,2,3,4,5],[1,0,3,2, 5,4], [0,1,4,5,2,3],[1,0,5,4,3,2]])
ksym = Symmetric(k, permutation)

#
from pyDOE import lhs

x = lhs(6, samples=10) # not a grid,...
x2 = lhs(6, samples=13)

K1 = k.K(x, x2)

# check k(px,x') = k(x, px')
K2 = k.K(x[:,permutation[1]],x2)
K2n = k.K(x,x2[:,permutation[1]])
np.max(np.abs(K2-K2n))

# check k(p1x, p2 x') = k(p2x, p1x')
K4 =  k.K(x[:, permutation[2]],x2[:,permutation[3]])
K4n =  k.K(x[:, permutation[3]],x2[:,permutation[2]])
np.max(np.abs(K4-K4n))

K4 =  k.K(x[:, permutation[1]],x2[:,permutation[3]])
K4n =  k.K(x[:, permutation[3]],x2[:,permutation[1]])
K4==K4n
np.max(np.abs(K4-K4n))

# check k(p1x, x') = k(p2 x, p3 x')

K3 = k.K(x[:, permutation[1]],x2)
K3n =  k.K(x[:, permutation[2]],x2[:,permutation[3]])
K3-K3n
np.max(np.abs(K3-K3n))

#

from pyDOE import lhs
xgrid = lhs(6, samples=1000)*2. # not a grid,...


mu = np.zeros(xgrid.shape[0])

K3 = k2.K(xgrid)+np.eye(xgrid.shape[0])*0.000001
Z = np.random.multivariate_normal(mu,K3,1)

# pick a random selection for the data
keep = np.random.choice(Z.shape[1], 800)#[:,0]>x[:,1]

x= xgrid[keep,:]#x[keep,:]
keep2 = np.logical_and(x[:,0]>x[:,1], x[:,2]>x[:,3])
x = x[keep2,:]
y = Z[0,keep]
y = y[keep2]
y = y.reshape(-1,1)

m=GPy.models.GPRegression(x, y, ksym)
m.Gaussian_noise.fix(0.001)




m.optimize_restarts(10)
print(m)


testgrid = np.concatenate((xgrid, xgrid[:,permutation[1]],
xgrid[:,permutation[2]],
xgrid[:,permutation[3]]), 0)

ypredmean, ypredvar = m._raw_predict(testgrid)

index = np.arange(1000)
print(np.max(np.abs(ypredmean[index,:] -ypredmean[index+1000,:])))
print(np.max(np.abs(ypredmean[index,:] -ypredmean[index+2000,:])))
print(np.max(np.abs(ypredmean[index,:] -ypredmean[index+3000,:])))
