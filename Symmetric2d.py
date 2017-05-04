import GPy
import numpy as np
from symmetric import Symmetric

k = GPy.kern.RBF(2, variance = 2, lengthscale=0.25)


permutation = np.array([[0,1],[1,0]])
k2 = Symmetric(k, permutation)



##################################################################
##  Now try simulating from the prior on a grid

xgrid1, xgrid2 = np.meshgrid(np.arange(0,2,0.05), np.arange(0,2,0.05))
xgrid =np.concatenate((xgrid1.reshape((-1,1)), xgrid2.reshape((-1,1))), 1)

mu = np.zeros(xgrid.shape[0])

K3 = k2.K(xgrid)+np.eye(xgrid.shape[0])*0.000001
Z = np.random.multivariate_normal(mu,K3,1)

import matplotlib
import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
CS = plt.contour(xgrid1[0,:], xgrid2[:,0], Z.reshape(xgrid1.shape[0], xgrid1.shape[0]))
plt.clabel(CS, inline=1, fontsize=10)



### Now lets check we can update the model

keep = np.random.choice(Z.shape[1], 100)#[:,0]>x[:,1]
x= xgrid[keep,:]#x[keep,:]
keep2 = x[:,0]>x[:,1]
x = x[keep2,:]
y = Z[0,keep]
y = y[keep2]
y = y.reshape(-1,1)


m=GPy.models.GPRegression(x, y, k2)
m.Gaussian_noise.fix(0.001)

plt.scatter(x[:,0], x[:,1])
# but is it working?

for i in np.arange(y.shape[0]):
    plt.annotate('(%.2f)' % y[i], xy=x[i,:], textcoords='offset points')

plt.title('True surface')


########
m.randomize()
m.optimize_restarts(10)
print(m)


ypredmean, ypredvar = m._raw_predict(xgrid, full_cov=True)
 # have to dp full cov as not implemented Kdiag yet
plt.figure(2)
CS2 = plt.contour(xgrid1[0,:], xgrid2[:,0], ypredmean.reshape(xgrid1.shape[0], xgrid1.shape[0]))
plt.scatter(x[:,0], x[:,1])
# but is it working?
plt.clabel(CS2, inline=1, fontsize=10)
for i in np.arange(y.shape[0]):
    plt.annotate('(%.2f)' % y[i], xy=x[i,:], textcoords='offset points')



plt.plot([0, 2], [0, 2], 'k-', lw=2)
plt.title('Predicted mean')





plt.figure(3)
CS3 = plt.contour(xgrid1[0,:], xgrid2[:,0], ypredvar.diagonal().reshape(xgrid1.\
shape[0], xgrid1.shape[0]))
plt.scatter(x[:,0], x[:,1])
# but is it working?
plt.clabel(CS3, inline=1, fontsize=10)
plt.title('Variance')
