import numpy as np
import matplotlib.pyplot as plt
import numpy.random as r

n_vals = 2000
target_cov = np.matrix([[1.0, 0.7],[0.7,0.6]])
vals = r.multivariate_normal([0,0],target_cov,[n_vals])

plt.plot(vals[:1000,0],vals[:1000,1],'k.')
plt.plot([0,0],[-3,3],'k--')
plt.plot([-3,3],[0,0],'k--')
plt.xlabel('input unit 1 activity')
plt.ylabel('input unit 2 activity')

plt.savefig('pca_mvn.eps')

lrate = 0.01
out_act = np.zeros([n_vals])
W = np.zeros([2,n_vals])
W[0,0] = 0.0
W[1,0] = 0.3
# save the weights to see if they stabilize
for i in range(n_vals):
        out_act[i] = np.dot(W[:,i],vals[i,:])
        dW = lrate * out_act[i] * (vals[i,:] - W[:,i] * out_act[i])
        if i < (n_vals-1):
                W[:,i+1] = W[:,i] + dW

plt.plot([0,W[0,n_vals-1]],[0,W[1,n_vals-1]],'r-')
plt.plot(W[0,1],W[1,1],'rx')
circle = plt.Circle((0, 0), 1, color='b', fill=False)
ax = plt.gca()
ax.add_artist(circle)

plt.savefig('pca_mvn_wt.eps')

plt.clf()
plt.plot(range(n_vals),W[0,:],'b-')
plt.plot(range(n_vals),W[1,:],'g-')
plt.xlabel('training iteration')
plt.ylabel('weight value')
plt.savefig('pca_wt1.eps')

plt.clf()
#plt.plot(range(n_vals),out_act,'k-')
plt.plot(range(500),out_act[:500],'k-')
plt.xlabel('training iteration')
plt.ylabel('output activity')
plt.savefig('pca_outact.eps')

#plt.clf()


print('hi!')