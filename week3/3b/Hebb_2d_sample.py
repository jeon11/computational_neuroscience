import numpy as np
import matplotlib.pyplot as plt
import numpy.random as r

n_pats = 400;

# start by creating points using a uniform random distribution
# we will interpret these numbers as radians
radi = r.rand(n_pats)
# say we want a range from pi/2 to pi/4
# calculate how wide this range is
patrange = np.pi/2 - np.pi/4;
# then multiply our original random numbers by this range
scaled_radi = radi * patrange;

# we shift by adding min value
# now our numbers are radians in the desired range
final_radi = scaled_radi + np.pi/4;

# finally, we can convert from radians to 2d patterns
# cosine gives the position on the x-axis
# sine gives the position on the y-axis
pats = np.zeros([2,n_pats])
pats[0,:] = np.cos(final_radi);
pats[1,:] = np.sin(final_radi);

# we can visualize our patterns
# plt.subplot(1,2,1)
plt.plot(pats[0,:100],pats[1,:100],'.')
circle = plt.Circle((0, 0), 1, color='b', fill=False)
ax = plt.gca()
ax.add_artist(circle)

# guidelines
plt.plot([0,0],[-1,1],'k--')
plt.plot([-1,1],[0,0],'k--')

# train network
n_out = 1
n_in = 2

n_training = n_pats
out_act = np.ones([n_out,n_training])
W = np.zeros([n_out,n_in])
# initialize weights
W[0,0] = np.cos(np.pi/8);
W[0,1] = np.sin(np.pi/8);
plt.plot(W[0,0],W[0,1],'rs')

lrate = 0.05
rec_W = np.zeros([2,n_training])
# the wts grow without bound when it is simple hebb
# but weight normalization stops that
# could also use Oja formalization of dW which builds the wt normalization into the dW equation
for t in range(n_training):
        dW = lrate * pats[:,t] * out_act[:,t]
        W = W + dW
        # normalize weights
        for i in range(n_out):
                W[i,:] = W[i,:] / np.linalg.norm(W[i,:])
        # save wt trajectories
        rec_W[:,t] = W


plt.plot(W[0,0],W[0,1],'ro')

plt.axis('square')
plt.axis([-1.05,1.05,-1.05,1.05])
plt.xlabel('unit 1 activation')
plt.ylabel('unit 2 activation')

plt.savefig('unit_circle_patterns.eps')

# plot the weight trajectories over the course of training
plt.clf()

plt.plot(range(n_training),rec_W[0,:],'k-')
plt.plot(range(n_training),rec_W[1,:],'r-')

plt.xlabel('training time step')
plt.ylabel('weight value')
plt.savefig('unit_circle_weights.eps')