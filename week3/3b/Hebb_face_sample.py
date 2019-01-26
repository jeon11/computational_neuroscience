import numpy as np
import matplotlib.pyplot as plt
import numpy.random as r

# update this path to point to the 4 images I provided
path = '/Users/polyn/Teaching/NSC3270/NSC3270_Spr19/Homework/HW4_hebb/sample_code/faces/'
# grab the pictures 
stewart = plt.imread(path+'stewart.png')
fisher = plt.imread(path+'fisher.png')
obama = plt.imread(path+'obama.png')
cranston = plt.imread(path+'cranston.png')

n_pats = 4
n_dims = np.prod(fisher.shape)

# these have vals ranging [0 1]
# list with the 4 image matrices
face = [fisher,stewart,obama,cranston]

plt.imshow(fisher,'gray')
#plt.show()

# turn each image into a pattern ranging [-1 1]
pats = np.zeros([n_dims,n_pats])

for i in range(n_pats):
   temp = face[i].reshape([n_dims])
   pats[:,i] = (temp*2)-1
   # normalize each pat to be unit length
   pats[:,i] = pats[:,i] / np.linalg.norm(pats[:,i])

# create noisy versions of the image patterns 
n_noisy = 2000
noise_scale = 0.07
noisy_pats = np.zeros([n_dims,n_noisy*n_pats])
out_acts = np.zeros([n_pats,n_noisy*n_pats])
out_labels = np.zeros([n_noisy*n_pats])
#training_pat = 0
pat_count = 0

# creating the input and output patterns
for j in range(n_noisy):
   for i in range(n_pats):
      this_pat = i      
      unit_noise = r.randn(n_dims)*noise_scale
      noisy_pats[:,pat_count] = pats[:,this_pat] + unit_noise
      # normalize activation to unit length
      noisy_pats[:,pat_count] = noisy_pats[:,pat_count] / np.linalg.norm(noisy_pats[:,pat_count])
      out_acts[this_pat,pat_count] = 1
      out_labels[pat_count] = this_pat
      pat_count += 1

# display some of the noisy pats
# figure(1); clf;
for i in range(16):
   plt.subplot(4,4,i+1)
   this = noisy_pats[:,i].reshape([32,32])
   # convert back to [0 1]
   this = (this+1)/2
   plt.imshow(this,'gray')
   if i<4:
      plt.title('Face '+str(int(out_labels[i])))
   plt.tick_params(axis='both',which='both',
          bottom=False,top=False,labelbottom=False,right=False,left=False,labelleft=False)   

plt.savefig('noisy_images.eps')

# image vector space to a single binary output unit
# hebbian learning on noisy pats
n_out = 4
n_in = 1024

n_training = n_noisy*n_pats
#out_act = np.ones([n_out,n_noisy])
# initialize as random wts
W = r.randn(n_out,n_in)
for i in range(n_out):
   W[i,:] = W[i,:] / np.linalg.norm(W[i,:])

lrate = 0.01
# they grow without bound when it is simple hebb
# but weight normalization stops that
for t in range(n_training):
   # non-vectorized version of Hebbian rule, steps through each synapse individually
   for i in range(n_in):
      for j in range(n_out):
         dW = lrate * noisy_pats[i,t] * out_acts[j,t]
         W[j,i] = W[j,i] + dW  
   # normalize weights
   for i in range(n_out):
      W[i,:] = W[i,:] / np.linalg.norm(W[i,:])
    

# visualize the weights
plt.clf()
for i in range(n_pats):
   plt.subplot(1,4,i+1)
   plt.imshow(W[i,:].reshape([32,32]),'gray')
   plt.tick_params(axis='both',which='both',
          bottom=False,top=False,labelbottom=False,right=False,left=False,labelleft=False)   

plt.savefig('recon_faces.eps')

# visualize the original images, pre-noise
for i in range(4):
   plt.subplot(1,4,i+1)
   plt.imshow(face[i],'gray')
   plt.tick_params(axis='both',which='both',
          bottom=False,top=False,labelbottom=False,right=False,left=False,labelleft=False)
plt.savefig('orig_faces.eps')

# calculate the incoming excitation for each of the face patterns after
# training on carrie fisher
incoming_faces = np.zeros([n_pats])

for i in range(4):
   incoming_faces[i] = np.dot(W[0,:],pats[:,i])


# show the distribution of incoming activation values for the noisy
# training patterns
incoming_noisy = np.zeros([n_noisy])
for i in range(n_noisy):
   incoming_noisy[i] = np.dot(W[0,:],noisy_pats[:,i])

# set a breakpoint here if you want access to the variables
print('done!')



    

