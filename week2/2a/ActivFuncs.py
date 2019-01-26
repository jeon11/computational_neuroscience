# examples of Keras / TensorFlow Activation Function
#

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras.activations as kact
import tensorflow as tf

# open an interactive TensorFlow session
session = tf.InteractiveSession()

def plt_act(net, act, title):
    sns.set()
    plt.plot(net, act)
    plt.ylabel('act', fontsize=20)
    plt.xlabel('net', fontsize=20)
    plt.title(title, fontsize=20)
    plt.savefig('./{0}.jpg'.format(title), dpi=300)
    plt.close()

# create a numpy array - TensorFlow defaults to single precision floating point
netnp = np.linspace(-5.0, 5.0, 1000, dtype='float32')
# convert to a TensorFlow tensor
nettf = tf.convert_to_tensor(netnp)

# linear activation function
acttf = kact.linear(nettf)
# need to convert from TensorFlow tensors to numpy arrays before plotting
# eval() is called because TensorFlow tensors have no values until they are "run"
plt_act(nettf.eval(), acttf.eval(), 'linear activation function')

# relu activation function
acttf = kact.relu(nettf)
plt_act(nettf.eval(), acttf.eval(), 'rectified linear (relu)')

# sigmoid activation function
acttf = kact.sigmoid(nettf)
plt_act(nettf.eval(), acttf.eval(), 'sigmoid')

# hard sigmoid activation function
acttf = kact.hard_sigmoid(nettf)
plt_act(nettf.eval(), acttf.eval(), 'hard sigmoid')

# tanh activation function
acttf = kact.tanh(nettf)
plt_act(nettf.eval(), acttf.eval(), 'tanh')

# softsign activation function
acttf = kact.softsign(nettf)
plt_act(nettf.eval(), acttf.eval(), 'softsign')

# close the TensorFlow session
session.close()

# done
print('Done!')







