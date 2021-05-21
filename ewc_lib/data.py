import numpy as np
import tensorflow as tf 
from copy import deepcopy
from sklearn.utils import shuffle 
from tensorflow.examples.tutorials.mnist import input_data


def load_mnist_data():    
    return  input_data.read_data_sets('MNIST_data', one_hot=True)


def permute_mnist(mnist):
    '''
    return a new mnist dataset w/ pixels randomly permuted (found this funct in Ari Seff's github)
    '''
    perm_inds = np.arange(mnist.train.images.shape[1])
    np.random.shuffle(perm_inds)
    mnist2 = deepcopy(mnist)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(mnist2, set_name) # shallow copy
        this_set._images = np.transpose(np.array([this_set.images[:,c] for c in perm_inds]))
    return mnist2