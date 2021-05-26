import numpy as np
import tensorflow as tf 
from copy import deepcopy
from sklearn.utils import shuffle 
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes


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



def gen_splitMNIST(bounds):
    '''
    returns a subsection of the mnist dataset, only containing numbers within bounds
    - bounds: upper and lower bound (incl.) of numbers to include 

    Example: gen_splitMNIST([0,4]) would return subset of mnist that only contains numbers 0 to 4
    '''
    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    sets = ["train", "validation", "test"]
    sets_list = []
    for set_name in sets:
        this_set = getattr(dataset, set_name)
        maxlabels = np.argmax(this_set.labels, 1)
        sets_list.append(DataSet(this_set.images[((maxlabels >= bounds[0]) & (maxlabels <= bounds[1])),:],
                                this_set.labels[((maxlabels >= bounds[0]) & (maxlabels <= bounds[1]))],
                                 dtype=dtypes.uint8, reshape=False))
    return base.Datasets(train=sets_list[0], validation=sets_list[1], test=sets_list[2])
