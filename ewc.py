'''
implementation of Elastic Weight Consolidation (Kirkpatrick et al, 2017)
Timo Flesch, 2020
'''
import tensorflow as tf 
import numpy as np 
import os 
import matplotlib.pyplot as plt

from datetime import datetime 
from sklearn.utils import shuffle 
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data

from model import Nnet


# ----------------------------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------------------------

# define a few variables 
N_INPUTS = 784
N_CLASSES = 10
N_HIDDEN = 100
WEIGHT_INIT = 1e-2

N_ITERS = int(1e4)

SGD_LRATE = 1e-1
RUN_EWC = True
EWC_LAM = 15
N_FIM_SAMPLES = 500
MINIBATCH_SIZE = 250

STEP_DISP = 100
VERBOSE = True


# ----------------------------------------------------------------------------------------
# data functions
# ----------------------------------------------------------------------------------------
def load_mnist_data():    
    return  input_data.read_data_sets('MNIST_data', one_hot=True)



def permute_mnist(mnist):
    '''
    # return a new mnist dataset w/ pixels randomly permuted (found this funct in Ari Seff's github)
    '''
    perm_inds = np.arange(mnist.train.images.shape[1])
    np.random.shuffle(perm_inds)
    mnist2 = deepcopy(mnist)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(mnist2, set_name) # shallow copy
        this_set._images = np.transpose(np.array([this_set.images[:,c] for c in perm_inds]))
    return mnist2

# ----------------------------------------------------------------------------------------
# result visualiser
# ----------------------------------------------------------------------------------------
def disp_results(results):
    '''
    displays results as time series of test accuracies 
    '''
    plt.figure(1)
    plt.plot(results['acc1'],'k-',linewidth=2)
    plt.plot(results['acc2'],'r-',linewidth=2)    
    plt.xlabel('iter')
    plt.ylabel('test accuracy')
    plt.legend(['first task','second task'])
    if RUN_EWC:
        plt.title('Elastic Weight Consolidation')
    else:
        plt.title('Vanilla SGD')
    plt.grid()
    plt.show()

# ----------------------------------------------------------------------------------------
# trainer
# ----------------------------------------------------------------------------------------
def train_nnet():
    # init variables 
    results = {
        'acc1': [],
        'acc2': []
    }
    # load dataset 
    dataset1 = load_mnist_data()
    # now create permuted mnist 
    dataset2 = permute_mnist(dataset1)
    with tf.Session() as sess:
        # initialise neural network 
        nnet = Nnet(sess)
        sess.run(tf.global_variables_initializer())
        # train on mnist 
        for ep in range(N_ITERS):
            mbatch = dataset1.train.next_batch(MINIBATCH_SIZE)
            nnet.train_step.run(feed_dict={nnet.x_features:mbatch[0],nnet.y_true:mbatch[1]})
            if ep%STEP_DISP==0:
                results['acc1'].append(nnet.accuracy.eval(feed_dict={nnet.x_features:dataset1.test.images,nnet.y_true:dataset1.test.labels}))
                results['acc2'].append(nnet.accuracy.eval(feed_dict={nnet.x_features:dataset2.test.images,nnet.y_true:dataset2.test.labels}))
                if VERBOSE:
                    print('episode {:d} on 1st task: accuracy 1st task {:2f}, accuracy 2nd task {:2f}'.format(ep,results['acc1'][-1],results['acc2'][-1]))
        
        if RUN_EWC:
            # ... run network with ewc:            
            nnet.switch_to_ewc(dataset1.train.images)
            for ep in range(N_ITERS):
                mbatch = dataset2.train.next_batch(MINIBATCH_SIZE)
                nnet.train_step.run(feed_dict={nnet.x_features:mbatch[0],nnet.y_true:mbatch[1]})
                if ep%STEP_DISP==0:
                    results['acc1'].append(nnet.accuracy.eval(feed_dict={nnet.x_features:dataset1.test.images,nnet.y_true:dataset1.test.labels}))
                    results['acc2'].append(nnet.accuracy.eval(feed_dict={nnet.x_features:dataset2.test.images,nnet.y_true:dataset2.test.labels}))
                    if VERBOSE: 
                        print('episode {:d} on 2nd task: accuracy 1st task {:2f}, accuracy 2nd task {:2f}'.format(ep,results['acc1'][-1],results['acc2'][-1]))
            
        else:
            # ... run vanilla sgd: 
            for ep in range(N_ITERS):
                mbatch = dataset2.train.next_batch(MINIBATCH_SIZE)
                nnet.train_step.run(feed_dict={nnet.x_features:mbatch[0],nnet.y_true:mbatch[1]})
                if ep%STEP_DISP==0:
                    results['acc1'].append(nnet.accuracy.eval(feed_dict={nnet.x_features:dataset1.test.images,nnet.y_true:dataset1.test.labels}))
                    results['acc2'].append(nnet.accuracy.eval(feed_dict={nnet.x_features:dataset2.test.images,nnet.y_true:dataset2.test.labels}))
                    if VERBOSE:
                        print('episode {:d}on 2nd task: accuracy 1st task {:2f}, accuracy 2nd task {:2f}'.format(ep,results['acc1'][-1],results['acc2'][-1]))

    return results


# ----------------------------------------------------------------------------------------
# main experiment
# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    results = train_nnet()
    disp_results(results)
