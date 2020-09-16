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
RUN_EWC = False
EWC_LAM = 15
N_FIM_SAMPLES = 500
MINIBATCH_SIZE = 250

STEP_DISP = 100
VERBOSE = True

# ----------------------------------------------------------------------------------------
# network class and helper functions 
# ----------------------------------------------------------------------------------------

def var_weights(shape,std=1e-3):
    return tf.Variable(tf.truncated_normal(shape,stddev=std))

def var_bias(shape,const=0.01):
    return tf.Variable(tf.constant(const,shape=shape))

class Nnet(object):
    def __init__(self,sess):
        self.x_features = tf.placeholder(tf.float32, [None, N_INPUTS], name='x_in')
        self.y_true = tf.placeholder(tf.float32, [None, N_CLASSES], name='y_true')

        # nnet 
        self.w_hf = var_weights((N_INPUTS,N_HIDDEN),std=np.sqrt(WEIGHT_INIT))
        self.b_hf = var_bias((1,N_HIDDEN))

        self.x_h = tf.add(tf.matmul(self.x_features,self.w_hf),self.b_hf)
        self.y_h = tf.nn.relu(self.x_h)

        self.w_o = var_weights((N_HIDDEN,N_CLASSES),std=1/np.sqrt(N_HIDDEN))
        self.b_o = var_bias((1,N_CLASSES))
        self.logits = tf.add(tf.matmul(self.y_h,self.w_o),self.b_o)

        self.thetas = [self.w_hf, self.b_hf, self.w_o, self.b_o]
        self.sess = sess

        # xent loss for multiclass 
        if N_CLASSES>1:
            self.y_pred = tf.nn.softmax(self.logits)
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true, logits=self.logits))
            correct_prediction = tf.equal(tf.argmax(self.logits,1), tf.argmax(self.y_true,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        else:
            self.y_pred = tf.nn.sigmoid(self.logits)
            self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_true,logits=self.logits))
            # self.cross_entropy = tf.reduce_mean(tf.multiply(self.y_true,tf.log(self.y_pred+1e-10))+tf.multiply(1-self.y_true,tf.log(1-self.y_pred+1e-10)))
            correct_prediction = tf.equal(self.y_pred, self.y_true)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.set_loss_funct()

    def switch_to_ewc(self,x_data):
        # copy old parameter values 
        self.copy_thetas()
        # compute (diagonal of ) fisher information matrix 
        self.compute_diag_fim(x_data)
        # change loss function 
        self.set_loss_funct(run_ewc=True)
        
    def set_loss_funct(self,run_ewc=False):
        if run_ewc==True:
            self.ewc_loss = self.compute_ewc_loss()            
            print('hooray')
            self.train_step = tf.train.GradientDescentOptimizer(SGD_LRATE).minimize(self.ewc_loss)
        else:            
            self.train_step = tf.train.GradientDescentOptimizer(SGD_LRATE).minimize(self.cross_entropy)
        


    def copy_thetas(self):
        '''
            copy thetas of old task 
        '''
        self.old_thetas = []
        for ii in range(len(self.thetas)):
            self.old_thetas.append(self.thetas[ii].eval())
        

    def compute_ewc_loss(self):
        '''
            compute ewc regulariser 
        '''
        loss = self.cross_entropy
        for ii in range(len(self.thetas)):
            loss += (EWC_LAM/2) * tf.reduce_sum(tf.multiply(self.FIM[ii].astype(np.float32),(self.thetas[ii] - self.old_thetas[ii])**2))
        return loss


    def compute_diag_fim(self, x_data, n_samples=N_FIM_SAMPLES):
        '''
            compute diagonal fisher information matrix
        '''
        FIM = []
        for ii in range(len(self.thetas)):
            FIM.append(np.zeros(self.thetas[ii].get_shape().as_list()))
        # -- set-up graph nodes--
        # true fisher information: use predicted label
        if N_CLASSES > 0:
            c_index = tf.argmax(self.y_pred,1)[0]
        else:
            c_index = 0
        # get gradients wrt log likelihood
        compute_gradients = tf.gradients(tf.log(self.y_pred[0,c_index]),self.thetas)
        
        # -- compute FIM (expected squared score of ll) --
        for ii in range(n_samples):
            idx = np.random.randint(x_data.shape[0])
            grads = self.sess.run(compute_gradients, feed_dict={self.x_features:x_data[idx:idx+1,:]})
            # add squared gradients of score:
            for ii in range(len(self.thetas)):
                FIM[ii] +=  grads[ii]**2 # np.square(grads[ii])
            # normalise:
            self.FIM  = [FIM[ii]/n_samples for ii in range(len(self.thetas))]
       
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