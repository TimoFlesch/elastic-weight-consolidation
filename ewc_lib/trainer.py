import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf 
from ewc_lib.data import load_mnist_data, permute_mnist
from ewc_lib.model import Nnet


def train_step(nnet,mbatch):    
    nnet.train_step.run(feed_dict={nnet.x_features:mbatch[0],nnet.y_true:mbatch[1]})

def eval_step(nnet,dataset1,dataset2,ep,results,verbose):
    results['acc1'].append(nnet.accuracy.eval(feed_dict={nnet.x_features:dataset1.test.images,nnet.y_true:dataset1.test.labels}))
    results['acc2'].append(nnet.accuracy.eval(feed_dict={nnet.x_features:dataset2.test.images,nnet.y_true:dataset2.test.labels}))
    if verbose:
        print('episode {:d} on 1st task: accuracy 1st task {:2f}, accuracy 2nd task {:2f}'.format(ep,results['acc1'][-1],results['acc2'][-1]))


def train_nnet(params):
    '''
    trains neural network 
    '''
    # init variables 
    results = {
        'acc1': [],
        'acc2': []
    }
    # load dataset 
    dataset1 = load_mnist_data()
    # now create permuted mnist 
    dataset2 = permute_mnist(dataset1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        # initialise neural network 
        nnet = Nnet(sess,n_inputs=params['n_inputs'],
                            n_classes=params['n_classes'],
                            n_hidden=params['n_hidden'],
                            learning_rate=params['lrate'],
                            weight_init=params['weight_init'],
                            ewc_lamb=params['ewc_lambda'],
                            n_samples=params['fim_samples'])
        sess.run(tf.global_variables_initializer())

        # train on first task 
        for ep in range(params['n_iters']):
            train_step(nnet,dataset1.train.next_batch(params['mbatch_size']))
            if ep%params['disp_n_steps']==0:
                eval_step(nnet,dataset1,dataset2,ep,results,params['verbose'])
        
        # train on second task
        if params['do_ewc']:
            # ... run network with ewc:            
            nnet.switch_to_ewc(dataset1.train.images)

        for ep in range(params['n_iters']):
            train_step(nnet,dataset2.train.next_batch(params['mbatch_size']))
            if ep%params['disp_n_steps']==0:
                eval_step(nnet,dataset1,dataset2,ep,results,params['verbose'])

    return results
