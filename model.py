import tensorflow as tf 
import numpy as np

def var_weights(shape,std=1e-3):
    return tf.Variable(tf.truncated_normal(shape,stddev=std))

def var_bias(shape,const=0.01):
    return tf.Variable(tf.constant(const,shape=shape))


class Nnet(object):
    def __init__(self,sess,n_inputs=784,
                            n_classes=10,
                            n_hidden=100,
                            learning_rate=1e-1,
                            weight_init=1e-2,
                            ewc_lamb=15,
                            n_samples=500):
        self.x_features = tf.placeholder(tf.float32, [None, n_inputs], name='x_in')
        self.y_true = tf.placeholder(tf.float32, [None, n_classes], name='y_true')

        # nnet 
        self.w_hf = var_weights((n_inputs,n_hidden),std=np.sqrt(weight_init))
        self.b_hf = var_bias((1,n_hidden))

        self.x_h = tf.add(tf.matmul(self.x_features,self.w_hf),self.b_hf)
        self.y_h = tf.nn.relu(self.x_h)

        self.w_o = var_weights((n_hidden,n_classes),std=1/np.sqrt(n_hidden))
        self.b_o = var_bias((1,n_classes))
        self.logits = tf.add(tf.matmul(self.y_h,self.w_o),self.b_o)

        self.thetas = [self.w_hf, self.b_hf, self.w_o, self.b_o]
        self.sess = sess
        self.lrate=learning_rate 
        self.ewc_lambda = ewc_lamb
        self.fim_samples = n_samples
        self.n_classes = n_classes

        # xent loss for multiclass 
        if n_classes>1:
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
            if not hasattr(self,'ewc_loss'):
                self.ewc_loss = self.compute_ewc_loss()            
                print('hooray')
                self.train_step = tf.train.GradientDescentOptimizer(self.lrate).minimize(self.ewc_loss)
        else:            
            self.train_step = tf.train.GradientDescentOptimizer(self.lrate).minimize(self.cross_entropy)
        


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
            loss += (self.ewc_lambda/2) * tf.reduce_sum(tf.multiply(self.FIM[ii].astype(np.float32),(self.thetas[ii] - self.old_thetas[ii])**2))
        return loss


    def compute_diag_fim(self, x_data):
        '''
            compute diagonal fisher information matrix
        '''
        FIM = []
        for ii in range(len(self.thetas)):
            FIM.append(np.zeros(self.thetas[ii].get_shape().as_list()))
        # -- set-up graph nodes--
        # true fisher information: use predicted label
        if self.n_classes > 0:
            c_index = tf.argmax(self.y_pred,1)[0]
        else:
            c_index = 0
        # get gradients wrt log likelihood
        compute_gradients = tf.gradients(tf.math.log(self.y_pred[0,c_index]),self.thetas)
        
        # -- compute FIM (expected squared score of ll) --
        for ii in range(self.fim_samples):
            idx = np.random.randint(x_data.shape[0])
            grads = self.sess.run(compute_gradients, feed_dict={self.x_features:x_data[idx:idx+1,:]})
            # add squared gradients of score:
            for ii in range(len(self.thetas)):
                FIM[ii] +=  grads[ii]**2 # np.square(grads[ii])
            # normalise:
            self.FIM  = [FIM[ii]/self.fim_samples for ii in range(len(self.thetas))]
       