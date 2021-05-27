'''
implementation of Elastic Weight Consolidation (Kirkpatrick et al, 2017)
Timo Flesch, 2021
'''
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from ewc_lib.model import Nnet
from ewc_lib.trainer import train_nnet
from ewc_lib.visualise import disp_results


# ----------------------------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------------------------

# define a few variables 
params = {}
params['n_inputs'] = 784
params['n_classes'] = 10
params['n_hidden'] = 100
params['weight_init'] = 1e-5

params['n_iters'] = 5000

params['lrate'] = 1e-1
params['do_ewc'] = True
params['ewc_lambda'] = 15
params['fim_samples'] = 1000
params['mbatch_size'] = 250

params['disp_n_steps'] = 100
params['verbose'] = True

params['task'] = 'permutedMNIST' # permutedMNIST or splitMNIST 


params['device'] = 'CPU'

if params['device']=='CPU':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ----------------------------------------------------------------------------------------
# main experiment
# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    results = train_nnet(params)
    disp_results(results,params['do_ewc'])
