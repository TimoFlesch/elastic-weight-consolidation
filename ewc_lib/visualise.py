import matplotlib.pyplot as plt

def disp_results(results,run_ewc):
    '''
    displays results as time series of test accuracies 
    '''
    plt.figure(1)
    plt.plot(results['acc1'],'k-',linewidth=2)
    plt.plot(results['acc2'],'r-',linewidth=2)    
    plt.xlabel('iter')
    plt.ylabel('test accuracy')
    plt.legend(['first task','second task'])
    if run_ewc:
        plt.title('Elastic Weight Consolidation')
    else:
        plt.title('Vanilla SGD')
    plt.grid()
    plt.show()