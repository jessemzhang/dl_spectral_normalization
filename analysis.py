import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import cm 
from spectral_adversarial_regularization import adversarial as ad
from spectral_adversarial_regularization import dl_utils

def train_network(Xtr, Ytr, val_set, arch, save_dir, eps=0.3, adv=None,
                  gpu_prop=0.2, num_epochs=200, save_every=25, beta=1, retrain=True):
    """Wrapper around build_graph_and_train that takes into account retraining"""
    
    if retrain: os.system('rm -rf %s'%(save_dir))
        
    _ = dl_utils.build_graph_and_train(Xtr, Ytr, save_dir, arch,
                                       eps=eps,
                                       adv=adv,
                                       num_epochs=num_epochs,
                                       save_every=save_every,
                                       num_channels=1,
                                       batch_size=256,
                                       val_set=val_set,
                                       early_stop_acc=0.999,
                                       early_stop_acc_num=25,
                                       gpu_prop=gpu_prop,
                                       beta=beta)


def get_adv_acc_curve(X, Y, save_dir, arch, eps_list, order=2, method=ad.pgm, beta=1.,
                      load_epoch=25, num_channels=1):
    """Sweeps through a list of eps for attacking a network, generating an adv performance curve"""

    load_epoch = np.min((dl_utils.latest_epoch(save_dir), load_epoch))
    
    adv_accs = np.zeros(len(eps_list))
    acc = dl_utils.build_graph_and_predict(X, save_dir, arch, Y=Y, beta=beta,
                                           num_channels=num_channels, load_epoch=load_epoch)
    print('Acc on examples: %.4f'%(acc))
    for i, eps in enumerate(eps_list):
        adv_accs[i] = ad.test_net_against_adv_examples(X, Y, save_dir, arch, beta=beta,
                                                       num_channels=num_channels, method=method,
                                                       order=order, eps=eps, load_epoch=load_epoch)
    return acc, adv_accs


def plot_acc_curves(adv_results, x_vals, title='PGM attacks', sort_func=None, logy=True, logx=False):
    """ adv_results should be a set of curves generated from get_adv_acc_curve
        (i.e. it's a dictionary where keys describe different networks)
        Plots all of these curves against one another.
    """
    
    colors = cm.rainbow(np.linspace(0, 1, len(adv_results)))
    plt.figure(figsize=(10, 7))
    if sort_func is not None:
        keys_in_ord = sorted(adv_results, key=sort_func)
    else:
        keys_in_ord = sorted(adv_results)
    for i, k in enumerate(keys_in_ord):
        if len(adv_results[k]) == len(x_vals):
            plt.plot(x_vals, 1.-adv_results[k], c=colors[i], label=k)
        elif len(adv_results[k]) > 2:
            plt.plot(x_vals, 1.-adv_results[k][1], c=colors[i], label='%s (test acc %.3f, sn %.3e)'\
                     %(k, adv_results[k][0], adv_results[k][2]))
        else:
            plt.plot(x_vals, 1.-adv_results[k][1], c=colors[i], label='%s (test acc %.3f)'\
                     %(k, adv_results[k][0]))
            

    plt.xlabel(r'$\epsilon/C_2$')
    if logx:
        plt.xscale('log')
    plt.ylabel('Error')
    if logy:
        plt.ylim(1e-2, 1e0)
        plt.yscale('log')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.4, 1))
    plt.grid()
    plt.show()
    
    
def beta_sweep_curves(X, Y, adv, eps_list, arch1, arch2, load_epoch=25):
    """Generates a set of adv performance curves from networks of different betas"""

    adv_results = {}
    for f in sorted(os.listdir('save_weights/mnist/')):
        if adv in f and 'rand' not in f and 'backup' not in f and 'pickle' not in f:
            save_dir = os.path.join('save_weights', 'mnist', f)
            if 'beta' in f:
                arch = arch1
                beta = float(f.split('beta')[1])
            else:
                arch = arch2
                beta = 1.
                
            acc, adv_accs = get_adv_acc_curve(X, Y, save_dir, arch, eps_list, order=2,
                                              method=ad.pgm, beta=beta, load_epoch=load_epoch)
            adv_results[f] = (acc, adv_accs)
            
    return adv_results


def sort_func(f):
    """used for sorting the trained networks in terms of beta"""
    if 'beta' not in f:
        return 0
    return float(f.split('beta')[1].split('_')[0])


def get_curves_for_arch(data, labeltype, arch, methods, eps_list, archname, beta=1):
    """Generates a set of adv performance curves from networks of the same architecture
       but trained using different adversarial attacks
    """
    
    Xtr, Ytr, Xtt, Ytt = data[labeltype]
    
    adv_results = {}
    for method in methods:
        save_dir = os.path.join('save_weights', 'cifar10', archname,
                                '%s_%s_%s'%(archname, method, labeltype))
        s_norm = dl_utils.get_overall_sn(save_dir, arch, num_channels=3, load_epoch=200)
        tr_acc = dl_utils.build_graph_and_predict(Xtr, save_dir, arch, Y=Ytr, beta=beta,
                                                  num_channels=3, load_epoch=200)
        print('Acc on training examples: %.4f'%(tr_acc))
        acc, adv_accs = get_adv_acc_curve(Xtt, Ytt, save_dir, arch, eps_list, order=2,
                                          method=ad.pgm, beta=beta, load_epoch=200,
                                          num_channels=3)
        adv_results[method] = (acc, adv_accs, s_norm, tr_acc)
        
    return adv_results