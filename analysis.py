import os
import pickle
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
                      load_epoch=25, num_channels=1, gpu_prop=0.2):
    """Sweeps through a list of eps for attacking a network, generating an adv performance curve"""

    load_epoch = np.min((dl_utils.latest_epoch(save_dir), load_epoch))
    
    adv_accs = np.zeros(len(eps_list))
    acc = dl_utils.build_graph_and_predict(X, save_dir, arch, Y=Y, beta=beta, gpu_prop=gpu_prop,
                                           num_channels=num_channels, load_epoch=load_epoch)
    print('Acc on examples: %.4f'%(acc))
    for i, eps in enumerate(eps_list):
        adv_accs[i] = ad.test_net_against_adv_examples(X, Y, save_dir, arch, beta=beta, gpu_prop=gpu_prop,
                                                       num_channels=num_channels, method=method,
                                                       order=order, eps=eps, load_epoch=load_epoch)
    return acc, adv_accs


def plot_acc_curves(adv_results, x_vals, title='PGM attacks', sort_func=None, logy=True, logx=False, ylim=(1e-2, 1e0)):
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
        plt.ylim(ylim)
        plt.yscale('log')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.4, 1))
    plt.grid(which="both")
    plt.show()
    
    
def beta_sweep_curves(X, Y, adv, eps_list, arch_sn, arch, load_epoch=25,
                      basedir='save_weights/mnist/'):
    """Generates a set of adv performance curves from networks of different betas"""
    
    adv_results = {}
    for f in sorted(os.listdir(basedir)):
        if adv in f and 'rand' not in f and 'backup' not in f and 'pickle' not in f:
            save_dir = os.path.join(basedir, f)
            if 'beta' in f:
                arch_ = arch_sn
                beta = float(f.split('beta')[1])
            else:
                arch_ = arch
                beta = 1.
                
            acc, adv_accs = get_adv_acc_curve(X, Y, save_dir, arch_, eps_list, order=2,
                                              method=ad.pgm, beta=beta, load_epoch=load_epoch)
            adv_results[f] = (acc, adv_accs)
            
    return adv_results


def sort_func(f):
    """used for sorting the trained networks in terms of beta"""
    if 'beta' not in f:
        return 0
    return float(f.split('beta')[1].split('_')[0])


def get_curves_for_arch(data, labeltype, arch, methods, eps_list, archname, beta=1,
                        maindir='save_weights/cifar10/'):
    """Generates a set of adv performance curves from networks of the same architecture
       but trained using different adversarial attacks
    """
    
    Xtr, Ytr, Xtt, Ytt = data[labeltype]
    
    adv_results = {}
    for method in methods:
        save_dir = os.path.join(maindir, archname, '%s_%s_%s'%(archname, method, labeltype))
        s_norm = dl_utils.get_overall_sn(save_dir, arch, num_channels=3, load_epoch=200)
        tr_acc = dl_utils.build_graph_and_predict(Xtr, save_dir, arch, Y=Ytr, beta=beta,
                                                  num_channels=3, load_epoch=200)
        print('Acc on training examples: %.4f'%(tr_acc))
        acc, adv_accs = get_adv_acc_curve(Xtt, Ytt, save_dir, arch, eps_list, order=2,
                                          method=ad.pgm, beta=beta, load_epoch=200,
                                          num_channels=3)
        adv_results[method] = (acc, adv_accs, s_norm, tr_acc)
        
    return adv_results


def plot_perturbation_curves(eps_list, adv_results, mode=1, betas_of_interest=None):
    """For plotting curves from perturbation experiment"""
    
    for labeltype in ['Rand', 'True']:

        colors = cm.rainbow(np.linspace(0, 1, len(adv_results)))
        plt.figure(figsize=(10, 7))

        for i, k in enumerate(sorted(adv_results)):
            if 'test' in k and labeltype in k:
                if 'beta' not in k:
                    c = 'k'
                    beta = 0
                else:
                    c = colors[i]
                    beta = float(k.split()[3])
                    
                if betas_of_interest is not None and beta not in betas_of_interest:
                    continue                
                
                if mode == 1:
                    plt.plot(eps_list, (1.-adv_results[k])-(1.-adv_results[k.replace('test', 'train')]),
                             c=c, label=k.replace(' (test_refit)', ''))
                    
                else:
                    plt.plot(eps_list, 1.-adv_results[k], '--', c=c, label=k)
                    plt.plot(eps_list, 1.-adv_results[k.replace('test', 'train')], c=c,
                             label=k.replace('test', 'train'))

        plt.xlabel(r'$\epsilon/C_2$')
        if mode == 1:
            plt.ylabel('Prop labels changed (train) - Prop labels changed (test) ')
        else:
            plt.ylabel('Proportion of labels changed')
        plt.title('Perturbation curves with PGM attacks (%s labels)'%(labeltype))
        plt.legend(bbox_to_anchor=(1.4, 1))
        plt.grid()
        plt.show()
        
        
def get_final_sn(adv, arch_sn, arch, load_epoch=25, snorms_file=None, maindir='save_weights/mnist/'):
    """For each network trained with some given adversarial training routine,
       obtain the spectral norms of the weights.
    """
    
    if snorms_file is not None and os.path.exists(snorms_file):
        results = pickle.load(file(snorms_file, 'rb'))
        
    else:
        results = {}
        
        for f in sorted(os.listdir(maindir)):
            if adv in f and 'backup' not in f and 'rand' not in f and 'pickle' not in f:
                save_dir = os.path.join(maindir, f)

                if 'beta' in f:
                    arch_ = arch_sn
                    beta = float(f.split('beta')[1].split('_')[0])
                else:
                    arch_ = arch
                    beta = 0
                    
                # Get spectral norms of learned weights
                results[f] = dl_utils.get_overall_sn(save_dir, arch_, num_channels=1,
                                                     load_epoch=load_epoch, beta=beta,
                                                     return_snorms=True)

        pickle.dump(results, file(snorms_file, 'wb'))
        
    return results


import pandas as pd
from IPython.display import display

def snorms_to_pd_table(results):
    """Get pandas dataframe from snorms results dict"""

    sn_dict = {
        'layers':[],
    }
    
    order = ['layers']

    for k in sorted(results, key=sort_func):
        
        if 'beta' in k:
            key = u'$\\beta$ = %s'%(float(k.split('beta')[1].split('_')[0]))
        else:
            key = k
            
        order += [key]

        if len(sn_dict['layers']) == 0:
            sn_dict['layers'] = [i.split('/')[0] for i in sorted(results[k].keys())]

        for i, j in sorted(results[k].items()):
            if key not in sn_dict:
                sn_dict[key] = [j]
            else:
                sn_dict[key].append(j)

    return pd.DataFrame.from_dict(sn_dict)[order]


def get_eps_wrm(Xtr, arch_sn, arch, eps_wrm, load_epoch=25, eps_file=None, maindir='save_weights/mnist/'):
    """For WRM trained networks, get eps/C2"""
    
    C2 = np.mean([np.sqrt(np.sum(np.square(i))) for i in Xtr])
    
    if eps_file is not None and os.path.exists(eps_file):
        results = pickle.load(file(eps_file, 'rb'))

    else:

        results = {}

        for f in sorted(os.listdir(maindir)):
            if 'wrm' in f and 'backup' not in f and 'rand' not in f and 'pickle' not in f:
                save_dir = os.path.join(maindir, f)

                if 'beta' in f:
                    arch_ = arch_sn
                    beta = float(f.split('beta')[1].split('_')[0])
                else:
                    arch_ = arch
                    beta = 0
                    
                # Generate adv samples, get eps/C2
                Xtr_adv_wrm = ad.build_graph_and_gen_adv_examples(Xtr, arch_, save_dir, beta=beta,
                                                                  num_channels=1, load_epoch=load_epoch,
                                                                  method=ad.wrm, eps=eps_wrm)
                eps = np.sqrt(np.mean([np.sum(np.square(i)) for i in Xtr_adv_wrm-Xtr]))
                print('For beta = %s, eps/C2 = %.10f'%(beta, eps/C2))
                results[f] = eps
                    
        pickle.dump(results, file(eps_file, 'wb'))
        
    return results


def get_train_v_test_adv_acc_curves(Xtr, Ytr, Xtt, Ytt, arch_sn, arch, adv, eps_list,
                                    load_epoch=25, curves_file=None, betas_of_interest=None, 
                                    maindir='save_weights/mnist/'):
    """Scans through the trained networks in a directory and for each network, gets
       adv attack curves on the training and test sets (uses true labels)
    """
    
    if curves_file is not None and os.path.exists(curves_file):
        adv_results = pickle.load(file(curves_file, 'rb'))

    else:
        adv_results = {}

        for f in sorted(os.listdir(maindir)):
            if adv in f and 'backup' not in f and 'pickle' not in f:
                save_dir = os.path.join(maindir, f)

                if 'beta' in f:
                    arch_ = arch_sn
                    beta = float(f.split('beta')[1].split('_')[0])
                    key_base = r'%s $\beta$ = %s'%(adv.upper(), beta)
                else:
                    arch_ = arch
                    beta = 0
                    key_base = '%s'%(adv.upper())

                if betas_of_interest is not None and beta not in betas_of_interest:
                    continue

                if 'rand' in f:
                    key_base += ' Rand'
                else:
                    key_base += ' True'

                cargs = {'beta': beta, 'load_epoch':200, 'num_channels':1}

                _, adv_results['%s (train)'%(key_base)] = \
                    get_adv_acc_curve(Xtr, Ytr, save_dir, arch_, eps_list, method=ad.pgm, **cargs)

                if 'rand' not in f:
                    _, adv_results['%s (test)'%(key_base)] = \
                        get_adv_acc_curve(Xtt, Ytt, save_dir, arch_, eps_list, method=ad.pgm, **cargs)

                else:
                    adv_results['%s (test)'%(key_base)] = np.ones(len(eps_list))*0.1

        pickle.dump(adv_results, file(curves_file, 'wb'))
        
    return adv_results


def get_perturbation_curves(Xtr, Xtt, arch_sn, arch, adv, eps_list,
                            load_epoch=25, curves_file=None, betas_of_interest=None, 
                            maindir='save_weights/mnist/', gpu_prop=0.1):
    """Scans through the trained networks in a directory and for each value of beta,
       recovers a perturbation curve, or the proportion of labels that change after
       adv attacks of various magnitudes.
    """

    if curves_file is not None and os.path.exists(curves_file):
        adv_results = pickle.load(file(curves_file, 'rb'))

    else:
        adv_results = {}

    for f in sorted(os.listdir(maindir)):
        if adv in f and 'backup' not in f and 'pickle' not in f:
            save_dir = os.path.join(maindir, f)

            if 'beta' in f:
                arch_ = arch_sn
                beta = float(f.split('beta')[1].split('_')[0])
                key_base = r'%s $\beta$ = %s'%(adv.upper(), beta)
            else:
                arch_ = arch
                beta = 0
                key_base = '%s'%(adv.upper())
                    
            if betas_of_interest is not None and beta not in betas_of_interest:
                continue

            if 'rand' in f:
                key_base += ' Rand'
            else:
                key_base += ' True'

            cargs = {'beta':beta, 'load_epoch':load_epoch, 'num_channels':1, 'gpu_prop':gpu_prop,}

            print(f)
                
            if '%s (train_refit)'%(key_base) not in adv_results:
                Ytrhat = dl_utils.build_graph_and_predict(Xtr, save_dir, arch_, **cargs)
                _, adv_results['%s (train_refit)'%(key_base)] = \
                    get_adv_acc_curve(Xtr, Ytrhat, save_dir, arch_, eps_list, method=ad.pgm, **cargs)

                Ytthat = dl_utils.build_graph_and_predict(Xtt, save_dir, arch_, **cargs)
                _, adv_results['%s (test_refit)'%(key_base)] = \
                    get_adv_acc_curve(Xtt, Ytthat, save_dir, arch_, eps_list, method=ad.pgm, **cargs)

    pickle.dump(adv_results, file(curves_file, 'wb'))
        
    return adv_results