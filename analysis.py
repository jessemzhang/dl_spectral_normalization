import os
import pickle
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from IPython.display import display
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

    
def perturb_images(X, eps, arch, adv, method=ad.wrm, n=None,
                   load_epoch=25, beta=1., seed=0, save_dir=None):
    """Perturb and classify X"""
    
    num_channels = np.shape(X)[-1]
    if save_dir is None:
        save_dir = os.path.join('save_weights/mnist/', adv)
    
    np.random.seed(seed)
    if n is not None:
        inds = np.random.choice(range(len(X)), n, replace=False)
        X_ = X[inds]
    else:
        inds = np.arange(len(X))
        X_ = X
        
    X_adv = ad.build_graph_and_gen_adv_examples(X_, arch, save_dir, num_channels=num_channels,
                                                beta=beta, method=method, eps=eps, load_epoch=load_epoch)
    Y_adv = dl_utils.build_graph_and_predict(X_adv, save_dir, arch, beta=beta, num_channels=num_channels,
                                             load_epoch=load_epoch)
    
    return X_adv, Y_adv, inds


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


def plot_acc_curves(adv_results, x_vals, title='PGM attacks', sort_func=None, logy=True, 
                    logx=False, ylim=(1e-2, 1e0), xlim=(0, 0.2), savename=None, wdmode=False,
                    report_test=True, report_sn=True, diff_methods=False, bigfig=False):
    """ adv_results should be a set of curves generated from get_adv_acc_curve
        (i.e. it's a dictionary where keys describe different networks)
        Plots all of these curves against one another.
    """
    
    n = len(adv_results)
    if sum([1 for i in adv_results if 'beta' not in i and 'wd' not in i]) > 0:
        n -= 1
    
    if diff_methods:
        colors = cm.rainbow(np.linspace(0, 1, len(adv_results))).tolist()
    else:                        
        colors = cm.rainbow(np.linspace(0, 1, n)).tolist()

    if bigfig: 
        plt.figure(figsize=(10, 7))
    else:
        plt.figure(figsize=(5, 4))
    if sort_func is not None:
        keys_in_ord = sorted(adv_results, key=sort_func)
    else:
        keys_in_ord = sorted(adv_results)
    for i, k in enumerate(keys_in_ord):
        if not diff_methods:
            
            if not wdmode:
                if 'beta' in k:
                    key = u'$\\beta$ = %s'%(float(k.split('beta')[1].split('_')[0]))
                    c = colors.pop(0)
                else:
                    key = u'$\\beta = \\infty$'
                    c = 'k'

            else:
                if 'wd' in k:
                    key = u'$\\lambda$/2 = %s'%(float(k.split('wd')[1].split('_')[0]))
                    c = colors.pop(0)
                else:
                    key = u'$\\lambda$/2 = 0'
                    c = 'k'
                
        else:
            c = colors.pop(0)
            key = k.upper()
            
        if len(adv_results[k]) == len(x_vals):
            plt.plot(x_vals, 1.-adv_results[k], c=c, label=key)
        elif len(adv_results[k]) > 2:
            label = key
            if report_test:
                label += ', test acc %.3f'%(adv_results[k][0])
            if report_sn:
                label += ', sn %.3e'%(adv_results[k][2])
                
            plt.plot(x_vals, 1.-adv_results[k][1], c=c, label=label)
        else:
            if report_test:
                label = '%s (test acc %.3f)'%(key, adv_results[k][0])
            else:
                label = '%s'%(key)
            plt.plot(x_vals, 1.-adv_results[k][1], c=c, label=label)
            
    plt.xlabel(r'$\epsilon/C_2$')
    if xlim is not None:
        plt.xlim(xlim)
    if logx:
        plt.xscale('log')
    plt.ylabel('Error')
    if ylim is not None:
        plt.ylim(ylim)
    if logy:
        plt.yscale('log')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.4, 1))
    plt.grid(which="both")
    if savename is not None:
        plt.savefig(savename.split('.')[0]+'.pdf', format='pdf', dpi=500, bbox_inches='tight')
    plt.show()
    
    
def param_sweep_curves(X, Y, adv, eps_list, arch_sn, arch, load_epoch=25,
                       basedir='save_weights/mnist/', wdmode=False):
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
                
            if wdmode and 'wd' not in f:
                continue
                
            acc, adv_accs = get_adv_acc_curve(X, Y, save_dir, arch_, eps_list, order=2,
                                              method=ad.pgm, beta=beta, load_epoch=load_epoch)
            adv_results[f] = (acc, adv_accs)
            
    return adv_results


def sort_func(f):
    """used for sorting the trained networks in terms of beta"""
    if 'beta' not in f and 'wd' not in f:
        return 0
    
    if 'beta' in f:
        return float(f.split('beta')[1].split('_')[0])
    
    return float(f.split('wd')[1].split('_')[0])


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
        
        
def get_final_sn(adv, arch_sn, arch, load_epoch=25, snorms_file=None, maindir='save_weights/mnist/', wdmode=False):
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
                
                if wdmode and 'wd' not in f:
                    continue

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


def get_output_norms(X, adv, arch_sn, arch, load_epoch=25, fcnorms_file=None,
                     maindir='save_weights/mnist/', wdmode=False):
    """For each network trained with some given adversarial training routine,
       obtain the L2 norm of the output (pre-softmax) layer.
    """
    
    if fcnorms_file is not None and os.path.exists(fcnorms_file):
        results = pickle.load(file(fcnorms_file, 'rb'))
        
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
                    
                if wdmode and 'wd' not in f:
                    continue
                    
                embed = dl_utils.get_embedding(X, save_dir, arch_, beta=beta, load_epoch=load_epoch, num_channels=1)
                results[f] = np.linalg.norm(embed, axis=1)

        pickle.dump(results, file(fcnorms_file, 'wb'))
        
    return results


def snorms_to_pd_table(results, wdmode=False):
    """Get pandas dataframe from snorms results dict"""

    sn_dict = {
        'layers':[],
    }
    
    order = ['layers']

    for k in sorted(results, key=sort_func):
        
        if not wdmode:
            if 'beta' in k:
                key = u'$\\beta$ = %s'%(float(k.split('beta')[1].split('_')[0]))
            else:
                key = u'$\\beta = \\infty$'
                
        else:
            if 'wd' in k:
                key = u'$\\lambda$/2 = %s'%(float(k.split('wd')[1].split('_')[0]))
            else:
                key = u'$\\lambda$/2 = 0'
            
        order += [key]

        if len(sn_dict['layers']) == 0:
            sn_dict['layers'] = [i.split('/')[0] for i in sorted(results[k].keys())]

        for i, j in sorted(results[k].items()):
            if key not in sn_dict:
                sn_dict[key] = [j]
            else:
                sn_dict[key].append(j)

    return pd.DataFrame.from_dict(sn_dict)[order]


def violin_plots_of_norm_output_to_input_ratios(results, norm_input, title=None, savename=None, wdmode=False):
    """Compare norm ratio of pre-softmax layer norms to input norms"""
    
    results_ = dict(results)

    for k in results_.keys():
            
        if not wdmode:
            if 'beta' in k:
                key = u'$\\beta$ = %s'%(float(k.split('beta')[1].split('_')[0]))
            else:
                key = u'$\\beta = \\infty$'
                
        else:
            if 'wd' in k:
                key = u'$\\lambda$/2 = %s'%(float(k.split('wd')[1].split('_')[0]))
            else:
                key = u'$\\lambda$/2 = 0'
                
        results_[key] = results_.pop(k)/norm_input

    df = pd.DataFrame.from_dict(results_)
    df = pd.melt(df, value_vars=df.columns)
    plt.figure(figsize=(4, 3))
    sb.violinplot(x='variable', y='value', cut=0, data=df, scale='width')
    plt.grid()
    plt.ylabel(r'$\Vert\phi(x)\Vert_2/\Vert x\Vert_2$')
    plt.xlabel('')
    if title is not None:
        plt.title(title)
    if savename is not None:
        plt.savefig(savename.split('.')[0]+'.pdf', format='pdf', dpi=500, bbox_inches='tight')
    plt.show()


def get_eps_wrm(Xtr, arch_sn, arch, eps_wrm, load_epoch=25, eps_file=None,
                num_channels=1, maindir='save_weights/mnist/', wdmode=False):
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
                    
                if wdmode and 'wd' not in f:
                    continue
                    
                # Generate adv samples, get eps/C2
                Xtr_adv_wrm = ad.build_graph_and_gen_adv_examples(Xtr, arch_, save_dir, beta=beta,
                                                                  num_channels=num_channels,
                                                                  load_epoch=load_epoch,
                                                                  method=ad.wrm, eps=eps_wrm)
                eps = np.sqrt(np.mean([np.sum(np.square(i)) for i in Xtr_adv_wrm-Xtr]))
                print('For beta = %s, eps/C2 = %.10f'%(beta, eps/C2))
                results[f] = eps
                    
        pickle.dump(results, file(eps_file, 'wb'))
        
    return results


def plot_wrm_robustness(results, C2=None, savename=None, ylabel=None, title=None, wdmode=False, logx=True):
    """Plot the epsilon parameter achieved by WRM"""
    
    param = 'wd' if wdmode else 'beta'
    
    plt.figure(figsize=(5, 4))
    
    param_list = [float(i.split(param)[1]) for i in sorted(results) if param in i]
    if C2 is not None:
        robustness = [results[i]/C2 for i in sorted(results) if param in i]
    else:
        robustness = [results[i] for i in sorted(results) if param in i]
    plt.plot(param_list, robustness)
    
    if sum([1 for i in results if param not in i]) > 0:
        if wdmode:
            label = u'$\\lambda/2$ = 0'
        else:
            label = u'$\\beta = \\infty$'
        if C2 is not None:
            plt.plot([np.min(param_list), np.max(param_list)],
                     [results['wrm']/C2, results['wrm']/C2], '--', label=label)
        else:
            plt.plot([np.min(param_list), np.max(param_list)],
                     [results['wrm'], results['wrm']], '--', label=label)
        plt.legend()
        
    if wdmode:
        plt.xlabel(u'$\\lambda$/2')
    else:
        plt.xlabel(u'$\\beta$')
    if logx:
        plt.xscale('log')
    if ylabel is None:
        if C2 is not None:
            plt.ylabel(u'$\\epsilon/C_2$')
        else:
            plt.ylabel(u'$\\epsilon$')
    else:
        plt.ylabel(ylabel)
    if title is None:
        plt.title('WRM adversarial robustness')
    else:
        plt.title(title)
    if wdmode:
        plt.grid()
    else:
        plt.grid(which="both")
    if savename is not None:
        plt.savefig(savename.split('.')[0]+'.pdf', format='pdf', dpi=500, bbox_inches='tight')
    plt.show()


def get_train_test_accs(Xtr, Ytr, Xtt, Ytt, arch_sn, arch, adv=None, num_channels=1, 
                        load_epoch=25, gpu_prop=0.1, train_test_accs_file=None,
                        maindir='save_weights/mnist/', verbose=True, randmode=False, wdmode=False):
    """Scans through the trained networks in a directory and for each network, gets
       train and test accuracies
    """

    if train_test_accs_file is not None and os.path.exists(train_test_accs_file):
        results = pickle.load(file(train_test_accs_file, 'rb'))

    else:
        results = {}

    for f in sorted(os.listdir(maindir)):
        
        if 'wrm' in f and 'randlabels' in f:
            if len(f.split('labels')[-1]) > 0:
                continue
        
        if f in results:
            continue
        if randmode and 'rand' not in f:
            continue
        elif not randmode and 'rand' in f:
            continue

        if 'backup' not in f and 'pickle' not in f:
            if adv is not None and adv not in f:
                continue

            save_dir = os.path.join(maindir, f)

            if wdmode and 'wd' not in f:
                continue
                
            if not wdmode and 'wd' in f:
                continue

            if 'beta' in f:
                arch_ = arch_sn
                beta = float(f.split('beta')[1].split('_')[0])
            else:
                arch_ = arch
                beta = 0

            # Get spectral norms of learned weights
            tr_acc = dl_utils.build_graph_and_predict(Xtr, save_dir, arch_, Y=Ytr, 
                                                      num_channels=num_channels,
                                                      load_epoch=load_epoch, 
                                                      beta=beta, gpu_prop=gpu_prop)

            tt_acc = dl_utils.build_graph_and_predict(Xtt, save_dir, arch_, Y=Ytt, 
                                                      num_channels=num_channels,
                                                      load_epoch=load_epoch, 
                                                      beta=beta, gpu_prop=gpu_prop)

            if verbose:
                print('%s done with train acc %.4f, test acc %.4f'%(f, tr_acc, tt_acc))
            results[f] = (tr_acc, tt_acc)

    pickle.dump(results, file(train_test_accs_file, 'wb'))

    return results


def train_test_accs_to_pd_table(results):
    """Get pandas dataframe from train/test acc results dict"""

    for k in results.keys():
        if 'beta' in k:
            key = u'$\\beta$ = %s'%(float(k.split('beta')[1].split('_')[0]))
        else:
            key = u'$\\beta = \\infty$'
        results[key] = results.pop(k)

    df = pd.DataFrame.from_dict(results).T
    df.columns = ['Train Accuracy', 'Test Accuracy']
    return df


def plot_train_test_accs(results, savename=None, title=None, logx=True, logy=True, ylim=None, wdmode=False):
    """Plots train and test accs from train/test acc results dict"""
    
    plt.figure(figsize=(5, 4))
    
    if wdmode:
        param = 'wd'
    else:
        param = 'beta'

    param_list = [float(i.split(param)[-1].split('_')[0]) for i in sorted(results) if param in i]
    
    train_accs = [results[i][0] for i in sorted(results) if param in i]
    test_accs = [results[i][1] for i in sorted(results) if param in i]
    plt.plot(param_list, train_accs, 'b', label='Train')
    plt.plot(param_list, test_accs, 'r', label='Test')

    if sum([1 for i in results if param in i]) > 0:
        nosn_tr_acc = [results[i][0] for i in results if param not in i][0]
        nosn_tt_acc = [results[i][1] for i in results if param not in i][0]
        
        if wdmode:
            label_tr = u'Train ($\\lambda/2$ = 0)'
            label_tt = u'Test ($\\lambda/2$ = 0)'
        else:
            label_tr = u'Train ($\\beta = \\infty$)'
            label_tt = u'Test ($\\beta = \\infty$)'
        
        plt.plot([np.min(param_list), np.max(param_list)],
                 [nosn_tr_acc, nosn_tr_acc], 'b--', label=label_tr)
        plt.plot([np.min(param_list), np.max(param_list)],
                 [nosn_tt_acc, nosn_tt_acc], 'r--', label=label_tt)

    if wdmode:
        plt.xlabel(u'$\\lambda$/2')
    else:
        plt.xlabel(u'$\\beta$')
    if logx:
        plt.xscale('log')
    plt.ylabel('Accuracy')
    if ylim is not None:
        plt.ylim(ylim)
    if logy:
        plt.yscale('log')
    if title is None:
        plt.title('Train and test accuracies with SN')
    else:
        plt.title(title)
    if param == 'wd':
        plt.grid()
    else:
        plt.grid(which="both")
    plt.legend()
    if savename is not None:
        plt.savefig(savename.split('.')[0]+'.pdf', format='pdf', dpi=500, bbox_inches='tight')
    plt.show()


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

                cargs = {'beta': beta, 'load_epoch':load_epoch, 'num_channels':1}

                _, adv_results['%s (train)'%(key_base)] = \
                    get_adv_acc_curve(Xtr, Ytr, save_dir, arch_, eps_list, method=ad.pgm, **cargs)

                if 'rand' not in f:
                    _, adv_results['%s (test)'%(key_base)] = \
                        get_adv_acc_curve(Xtt, Ytt, save_dir, arch_, eps_list, method=ad.pgm, **cargs)

                else:
                    adv_results['%s (test)'%(key_base)] = np.ones(len(eps_list))*0.1

        pickle.dump(adv_results, file(curves_file, 'wb'))
        
    return adv_results


def get_perturbation_curves(Xtr, Xtt, arch_sn, arch, adv, eps_list, num_channels=1,
                            load_epoch=25, curves_file=None, betas_of_interest=None, 
                            maindir='save_weights/mnist/', gpu_prop=0.1, verbose=True,
                            skiprand=False, skiptrue=False, eps_list_rand=None, wdmode=False):
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
                
            if wdmode:
                if'wd' not in f:
                    continue
                else:
                    arch_ = arch
                    wd = float(f.split('wd')[1].split('_')[0])
                    key_base = r'%s $\\lambda$/2 = %s'%(adv.upper(), wd)

            if 'rand' in f:
                if skiprand:
                    continue
                if eps_list_rand is not None:
                    eps_list_ = eps_list_rand
                else:
                    eps_list_ = eps_list
                key_base += ' Rand'
            else:
                if skiptrue:
                    continue
                eps_list_ = eps_list
                key_base += ' True'

            cargs = {'beta':beta, 'load_epoch':load_epoch, 'num_channels':num_channels, 'gpu_prop':gpu_prop}

            if verbose: print(f, beta, load_epoch, num_channels, gpu_prop, arch_, np.sum(np.abs(Xtr)))
                
            if '%s (train_refit)'%(key_base) not in adv_results:
                Ytrhat = dl_utils.build_graph_and_predict(Xtr, save_dir, arch_, **cargs)
                _, adv_results['%s (train_refit)'%(key_base)] = \
                    get_adv_acc_curve(Xtr, Ytrhat, save_dir, arch_, eps_list_, method=ad.pgm, **cargs)

                Ytthat = dl_utils.build_graph_and_predict(Xtt, save_dir, arch_, **cargs)
                _, adv_results['%s (test_refit)'%(key_base)] = \
                    get_adv_acc_curve(Xtt, Ytthat, save_dir, arch_, eps_list_, method=ad.pgm, **cargs)

#    pickle.dump(adv_results, file(curves_file, 'wb'))
        
    return adv_results


def plot_perturbation_curves(eps_list, adv_results, mode=1, betas_of_interest=None, title=None,
                             savename=None, ylim=None, logy=False, eps_list_rand=None, wdmode=False):
    """For plotting curves from perturbation experiment"""
    
    if wdmode:
        param = 'lambda'
    else:
        param = 'beta'
         
    
    for labeltype in ['Rand', 'True']:
        
        if sum([1 for i in adv_results if labeltype in i]) == 0:
            continue
        
        n = sum([1 for i in adv_results if param in i and labeltype in i])/2
        colors = cm.rainbow(np.linspace(0, 1, n)).tolist()
        plt.figure(figsize=(5, 4))

        for i, k in enumerate(sorted(adv_results)):
            if 'test' in k and labeltype in k:
                if 'beta' not in k and 'lambda' not in k:
                    c = 'k'
                    beta = 0
                    label = u'$\\beta = \\infty$'
                else:
                    c = colors.pop(0)
                    
                    if 'lambda' in k:
                        wd = float(k.split()[3])
                        label = u'$\\lambda/2 = %s$'%(wd)
                    else:
                        beta = float(k.split()[3])
                        label = u'$\\beta = %s$'%(beta)
                    
                if betas_of_interest is not None and beta not in betas_of_interest:
                    continue 
                    
                if 'Rand' in k and eps_list_rand is not None:
                    eps_list_ = eps_list_rand
                else:
                    eps_list_ = eps_list
                
                if mode == 1:
                    plt.plot(eps_list_, (1.-adv_results[k])-(1.-adv_results[k.replace('test', 'train')]),
                             c=c, label=label)
                    
                else:
                    plt.plot(eps_list_, 1.-adv_results[k], '--', c=c, label=k)
                    plt.plot(eps_list_, 1.-adv_results[k.replace('test', 'train')], c=c,
                             label=k.replace('test', 'train'))

        plt.xlabel(r'$\epsilon/C_2$')
        if ylim is not None:
            plt.ylim(ylim)
        if logy:
            plt.yscale('log')
        if mode == 1:
            plt.ylabel(u'Test $P_{\mathregular{adv}}$-Train $P_{\mathregular{adv}}$')
        else:
            plt.ylabel('Proportion of labels changed')
        if title is None:
            plt.title('Networks Fit on %s Labels'%(labeltype.replace('Rand', 'Random')))
        else:
            if 'Rand' == labeltype:
                plt.title(title + ' (random labels)')
            else:
                plt.title(title)
        plt.legend(bbox_to_anchor=(1.4, 1))
        plt.grid()
        if savename is not None:
            plt.savefig(savename.split('.')[0]+'_%s.pdf'%(labeltype), format='pdf', dpi=500, bbox_inches='tight')
        plt.show()

        
def per_image_whitening(images):
    "Mimic tensorflow per_image_whitening"
    orig_shape = images.shape
    images = images.reshape((images.shape[0], -1))
    img_means = np.mean(images, axis=1, keepdims=True)
    img_stds = np.std(images, axis=1, keepdims=True)
    adj_stds = np.maximum(img_stds, 1.0 / np.sqrt(images.shape[1]))
    whiten_imgs = (images - img_means) / adj_stds
    return whiten_imgs.reshape(orig_shape), img_means, adj_stds


def inv_per_image_whitening(images, img_means, adj_stds):
    "Invert whitening operation"
    orig_shape = images.shape
    images = images.reshape((images.shape[0], -1))
    orig_imgs = images * adj_stds + img_means
    return orig_imgs.reshape(orig_shape)

        
def perturb_images_wrm_v_wrmsn(X_, Y_, eps, beta, arch, arch_sn, save_dir, save_dir_sn,
                               preprocess=True, load_epoch=200):
    """Compare perturbations for wrm v wrmsn"""
    
    n = len(X_)
    
    if preprocess:    
        # Whiten images
        W_, img_means, adj_stds = per_image_whitening(X_)

        # Crop
        W_ = np.array([i[2:30, 2:30, :] for i in W_])
        
    else:
        W_ = X_

    # Perturb using WRM
    W_adv, Y_adv, inds = perturb_images(W_, eps, arch, 'wrm', n=None, load_epoch=load_epoch,
                                        method=ad.pgm, save_dir=save_dir)
    print('Acc on adv examples: %.4f'%(np.sum(Y_adv == Y_)/float(n)))
    
    # Perturb using WRM SN
    W_adv_sn, Y_adv_sn, inds_sn = perturb_images(W_, eps, arch_sn, 'wrm', n=None, load_epoch=load_epoch,
                                                 method=ad.pgm, beta=beta, save_dir=save_dir_sn)
    print('Acc on adv examples: %.4f'%(np.sum(Y_adv_sn == Y_)/float(n)))
    
    if preprocess:
        # Undo cropping
        def pad(x):
            return np.pad(x, ((2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)
        W_adv = np.array([pad(i) for i in W_adv])
        W_adv_sn = np.array([pad(i) for i in W_adv_sn])

        # Undo whitening
        X_adv = inv_per_image_whitening(W_adv, img_means, adj_stds)
        X_adv_sn = inv_per_image_whitening(W_adv_sn, img_means, adj_stds)

        # Redo cropping
        X_adv = np.array([i[2:30, 2:30, :] for i in X_adv])
        X_adv_sn = np.array([i[2:30, 2:30, :] for i in X_adv_sn])
        
    else:
        X_adv = W_adv
        X_adv_sn = W_adv_sn

    print('Num test samples where sn with beta = %s does better: %s'\
          %(beta, np.sum((Y_ != Y_adv) & (Y_ == Y_adv_sn))))
    print('Num test samples where sn with beta = infinity does better: %s'\
          %(np.sum((Y_ != Y_adv_sn) & (Y_ == Y_adv))))
    
    return X_adv, Y_adv, X_adv_sn, Y_adv_sn


def show_perturbed_images(X_, X_adv, X_adv_sn, Y_, Y_adv, Y_adv_sn, cifar10_label_dict, eps, beta,
                          n=10, seed=0, inds_of_interest=None, savename=None):
    """Look at images where the beta approach does better"""
    
    if len(X_.shape) < 4 or X_.shape[-1] == 1:
        graycmap = True
    else: 
        graycmap = False
    
    plt.figure(figsize=(5, n*2))

    p_count = 0
    np.random.seed(seed)

    if inds_of_interest is None:
        inds_of_interest = np.arange(len(Y_))

    for i in np.random.choice(inds_of_interest, n, replace=False):
        plt.subplot(n, 3, p_count+1)
        if graycmap:
            plt.imshow(X_[i], cmap='gray')
        else:
            plt.imshow(X_[i])
        plt.axis('off')
        plt.title('Original\n$y$ = %s'%(cifar10_label_dict[Y_[i]]))
        plt.subplot(n, 3, p_count+2)
        if graycmap:
            plt.imshow(X_adv[i], cmap='gray')
        else:
            plt.imshow(X_adv[i])
        plt.axis('off')
        plt.title(u'WRM\n$\^y$ = %s' %(cifar10_label_dict[int(Y_adv[i])]))
        plt.subplot(n, 3, p_count+3)
        if graycmap:
            plt.imshow(X_adv_sn[i], cmap='gray')
        else:
            plt.imshow(X_adv_sn[i])
        plt.axis('off')
        plt.title(u'SAR\n$\^y$ = %s'%(cifar10_label_dict[int(Y_adv_sn[i])]))
        p_count += 3

    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename.split('.')[0]+'_true.pdf', format='pdf', dpi=500, bbox_inches='tight')
    plt.show()