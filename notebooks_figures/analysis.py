import os
import itertools
import pickle
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import sys
sys.path.insert(0, '../') 
from dl_spectral_normalization import dl_utils


def plot_adv_attack_curves(data, eps_list, eps_train=None, lw=3,
                           xlim=None, ylim=None, title=None, legend=True):
    """Generate plot of error after sweeping through various magnitude attacks"""
    markers = ['s', 'o', 'v', 'D', '+', '.', '^', '*']
    for i, k in enumerate(data):
        plt.plot(eps_list, 1-data[k], label=k, lw=lw, marker=markers[i], markeredgecolor='k', ms=7)
    if eps_train is not None:
        plt.axvline(x=eps_train, color='k', linestyle='--')
    if legend: plt.legend()
    plt.xlabel(r'$\epsilon$')
    plt.ylabel('error')
    if ylim is not None: plt.ylim(ylim)
    if xlim is not None: plt.xlim(xlim)
    if title is not None: plt.title(title)
        
        
def get_l2_of_grads(X, Y, beta, save_dir, defense, arch, gpu_prop=0.2, load_epoch=None, batch_size=100):
    """Get the L2 norm of gradients through the network"""
    
    tf.reset_default_graph()
    g = dl_utils.graph_builder_wrapper(arch, adv='erm', save_dir=save_dir, update_collection='_', beta=beta)
    grads = tf.gradients(g['total_loss'], g['input_data'])
    if load_epoch is None:
        load_epoch = dl_utils.latest_epoch(save_dir)
    else:
        load_epoch = np.min((dl_utils.latest_epoch(save_dir), load_epoch))

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_prop))) as sess:
        g['saver'].restore(sess, os.path.join(save_dir, 'checkpoints', 'epoch%s'%(load_epoch)))
        
        grads_eval = np.zeros(np.shape(X))
        for i in range(0, len(X), batch_size):
            feed_dict = {g['input_data']: X[i:i+batch_size], g['input_labels']: Y[i:i+batch_size]}
            grads_eval[i:i+batch_size] = sess.run(grads, feed_dict=feed_dict)[0] * batch_size
            
    l2_norms = np.zeros(len(grads_eval))
    for i, grad in enumerate(grads_eval):
        l2_norms[i] = np.sqrt(np.sum(np.square((grad))))

    return l2_norms


def make_kappa_plots(X, Y, beta_list, dirname, defense, resultsfile, arch, plot_stuff=True):
    """For all betas in a directory, get the l2_norms of the gradients"""
    if os.path.isfile(resultsfile):
        results = pickle.load(file(resultsfile, 'rb'))
    else:
        results = {}
    if plot_stuff:
        plt.figure(figsize=(16, 6))
    for i, beta in enumerate(beta_list):
        if beta not in results:
            save_dir = os.path.join(dirname, '%s_beta%s'%(defense, beta))
            l2_norms = get_l2_of_grads(X, Y, beta, save_dir, defense, arch, load_epoch=None)
            results[beta] = l2_norms
            pickle.dump(results, file(resultsfile, 'wb'))
        else:
            l2_norms = results[beta]
        if plot_stuff:
            plt.subplot(2, 3, i+1)
            plt.hist(l2_norms, density=True, bins=100)
            plt.xlabel(r'$\kappa$')
            plt.title(r'%s with $\beta = %s$'%(defense.upper(), beta))
    if plot_stuff:
        plt.tight_layout()
        plt.show()
    return results


def plot_hists(ratios, title=None, value_name=None, legend=True):
    """Plots multiple histograms, one for each key: value pair in ratios 
       (key = legend label, value = array of values to make hist of)
    """
    df = pd.DataFrame.from_dict(ratios)
    for beta in df.columns:
        sns.distplot(df[beta], rug=False, label=r'$\beta$=%s'%(beta))
    plt.xlabel('data' if value_name is None else value_name)
    if title is not None:
        plt.title(title)
    if legend: plt.legend()        
    

def plot_training_curves(beta, defense, dirname, compare_inf=True, ylim=None, lw=2,
                         num_batches_per_epoch=1, legend=True, ylabel=True):
    """Plot training and validation curves for paper"""
    markers = ['s', 'D', 'v', 'o', '+', '.', '^', '*']
    
    def smoothen(x, c=10):
        x_new = np.array(x)
        for i in range(len(x)):
            if i >= c-1:
                x_new[i] = np.mean(x[np.max((c-1, i-c+1)):i+1])
        return x_new
    
    if compare_inf:
        save_dir = os.path.join(dirname, '%s_betainf'%(defense))
        curves1 = dl_utils.extract_train_valid_tensorboard(save_dir, curve='accuracy',
                                                           show_plot=False, only_final_value=False)
        tr_acc1, tt_acc1 = smoothen(curves1[0]), smoothen(curves1[1])
        xaxis = np.arange(len(tr_acc1))*num_batches_per_epoch
        plt.plot(xaxis, tr_acc1, lw=lw, c='navy')
        plt.plot(xaxis, tt_acc1, lw=lw, c='royalblue')

    save_dir = os.path.join(dirname, '%s_beta%s'%(defense, beta))
    curves2 = dl_utils.extract_train_valid_tensorboard(save_dir, curve='accuracy',
                                                       show_plot=False, only_final_value=False)
    tr_acc2, tt_acc2 = smoothen(curves2[0]), smoothen(curves2[1])
    xaxis = np.arange(len(tr_acc2))*num_batches_per_epoch
    plt.plot(xaxis, tr_acc2, lw=lw, c='orangered')
    plt.plot(xaxis, tt_acc2, lw=lw, c='orange')
    if len(tr_acc1) < len(tr_acc2):
        plt.plot([len(tr_acc1)*num_batches_per_epoch, len(tr_acc2)*num_batches_per_epoch],
                 [tr_acc1[-1], tr_acc1[-1]], '--', lw=lw, c='navy')
        plt.plot([len(tt_acc1)*num_batches_per_epoch, len(tr_acc2)*num_batches_per_epoch],
                 [tt_acc1[-1], tt_acc1[-1]], '--', lw=lw, c='royalblue')
    elif len(tr_acc2) < len(tr_acc1):
        plt.plot([len(tr_acc2)*num_batches_per_epoch, len(tr_acc1)*num_batches_per_epoch],
                 [tr_acc2[-1], tr_acc2[-1]], '--', lw=lw, c='orangered')
        plt.plot([len(curves2[1])*num_batches_per_epoch, len(curves1[1])*num_batches_per_epoch],
                 [tt_acc2[-1], tt_acc2[-1]], '--', lw=lw, c='orange')
        
    xmax = int(np.max([[len(tr_acc1), len(tr_acc2)]]))
    xm = xmax/2 - int(0.075*xmax)
    plt.plot(xm*num_batches_per_epoch, tr_acc1[xm] if xm < len(tr_acc1) else tr_acc1[-1],
             marker=markers[0], label=r'train', lw=lw, ms=10, markeredgecolor='k', c='navy')
    xm = xmax/2 - int(0.0375*xmax)
    plt.plot(xm*num_batches_per_epoch, tt_acc1[xm] if xm < len(tt_acc1) else tt_acc1[-1], 
             marker=markers[1], label=r'valid', lw=lw, ms=10, markeredgecolor='k', c='royalblue' )
    xm = xmax/2 + int(0.0375*xmax)
    plt.plot(xm*num_batches_per_epoch, tr_acc2[xm] if xm < len(tr_acc2) else tr_acc2[-1], 
             marker=markers[2], label=r'train (SN)', lw=lw, ms=10, markeredgecolor='k', c='orangered' )
    xm = xmax/2 + int(0.075*xmax)
    plt.plot(xm*num_batches_per_epoch, tt_acc2[xm] if xm < len(tt_acc2) else tt_acc2[-1],
             marker=markers[3], label=r'valid (SN)', lw=lw, ms=10, markeredgecolor='k', c='orange' )
        
    if num_batches_per_epoch != 1:
        plt.xlabel('training steps')
    else:
        plt.xlabel('epoch')
    if ylabel: plt.ylabel('accuracy')
    if ylim is not None: plt.ylim(ylim)
    plt.title('%s training'%(defense.upper()))
#     plt.grid()
    if legend: plt.legend()
        
        
def get_margins(X, Y, save_dir, beta, arch, adv='erm', eps=0.3):
    """Get margins from trained network
        For a (X, Y) pair, the margin is the difference between 
           1) the fully-connected-layer-entry corresponding to the correct prediction
           2) the highest other fully-connected-layer-entry
    """
    embeddings = dl_utils.get_embedding(X, save_dir, arch, beta=beta, adv=adv, eps=eps)
    print('Sanity check: accuracy is %.5f.'
          %(np.sum(np.argmax(embeddings, 1) == Y)/float(len(Y))))
    margins = np.zeros(len(embeddings))
    for i, e in enumerate(embeddings):
        margins[i] = e[Y[i]]-np.max(np.delete(e, Y[i]))
    return margins


def compute_margins(X, Y, dirname, beta_list, defense, eps=0.3, mode=1, extra_correction=False):
    """For AlexNet, get the margins and normalize using the bounds discussed in the paper"""
    key_order = [
        'conv1/weights:0',
        'conv2/weights:0',
        'linear1/weights:0',
        'linear2/weights:0',
        'fc/weights:0'
    ]
    conv_input_shapes = {
        'conv1/weights:0': (28, 28, 3),
        'conv2/weights:0': (13, 13, 96)
    }
    
    all_margins = []
    all_gammas = []
    for beta in beta_list:
        save_dir = os.path.join(dirname, '%s_beta%s'%(defense, beta))
        margins = get_margins(X, Y, save_dir, beta, arch, adv=defense, eps=eps)
        if mode == 0:
            gamma = 1.
        elif mode == 1:
            gamma = dl_utils.get_overall_sn(save_dir, arch, return_snorms=False, beta=beta)
        else:
            snorms = dl_utils.get_overall_sn(save_dir, arch, return_snorms=True, beta=beta)
            c = 1.
            for i in range(len(key_order)):
                c += np.prod([snorms[key_order[j]] for j in range(i)])
            gamma = np.prod(snorms.values())*c
        # correct by sum of frobenius norms divided by spectral norms
        if extra_correction:
            d = dl_utils.get_weights(save_dir, arch, beta=beta)
            snorms = dl_utils.get_overall_sn(save_dir, arch, return_snorms=True, beta=beta)
            c = 0.
            for k in key_order:
                fnorm = np.sum(np.square(d[k]))
                if 'conv' in k: 
                    fnorm *= np.prod(conv_input_shapes[k])/2**2 # Divide by stride length squared
                c += fnorm/snorms[k]**2
            gamma *= np.sqrt(c)
        all_gammas.append(gamma)
        all_margins.append(margins/gamma)
    return all_margins, all_gammas   


def smoothen(x, c=10):
    """Smoothen curves for plotting"""
    x_new = np.array(x)
    for i in range(len(x)):
        if i >= c-1:
            x_new[i] = np.mean(x[np.max((c-1, i-c+1)):i+1])
    return x_new


def plot_curve_set(curve_set, num_batches_per_epoch=1, lw=2, xm=None,
                   left_marker_pos=0.2, right_marker_pos=0.8, legend=True):
    """Plot set of curves in curve_set, which is a dictionary with 
       keys being legend labels and values being arrays to plot
    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['s', 'D', 'v', 'o', '+', '.', '^', '*']
    
    num_points = np.max([len(curve_set[i]) for i in curve_set])
    if xm is None or len(xm) != len(curve_set):
        xm = np.random.choice(range(int(num_points*left_marker_pos),
                                    int(num_points*right_marker_pos)), 4)
    
    for i, k in enumerate(sorted(curve_set)):
        xaxis = np.arange(len(curve_set[k]))*num_batches_per_epoch
        plt.plot(xaxis, curve_set[k], lw=lw, c=colors[i])
        
        if len(curve_set[k]) < num_points:
            plt.plot([len(curve_set[k])*num_batches_per_epoch, num_points*num_batches_per_epoch],
                     [curve_set[k][-1], curve_set[k][-1]], '--', lw=lw, c=colors[i])
        
        if xm[i] > len(curve_set[k]):
            y_dots = [curve_set[k][-1], curve_set[k][-1]]
        else:
            y_dots = [curve_set[k][xm[i]], curve_set[k][xm[i]]]
        plt.plot([xm[i]*num_batches_per_epoch, xm[i]*num_batches_per_epoch],
                 y_dots,
                 lw=lw, c=colors[i], label=k,
                 marker=markers[i], ms=10, markeredgecolor='k')
        
    if legend: plt.legend()
        
        
def print_best_beta(tr_accs, va_accs, tt_accs, verbose=False, printinf=True, return_beta=False):
    """Choose the value of beta that achieved the highest validation accuracy"""
    best_beta = sorted(va_accs.items(), key=lambda x:x[1])[-1][0]
    if verbose:
        if printinf:
            print('beta = inf:\ttrain acc %.4f\tvalidation acc %.4f\ttest acc %.4f'\
                  %(tr_accs[np.inf], va_accs[np.inf], tt_accs[np.inf]))
        print('beta = %s:\ttrain acc %.4f\tvalidation acc %.4f\ttest acc %.4f'\
              %(best_beta, tr_accs[best_beta], va_accs[best_beta], tt_accs[best_beta]))
    if return_beta: 
        return best_beta


def get_table_results(Xtr, Ytr, Xva, Yva, Xtt, Ytt, arch, attacks_dict, results_file, dirname,
                      load_epoch=None, printinf=True, opt='momentum', order=2):
    """Get final train, validation, test accuracies for all trained networks in a directory"""
    num_channels = Xtr.shape[-1]
    if os.path.isfile(results_file):
        table_results = pickle.load(file(results_file, 'rb'))
    else:
        table_results = {}

    for adv in attacks_dict.keys():
        print('%s training'%(adv.upper()))
        eps = attacks_dict[adv]
        if adv in table_results:
            tr_accs, va_accs, tt_accs = table_results[adv]
        else:
            tr_accs, va_accs, tt_accs = {}, {}, {}
        for f in os.listdir(dirname):
            if adv not in f or 'pickle' in f or 'beta' not in f or 'rand' in f: continue
            save_dir = os.path.join(dirname, f)
            beta = float(f.split('beta')[1])
            if beta in tr_accs and beta in va_accs and beta in tt_accs: continue
            print('processing file %s...'%(f))
            tr_accs[beta] = dl_utils.build_graph_and_get_acc(Xtr, Ytr, arch, adv=adv, eps=eps,
                                                             save_dir=save_dir, beta=beta,
                                                             num_channels=num_channels, order=order,
                                                             load_epoch=load_epoch, opt=opt)
            va_accs[beta] = dl_utils.build_graph_and_get_acc(Xva, Yva, arch, adv=adv, eps=eps,
                                                             save_dir=save_dir, beta=beta,
                                                             num_channels=num_channels, order=order,
                                                             load_epoch=load_epoch, opt=opt)
            tt_accs[beta] = dl_utils.build_graph_and_get_acc(Xtt, Ytt, arch, adv=adv, eps=eps,
                                                             save_dir=save_dir, beta=beta,
                                                             num_channels=num_channels, order=order,
                                                             load_epoch=load_epoch, opt=opt)
            table_results[adv] = (tr_accs, va_accs, tt_accs)
            pickle.dump(table_results, file(results_file, 'wb'))
        print_best_beta(tr_accs, va_accs, tt_accs, verbose=True, printinf=printinf)
    return table_results        