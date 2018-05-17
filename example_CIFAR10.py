# Example script for training a model

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import numpy as np

from spectral_adversarial_regularization import dl_utils
from spectral_adversarial_regularization.models import alexnet as model

models = [model.alexnet, model.alexnet_sn]
beta_list = [0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 4.0]
methods = ['erm', 'fgm', 'pgm', 'wrm']
arch_name = 'alexnet'
maindir = 'save_weights/cifar10/%s/'%(arch_name)

# ------------------------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------------------------
from get_cifar10 import get_cifar10_dataset
n_samps = 50000
# each value: Xtr, Ytr, Xtt, Ytt
data = {
    'true': get_cifar10_dataset(0, n_samps=n_samps),
    'rand': get_cifar10_dataset(100, n_samps=n_samps)
}
C2 = np.mean([np.sqrt(np.sum(np.square(i))) for i in data['true'][0]])
eps = 0.05*C2


# ------------------------------------------------------------------------------
# Train network
# ------------------------------------------------------------------------------
def train_network(Xtr, Ytr, val_set, arch, save_dir, eps=0.3, adv=None,
                  gpu_prop=0.2, num_epochs=200, save_every=25, beta=1,
                  retrain=True):
    
    if retrain: os.system('rm -rf %s'%(save_dir))
        
    _ = dl_utils.build_graph_and_train(Xtr, Ytr, save_dir, arch,
                                       eps=eps,
                                       adv=adv,
                                       num_epochs=num_epochs,
                                       save_every=save_every,
                                       num_channels=3,
                                       batch_size=128,
                                       val_set=val_set,
                                       early_stop_acc=0.999,
                                       early_stop_acc_num=5,
                                       gpu_prop=gpu_prop,
                                       beta=beta)

    
# ------------------------------------------------------------------------------
# 1. Basic model (no spectral normalization)
# ------------------------------------------------------------------------------

for k in data:
    
    Xtr, Ytr, Xtt, Ytt = data[k]
    val_set = {'X': Xtt[:500], 'Y': Ytt[:500]}
    
    for method in methods:
        save_dir = os.path.join(maindir, '%s_%s_%s'%(arch_name, method, k))
        train_network(Xtr, Ytr, val_set, models[0], save_dir, eps=eps, adv=method)


# ------------------------------------------------------------------------------
# 2. Spectral normalization on all layers
# ------------------------------------------------------------------------------
        
for k in data:
    
    Xtr, Ytr, Xtt, Ytt = data[k]
    val_set = {'X': Xtt[:500], 'Y': Ytt[:500]}
    
    for beta in beta_list:
        for method in methods:
            save_dir = os.path.join(maindir, '%s_%s_%s_beta%s'%(arch_name, method, k, beta))
            train_network(Xtr, Ytr, val_set, models[1], save_dir, eps=eps, adv=method, beta=beta)