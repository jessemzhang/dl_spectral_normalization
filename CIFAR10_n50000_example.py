# Example script for training a model

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from spectral_adversarial_regularization import dl_utils
from get_cifar10 import get_cifar10_dataset

from spectral_adversarial_regularization.models import alexnet as model

arch_name = 'alexnet'
models = [
    model.alexnet,
    model.alexnet_sn,
    model.alexnet_sar
]
wd_list = [0, 4e-2, 4e-1, 1, 4]

maindir = 'save_weights_n50000_%s/'%(arch_name)
retrain = True
save_every = 50
num_classes = 10
num_epochs = 500

# ------------------------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------------------------
n_samps = 50000

# each value: Xtr, Ytr, Xtt, Ytt
data = {
    'true': get_cifar10_dataset(0, n_samps=n_samps),
    'rand': get_cifar10_dataset(100, n_samps=n_samps)
}

# ------------------------------------------------------------------------------
# 1. Basic model (no spectral normalization or regularization)
# ------------------------------------------------------------------------------

for k in data:
    
    Xtr, Ytr, Xtt, Ytt = data[k]
    val_set = {'X': Xtt[:500], 'Y': Ytt[:500]}

    save_dir = os.path.join(maindir, '%s_%s'%(arch_name, k))
    if retrain: os.system('rm -rf %s'%(save_dir))

    _ = dl_utils.build_graph_and_train(Xtr, Ytr, save_dir, num_classes,
                                       arch=models[0],
                                       num_epochs=num_epochs,
                                       save_every=save_every,
                                       val_set=val_set,
                                       early_stop_acc=0.995)


# ------------------------------------------------------------------------------
# 2. Spectral normalization on all layers
# ------------------------------------------------------------------------------

for k in data:
    
    Xtr, Ytr, Xtt, Ytt = data[k]
    val_set = {'X': Xtt[:500], 'Y': Ytt[:500]}

    save_dir = os.path.join(maindir, '%s_%s_sn'%(arch_name, k))
    if retrain: os.system('rm -rf %s'%(save_dir))

    _ = dl_utils.build_graph_and_train(Xtr, Ytr, save_dir, num_classes,
                                       arch=models[1],
                                       num_epochs=num_epochs,
                                       save_every=save_every,
                                       val_set=val_set,
                                       early_stop_acc=0.995)


# ------------------------------------------------------------------------------
# 3. Spectral normalization on all except last layer, which is L2 regularized
# ------------------------------------------------------------------------------

ignore_set = set()
for wd in wd_list:

    for k in data:
        
        if k in ignore_set:
            print('Graph unable to fit %s labels with wd < %.2e. Skipping..'
                  %(k, wd))
            continue

        Xtr, Ytr, Xtt, Ytt = data[k]
        val_set = {'X': Xtt[:500], 'Y': Ytt[:500]}

        save_dir = os.path.join(maindir, '%s_%s_sar_wd%s'%(arch_name, k, wd))
        if retrain: os.system('rm -rf %s'%(save_dir))

        tr_acc = dl_utils.build_graph_and_train(Xtr, Ytr, save_dir, num_classes,
                                                wd=wd,
                                                arch=models[2],
                                                num_epochs=num_epochs,
                                                save_every=save_every,
                                                val_set=val_set,
                                                early_stop_acc=0.995)
        
        if tr_acc < 0.22:
            ignore_set.add(k)

