
# coding: utf-8

# In[ ]:

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[ ]:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from spectral_adversarial_regularization.models import alexnet_small as model
from spectral_adversarial_regularization import dl_utils
from get_cifar10 import get_cifar10_dataset

maindir = 'save_weights_n20000_alexnet_small/'
retrain = True
save_every = 50
num_classes = 10
num_epochs = 500


# In[ ]:

# Load dataset
n_samps = 50000

# each value: Xtr, Ytr, Xtt, Ytt
data = {
    'true': get_cifar10_dataset(0, n_samps=n_samps),
    'rand': get_cifar10_dataset(100, n_samps=n_samps)
}


# # 1. Basic model (no spectral normalization or regularization)

# In[ ]:

for k in data:
    
    Xtr, Ytr, Xtt, Ytt = data[k]
    val_set = {'X': Xtt[:500], 'Y': Ytt[:500]}

    save_dir = '%salexnet_%s/'%(maindir, k)
    if retrain: os.system('rm -rf %s'%(save_dir))

    _ = dl_utils.build_graph_and_train(Xtr, Ytr, save_dir, num_classes,
                                       arch=model.alexnet,
                                       num_epochs=num_epochs,
                                       save_every=save_every,
                                       val_set=val_set,
                                       early_stop_acc=0.995)


# # 2. Spectral normalization on all layers

# In[ ]:

for k in data:
    
    Xtr, Ytr, Xtt, Ytt = data[k]
    val_set = {'X': Xtt[:500], 'Y': Ytt[:500]}

    save_dir = '%salexnet_%s_sn/'%(maindir, k)
    if retrain: os.system('rm -rf %s'%(save_dir))

    _ = dl_utils.build_graph_and_train(Xtr, Ytr, save_dir, num_classes,
                                       arch=model.alexnet_sn,
                                       num_epochs=num_epochs,
                                       save_every=save_every,
                                       val_set=val_set,
                                       early_stop_acc=0.995)


# # 3. Spectral normalization on all except last layer, which is L2 regularized

# In[ ]:

for wd in [0, 4e-2, 4e-1, 1, 4]:

    for k in data:

        Xtr, Ytr, Xtt, Ytt = data[k]
        val_set = {'X': Xtt[:500], 'Y': Ytt[:500]}

        save_dir = '%salexnet_%s_sar_wd%s/'%(maindir, k, wd)
        if retrain: os.system('rm -rf %s'%(save_dir))

        _ = dl_utils.build_graph_and_train(Xtr, Ytr, save_dir, num_classes,
                                           wd=wd,
                                           arch=model.alexnet_sar,
                                           num_epochs=num_epochs,
                                           save_every=save_every,
                                           val_set=val_set,
                                           early_stop_acc=0.995)

