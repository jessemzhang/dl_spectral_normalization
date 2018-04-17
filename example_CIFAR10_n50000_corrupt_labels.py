# Example script for training a model on datasets of varying sample corruption

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from spectral_adversarial_regularization import dl_utils
from get_cifar10 import get_cifar10_dataset

from spectral_adversarial_regularization.models import alexnet as model

arch_name = 'alexnet'
model = model.alexnet
wd = 1.0

maindir = 'save_weights_n50000_label_corrupt_%s_wd%s/'%(arch_name, wd)
retrain = True
save_every = 50
num_classes = 10
num_epochs = 500
n_samps = 50000

for p_corrupt_label in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
    
    # Load dataset
    Xtr, Ytr, Xtt, Ytt = get_cifar10_dataset(p_corrupt_label, n_samps=n_samps)
    val_set = {'X': Xtt[:500], 'Y': Ytt[:500]}
    
    # Train model
    save_dir = os.path.join(maindir, '%s_p%s/'%(arch_name, p_corrupt_label))
    if retrain: os.system('rm -rf %s'%(save_dir))
    _ = dl_utils.build_graph_and_train(Xtr, Ytr, save_dir, num_classes,
                                       arch=model,
                                       num_epochs=num_epochs,
                                       save_every=save_every,
                                       val_set=val_set,
                                       early_stop_acc=0.995)

