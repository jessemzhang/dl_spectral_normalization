# https://github.com/rharish101/DLGeneralization/

import tensorflow as tf
import numpy as np
from .. import sn

def mlp1(input_data, NUM_CLASSES, wd=0, update_collection=None, beta=1.):
    """1-hidden-layer Multilayer Perceptron architecture"""

    hidden = tf.nn.relu(sn.linear(input_data, 512, scope_name='hidden', xavier=True, spectral_norm=False))
#    tf.add_to_collection('hidden', hidden) # useful for debugging
    fc = sn.linear(hidden, NUM_CLASSES, scope_name='fc', xavier=True, spectral_norm=False)

    return fc


def mlp1_sn(input_data, NUM_CLASSES, wd=0, update_collection=None, beta=1.):
    """1-hidden-layer Multilayer Perceptron architecture with spectral normalization on all layers"""

    hidden = tf.nn.relu(sn.linear(input_data, 512, scope_name='hidden', xavier=True, update_collection=update_collection, beta=beta))
    fc = sn.linear(hidden, NUM_CLASSES, scope_name='fc', xavier=True, update_collection=update_collection, beta=beta)
        
    return fc


def mlp1_sar(input_data, NUM_CLASSES, wd=0, update_collection=None, beta=1.):
    """1-hidden-layer Multilayer Perceptron architecture with spectral adversarial regularization"""

    hidden = tf.nn.relu(sn.linear(input_data, 512, scope_name='hidden', xavier=True, update_collection=update_collection, beta=beta))
    fc = sn.linear(hidden, NUM_CLASSES, scope_name='fc', spectral_norm=False,
                   xavier=True, wd=wd, l2_norm=True)
        
    return fc


def mlp3(input_data, NUM_CLASSES, wd=0, update_collection=None, beta=1.):
    """3-hidden-layer Multilayer Perceptron architecture"""

    hidden1 = tf.nn.relu(sn.linear(input_data, 512, scope_name='hidden1', xavier=True, spectral_norm=False))
    hidden2 = tf.nn.relu(sn.linear(hidden1, 512, scope_name='hidden2', xavier=True, spectral_norm=False))
    hidden3 = tf.nn.relu(sn.linear(hidden2, 512, scope_name='hidden3', xavier=True, spectral_norm=False))
    fc = sn.linear(hidden3, NUM_CLASSES, scope_name='fc', xavier=True, spectral_norm=False)
        
    return fc


def mlp3_sn(input_data, NUM_CLASSES, wd=0, update_collection=None, beta=1.):
    """1-hidden-layer Multilayer Perceptron architecture with spectral normalization on all layers"""

    hidden1 = tf.nn.relu(sn.linear(input_data, 512, scope_name='hidden1', xavier=True, update_collection=update_collection, beta=beta))
    hidden2 = tf.nn.relu(sn.linear(hidden1, 512, scope_name='hidden2', xavier=True, update_collection=update_collection, beta=beta))
    hidden3 = tf.nn.relu(sn.linear(hidden2, 512, scope_name='hidden3', xavier=True, update_collection=update_collection, beta=beta))
    fc = sn.linear(hidden3, NUM_CLASSES, scope_name='fc', xavier=True, update_collection=update_collection, beta=beta)
        
    return fc


def mlp3_sar(input_data, NUM_CLASSES, wd=0, update_collection=None, beta=1.):
    """3-hidden-layer Multilayer Perceptron architecture with spectral adversarial regularization"""

    hidden1 = tf.nn.relu(sn.linear(input_data, 512, scope_name='hidden1', xavier=True, update_collection=update_collection, beta=beta))
    hidden2 = tf.nn.relu(sn.linear(hidden1, 512, scope_name='hidden2', xavier=True, update_collection=update_collection, beta=beta))
    hidden3 = tf.nn.relu(sn.linear(hidden2, 512, scope_name='hidden3', xavier=True, update_collection=update_collection, beta=beta))
    fc = sn.linear(hidden3, NUM_CLASSES, scope_name='fc', spectral_norm=False,
                   xavier=True, wd=wd, l2_norm=True)
        
    return fc
