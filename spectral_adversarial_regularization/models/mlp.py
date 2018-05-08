# https://github.com/rharish101/DLGeneralization/

import tensorflow as tf
import numpy as np
from .. import sn

def mlp1(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None):
    """1-hidden-layer Multilayer Perceptron architecture"""

    hidden = tf.nn.relu(sn.linear(input_data, 512, scope_name='hidden', xavier=True, spectral_norm=False, reuse=reuse))
#    tf.add_to_collection('hidden', hidden) # useful for debugging
    fc = sn.linear(hidden, num_classes, scope_name='fc', xavier=True, spectral_norm=False, reuse=reuse)

    return fc


def mlp1_sn(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None):
    """1-hidden-layer Multilayer Perceptron architecture with spectral normalization on all layers"""

    hidden = tf.nn.relu(sn.linear(input_data, 512, scope_name='hidden', xavier=True,
                                  update_collection=update_collection, beta=beta, reuse=reuse))
    fc = sn.linear(hidden, num_classes, scope_name='fc', xavier=True,
                   update_collection=update_collection, beta=beta, reuse=reuse)
        
    return fc


def mlp1_snl2(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None):
    """1-hidden-layer Multilayer Perceptron architecture with spectral normalization on all layers
       except last one, which can be L2 regularized
    """

    hidden = tf.nn.relu(sn.linear(input_data, 512, scope_name='hidden', xavier=True,
                                  update_collection=update_collection, beta=beta, reuse=reuse))
    fc = sn.linear(hidden, num_classes, scope_name='fc', spectral_norm=False,
                   xavier=True, wd=wd, l2_norm=True, reuse=reuse)
        
    return fc


def mlp3(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None):
    """3-hidden-layer Multilayer Perceptron architecture"""

    hidden1 = tf.nn.relu(sn.linear(input_data, 512, scope_name='hidden1', xavier=True, spectral_norm=False, reuse=reuse))
    hidden2 = tf.nn.relu(sn.linear(hidden1, 512, scope_name='hidden2', xavier=True, spectral_norm=False, reuse=reuse))
    hidden3 = tf.nn.relu(sn.linear(hidden2, 512, scope_name='hidden3', xavier=True, spectral_norm=False, reuse=reuse))
    fc = sn.linear(hidden3, num_classes, scope_name='fc', xavier=True, spectral_norm=False, reuse=reuse)
        
    return fc


def mlp3_sn(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None):
    """1-hidden-layer Multilayer Perceptron architecture with spectral normalization on all layers"""

    hidden1 = tf.nn.relu(sn.linear(input_data, 512, scope_name='hidden1', xavier=True,
                                   update_collection=update_collection, beta=beta, reuse=reuse))
    hidden2 = tf.nn.relu(sn.linear(hidden1, 512, scope_name='hidden2', xavier=True,
                                   update_collection=update_collection, beta=beta, reuse=reuse))
    hidden3 = tf.nn.relu(sn.linear(hidden2, 512, scope_name='hidden3', xavier=True,
                                   update_collection=update_collection, beta=beta, reuse=reuse))
    fc = sn.linear(hidden3, num_classes, scope_name='fc', xavier=True,
                   update_collection=update_collection, beta=beta, reuse=reuse)
        
    return fc


def mlp3_snl2(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None):
    """3-hidden-layer Multilayer Perceptron architecture with spectral normalization on all layers 
       except last one, which can be L2 regularized
    """

    hidden1 = tf.nn.relu(sn.linear(input_data, 512, scope_name='hidden1', xavier=True,
                                   update_collection=update_collection, beta=beta, reuse=reuse))
    hidden2 = tf.nn.relu(sn.linear(hidden1, 512, scope_name='hidden2', xavier=True,
                                   update_collection=update_collection, beta=beta, reuse=reuse))
    hidden3 = tf.nn.relu(sn.linear(hidden2, 512, scope_name='hidden3', xavier=True,
                                   update_collection=update_collection, beta=beta, reuse=reuse))
    fc = sn.linear(hidden3, num_classes, scope_name='fc', spectral_norm=False,
                   xavier=True, wd=wd, l2_norm=True, reuse=reuse)
        
    return fc
