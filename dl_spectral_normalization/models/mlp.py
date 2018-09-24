# https://github.com/rharish101/DLGeneralization/

import tensorflow as tf
import numpy as np
from .. import sn


def mlp1_relu_sn(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None, training=False):
    """1-hidden-layer Multilayer Perceptron architecture with spectral normalization on all layers"""

    hidden = tf.nn.relu(sn.linear(input_data, 512, scope_name='hidden', xavier=True,
                                  update_collection=update_collection, beta=beta, reuse=reuse))
    fc = sn.linear(hidden, num_classes, scope_name='fc', xavier=True,
                   update_collection=update_collection, beta=beta, reuse=reuse)
        
    return fc


def mlp1_elu_sn(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None, training=False):
    """1-hidden-layer Multilayer Perceptron architecture with spectral normalization on all layers"""

    hidden = tf.nn.elu(sn.linear(input_data, 512, scope_name='hidden', xavier=True,
                                  update_collection=update_collection, beta=beta, reuse=reuse))
    fc = sn.linear(hidden, num_classes, scope_name='fc', xavier=True,
                   update_collection=update_collection, beta=beta, reuse=reuse)
        
    return fc


def mlp2_relu_sn(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None, training=False):
    """2-hidden-layer Multilayer Perceptron architecture with spectral normalization on all layers"""

    hidden1 = tf.nn.relu(sn.linear(input_data, 512, scope_name='hidden1', xavier=True,
                                   update_collection=update_collection, beta=beta, reuse=reuse))
    hidden2 = tf.nn.relu(sn.linear(hidden1, 512, scope_name='hidden2', xavier=True,
                                   update_collection=update_collection, beta=beta, reuse=reuse))
    fc = sn.linear(hidden2, num_classes, scope_name='fc', xavier=True,
                   update_collection=update_collection, beta=beta, reuse=reuse)
        
    return fc


def mlp2_elu_sn(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None, training=False):
    """2-hidden-layer Multilayer Perceptron architecture with spectral normalization on all layers"""

    hidden1 = tf.nn.elu(sn.linear(input_data, 512, scope_name='hidden1', xavier=True,
                                  update_collection=update_collection, beta=beta, reuse=reuse))
    hidden2 = tf.nn.elu(sn.linear(hidden1, 512, scope_name='hidden2', xavier=True,
                                  update_collection=update_collection, beta=beta, reuse=reuse))
    fc = sn.linear(hidden2, num_classes, scope_name='fc', xavier=True,
                   update_collection=update_collection, beta=beta, reuse=reuse)
        
    return fc
