# https://github.com/rharish101/DLGeneralization/

import tensorflow as tf
import numpy as np
from .. import sn

def mlp1(NUM_CLASSES, wd=0, update_collection=None):
    """1-hidden-layer Multilayer Perceptron architecture"""

    input_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='in_data')
    input_labels = tf.placeholder(tf.int64, shape=[None], name='in_labels')

    hidden = tf.nn.relu(sn.linear(input_data, 512, scope_name='hidden', xavier=True, spectral_norm=False))
#    tf.add_to_collection('hidden', hidden) # useful for debugging
    fc = sn.linear(hidden, NUM_CLASSES, scope_name='fc', xavier=True, spectral_norm=False)

    return input_data, input_labels, fc


def mlp1_sn(NUM_CLASSES, wd=0, update_collection=None):
    """1-hidden-layer Multilayer Perceptron architecture with spectral normalization on all layers"""

    input_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='in_data')
    input_labels = tf.placeholder(tf.int64, shape=[None], name='in_labels')

    hidden = tf.nn.relu(sn.linear(input_data, 512, scope_name='hidden', xavier=True, update_collection=update_collection))
    fc = sn.linear(hidden, NUM_CLASSES, scope_name='fc', xavier=True, update_collection=update_collection)
        
    return input_data, input_labels, fc


def mlp1_sar(NUM_CLASSES, wd=0, update_collection=None):
    """1-hidden-layer Multilayer Perceptron architecture with spectral adversarial regularization"""

    input_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='in_data')
    input_labels = tf.placeholder(tf.int64, shape=[None], name='in_labels')

    hidden = tf.nn.relu(sn.linear(input_data, 512, scope_name='hidden', xavier=True, update_collection=update_collection))
    fc = sn.linear(hidden, NUM_CLASSES, scope_name='fc', spectral_norm=False,
                   xavier=True, wd=wd, l2_norm=True)
        
    return input_data, input_labels, fc


def mlp3(NUM_CLASSES, wd=0, update_collection=None):
    """3-hidden-layer Multilayer Perceptron architecture"""

    input_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='in_data')
    input_labels = tf.placeholder(tf.int64, shape=[None], name='in_labels')

    hidden1 = tf.nn.relu(sn.linear(input_data, 512, scope_name='hidden1', xavier=True, spectral_norm=False))
    hidden2 = tf.nn.relu(sn.linear(hidden1, 512, scope_name='hidden2', xavier=True, spectral_norm=False))
    hidden3 = tf.nn.relu(sn.linear(hidden2, 512, scope_name='hidden3', xavier=True, spectral_norm=False))
    fc = sn.linear(hidden3, NUM_CLASSES, scope_name='fc', xavier=True, spectral_norm=False)
        
    return input_data, input_labels, fc


def mlp3_sn(NUM_CLASSES, wd=0, update_collection=None):
    """1-hidden-layer Multilayer Perceptron architecture with spectral normalization on all layers"""

    input_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='in_data')
    input_labels = tf.placeholder(tf.int64, shape=[None], name='in_labels')

    hidden1 = tf.nn.relu(sn.linear(input_data, 512, scope_name='hidden1', xavier=True, update_collection=update_collection))
    hidden2 = tf.nn.relu(sn.linear(hidden1, 512, scope_name='hidden2', xavier=True, update_collection=update_collection))
    hidden3 = tf.nn.relu(sn.linear(hidden2, 512, scope_name='hidden3', xavier=True, update_collection=update_collection))
    fc = sn.linear(hidden3, NUM_CLASSES, scope_name='fc', xavier=True, update_collection=update_collection)
        
    return input_data, input_labels, fc


def mlp3_sar(NUM_CLASSES, wd=0, update_collection=None):
    """3-hidden-layer Multilayer Perceptron architecture with spectral adversarial regularization"""

    input_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='in_data')
    input_labels = tf.placeholder(tf.int64, shape=[None], name='in_labels')

    hidden1 = tf.nn.relu(sn.linear(input_data, 512, scope_name='hidden1', xavier=True, update_collection=update_collection))
    hidden2 = tf.nn.relu(sn.linear(hidden1, 512, scope_name='hidden2', xavier=True, update_collection=update_collection))
    hidden3 = tf.nn.relu(sn.linear(hidden2, 512, scope_name='hidden3', xavier=True, update_collection=update_collection))
    fc = sn.linear(hidden3, NUM_CLASSES, scope_name='fc', spectral_norm=False,
                   xavier=True, wd=wd, l2_norm=True)
        
    return input_data, input_labels, fc
