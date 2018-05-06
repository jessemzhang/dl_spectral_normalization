import tensorflow as tf
import numpy as np
from .. import sn

def elunet(input_data, num_classes=10, nb_filters=64, wd=0, beta=1, update_collection=None, reuse=None):
    """Simple network for MNIST dataset 
        (as described in https://arxiv.org/pdf/1710.10571.pdf)
    """
    
    conv1 = tf.nn.elu(sn.conv2d(input_data, [8, 8, 1, nb_filters], stride=2, padding="SAME",
                                scope_name='conv1', spectral_norm=False, xavier=True, reuse=reuse))
    
    conv2 = tf.nn.elu(sn.conv2d(conv1, [6, 6, nb_filters, 2*nb_filters], stride=2, padding="VALID",
                                scope_name='conv2', spectral_norm=False, xavier=True, reuse=reuse))
    
    conv3 = tf.nn.elu(sn.conv2d(conv2, [5, 5, 2*nb_filters, 2*nb_filters], stride=1, padding="VALID",
                                scope_name='conv3', spectral_norm=False, xavier=True, reuse=reuse))
    
    reshape = tf.reshape(conv3, [-1, 128])
    
    fc = sn.linear(reshape, num_classes, scope_name='fc', spectral_norm=False, xavier=True, reuse=reuse)
    
    return fc


def elunet_sn(input_data, num_classes=10, nb_filters=64, wd=0, beta=1, update_collection=None, reuse=None):
    """Simple network for MNIST dataset with spectral normalization on all layers
        (as described in https://arxiv.org/pdf/1710.10571.pdf)
    """
    
    conv1 = tf.nn.elu(sn.conv2d(input_data, [8, 8, 1, nb_filters], stride=2, padding="SAME",
                                scope_name='conv1', update_collection=update_collection,
                                beta=beta, reuse=reuse))
    
    conv2 = tf.nn.elu(sn.conv2d(conv1, [6, 6, nb_filters, 2*nb_filters], stride=2, padding="VALID",
                                scope_name='conv2', update_collection=update_collection,
                                beta=beta, reuse=reuse))
    
    conv3 = tf.nn.elu(sn.conv2d(conv2, [5, 5, 2*nb_filters, 2*nb_filters], stride=1, padding="VALID",
                                scope_name='conv3', update_collection=update_collection,
                                beta=beta, reuse=reuse))
    
    reshape = tf.reshape(conv3, [-1, 128])
    
    fc = sn.linear(reshape, num_classes, scope_name='fc', update_collection=update_collection,
                   beta=beta, reuse=reuse)
    
    return fc