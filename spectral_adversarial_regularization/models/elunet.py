import tensorflow as tf
import numpy as np
from .. import sn

def elunet(input_data, NUM_CLASSES, nb_filters=64, wd=0, beta=1, update_collection=None):
    """Simple network for MNIST dataset 
        (as described in https://arxiv.org/pdf/1710.10571.pdf)
    """
    
    conv1 = tf.nn.elu(sn.conv2d(input_data, [8, 8, 1, nb_filters], stride=2, padding="SAME",
                                scope_name='conv1', spectral_norm=False))
    
    conv2 = tf.nn.elu(sn.conv2d(conv1, [6, 6, nb_filters, 2*nb_filters], stride=2, padding="VALID",
                                scope_name='conv2', spectral_norm=False))
    
    conv3 = tf.nn.elu(sn.conv2d(conv2, [5, 5, 2*nb_filters, 2*nb_filters], stride=1, padding="VALID",
                                scope_name='conv3', spectral_norm=False))
    
    reshape = tf.reshape(conv3, [-1, 128])
    
    fc = sn.linear(reshape, NUM_CLASSES, scope_name='fc', spectral_norm=False)
    
    return fc


def elunet_sn(input_data, NUM_CLASSES, nb_filters=64, wd=0, beta=1, update_collection=None):
    """Simple network for MNIST dataset with spectral normalization on all layers
        (as described in https://arxiv.org/pdf/1710.10571.pdf)
    """
    
    conv1 = tf.nn.elu(sn.conv2d(input_data, [8, 8, 1, nb_filters], stride=2, padding="SAME",
                                scope_name='conv1', tighter_sn=True, update_collection=update_collection, beta=beta))
    
    conv2 = tf.nn.elu(sn.conv2d(conv1, [6, 6, nb_filters, 2*nb_filters], stride=2, padding="VALID",
                                scope_name='conv2', tighter_sn=True, update_collection=update_collection, beta=beta))
    
    conv3 = tf.nn.elu(sn.conv2d(conv2, [5, 5, 2*nb_filters, 2*nb_filters], stride=1, padding="VALID",
                                scope_name='conv3', tighter_sn=True, update_collection=update_collection, beta=beta))
    
    reshape = tf.reshape(conv3, [-1, 128])
    
    fc = sn.linear(reshape, NUM_CLASSES, scope_name='fc', update_collection=update_collection, beta=beta)
    
    return fc