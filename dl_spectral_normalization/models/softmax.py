import tensorflow as tf
import numpy as np
from .. import sn

def softmax_sn(input_data, num_classes=10, wd=0, beta=1, update_collection=None, reuse=None, training=False):
    """Tensorflow implementation of softmax regression (one-layer NN)"""
    
    fc = sn.linear(input_data, num_classes, scope_name='fc', update_collection=update_collection,
                   beta=beta, reuse=reuse)
    
    return fc