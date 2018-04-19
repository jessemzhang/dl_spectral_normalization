# Code adapted and modified from https://github.com/openai/cleverhans

import time
import os
import numpy as np
import tensorflow as tf
import dl_utils

def fgm(x, preds, y=None, eps=0.3, order=np.inf, clip_min=None, clip_max=None):
    """
    TensorFlow implementation of the Fast Gradient Method.
    :param x: the input placeholder
    :param preds: the model's output tensor
    :param y: (optional) A placeholder for the model labels. Only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param eps: the epsilon (input variation parameter)
    :param ord: (optional) Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :return: a tensor for the adversarial example
    """

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        y = tf.argmax(preds, 1)


    # Compute loss (without taking the mean across samples)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=preds)

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss_, x)

    if order == np.inf:
        # Take sign of gradient
        signed_grad = tf.sign(grad)
    elif order == 1:
        reduc_ind = list(xrange(1, len(x.get_shape())))
        signed_grad = grad / tf.reduce_sum(tf.abs(grad),
                                           reduction_indices=reduc_ind,
                                           keep_dims=True)
    elif order == 2:
        reduc_ind = list(xrange(1, len(x.get_shape())))
        signed_grad = grad / tf.sqrt(tf.reduce_sum(tf.square(grad),
                                                   reduction_indices=reduc_ind,
                                                   keep_dims=True))
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")

    # Multiply by constant epsilon
    scaled_signed_grad = eps * signed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_signed_grad)

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x


def gen_adv_examples_in_sess(X, graph, sess, batch_size=100):
    """Use trained model to generate adversarial examples from X within a session"""
    
    adv_x = np.zeros(np.shape(X))
    for i in range(0,len(X),batch_size):
        adv_x_ = sess.run(graph['adv_x'], feed_dict={graph['input_data']:X[i:i+batch_size]})
        adv_x[i:i+batch_size] = adv_x_
    return adv_x
    

def gen_adv_examples(X, graph, load_dir, sess=None, batch_size=100, load_epoch=None):
    """Use trained model to generate adversarial examples from X"""

    # Load from checkpoint corresponding to latest epoch if none given
    if load_epoch == None:
        load_epoch = max([int(f.split('epoch')[1].split('.')[0]) 
                          for f in os.listdir(os.path.join(load_dir, 'checkpoints')) if 'epoch' in f])

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        graph['saver'].restore(sess, os.path.join(load_dir, 'checkpoints', 'epoch%s'%(load_epoch)))
        return adv_x(X, graph, sess)


def test_net_against_adv_examples(X, Y, load_dir, arch, d=None,
                                  eps=0.3, order=np.inf, gpu_id=0, verbose=True):
    """For a trained network, generate and get accuracy for adversarially-perturbed samples"""
    
    num_classes = len(np.unique(Y))
    start = time.time()
        
    # Use previously fitted network which had achieved 100% training accuracy
    tf.reset_default_graph()
    with tf.device("/gpu:%s"%(gpu_id)):
        graph = dl_utils.graph_builder_wrapper(num_classes, load_dir, eps=eps, order=order,
                                               arch=arch, update_collection='_')

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            if d is None:
                load_epoch = dl_utils.latest_epoch(load_dir)
                graph['saver'].restore(sess, os.path.join(load_dir, 'checkpoints', 'epoch%s'%(load_epoch)))

            else:
                for v in tf.trainable_variables():
                    sess.run(v.assign(d[v.name]))
            
            # Generate adversarial samples and predict
            X_adv = gen_adv_examples_in_sess(X, graph, sess)
            Yhat_adv = dl_utils.predict_labels_in_sess(X_adv, graph, sess)

    accs_adv = np.sum(Yhat_adv == Y)/float(len(Y))

    if verbose:
        print('Acc on adv examples: %.2f (%.3f s elapsed)' \
              %(accs_adv, time.time()-start))

    return accs_adv
            
