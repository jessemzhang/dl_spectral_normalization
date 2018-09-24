import time
import os
import numpy as np
import tensorflow as tf
import dl_utils

def l1_norm_tf(input_x, epsilon=1e-24):
    """get L1 norm"""
    reduc_ind = list(xrange(1, len(input_x.get_shape())))
    return tf.reduce_sum(tf.abs(input_x),
                         reduction_indices=reduc_ind,
                         keep_dims=True) + epsilon


def l2_norm_tf(input_x, epsilon=1e-24):
    """get L2 norm"""
    reduc_ind = list(xrange(1, len(input_x.get_shape())))
    return tf.sqrt(tf.reduce_sum(tf.square(input_x),
                                 reduction_indices=reduc_ind,
                                 keep_dims=True)) + epsilon


def project_back_onto_unit_ball(x_adv, x, eps=0.3, order=2):
    """Projects x_adv back to eps-ball around x"""
    
    delta = x_adv-x
    
    if order == 1:
        norms = l1_norm_tf(delta)
    elif order == 2:
        norms = l2_norm_tf(delta)
        
    adj_norms = tf.maximum(tf.ones_like(norms), norms/eps)
    return x+delta/adj_norms


def fgm(x, preds, y=None, eps=0.3, order=2, clip_min=None, clip_max=None,
        **kwargs):
    """
    TensorFlow implementation of the Fast Gradient Method. Code adapted from
    https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks_tf.py
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
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=preds)

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss_, x)
    
    if order == np.inf:
        # Take sign of gradient
        signed_grad = tf.sign(grad)
        
    elif order == 1:
        signed_grad = grad / l1_norm_tf(grad)

    elif order == 2:
        signed_grad = grad / l2_norm_tf(grad)

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

    
def wrm(x, preds, y=None, eps=0.3, order=2, model=None, k=15,
        reuse=True, update_collection='_', graph_beta=1.0, num_classes=10, training=False):
  
    """
    TensorFlow implementation of the Wasserstein distributionally
    adversarial training method. Code adapted from
    https://github.com/duchi-lab/certifiable-distributional-robustness/blob/master/attacks_tf.py
    :param x: the input placeholder
    :param preds: the model's output tensor
    :param y: (optional) A placeholder for the model labels. Only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param eps: .5 / gamma (Lagrange dual parameter) 
              in the ICLR paper (see link above)
    :param model: TF graph model (**kwargs goes to this)
    :param k: how many gradient ascent steps to take
              when finding adversarial example 
    :return: a tensor for the adversarial example
    """
    
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        y = tf.argmax(preds, 1)

    # Compute loss
    loss_ = dl_utils.loss(preds, y, mean=False)

    grad, = tf.gradients(eps*loss_, x)
    x_adv = tf.stop_gradient(x+grad)
    x = tf.stop_gradient(x)
    
    for t in xrange(k):
        loss_ = dl_utils.loss(model(x_adv, reuse=True, beta=graph_beta, 
                                    update_collection=update_collection,
                                    num_classes=num_classes), y, mean=False)
        grad, = tf.gradients(eps*loss_, x_adv)
        grad2, = tf.gradients(tf.nn.l2_loss(x_adv-x), x_adv)
        grad = grad - grad2
        x_adv = tf.stop_gradient(x_adv+1./np.sqrt(t+2)*grad)

    return x_adv


def pgm(x, preds, y=None, eps=0.3, order=2, model=None, a=None, k=15,
        reuse=True, update_collection='_', graph_beta=1., num_classes=10, training=False):
    """
    TensorFlow implementation of the Projected Gradient Method.
    :param x: the input placeholder
    :param preds: the model's output tensor
    :param y: (optional) A placeholder for the model labels. Only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param eps: the epsilon (input variation parameter)
    :param k: number of steps to take, each of size a
    :param a: size of each step
    :param model: TF graph model (**kwargs goes to this)
    :param ord: (optional) Order of the norm (mimics Numpy).
                Possible values: 1 or 2.
    :return: a tensor for the adversarial example
    """

    if a is None:
        a = 2.*eps/k

    if y is None:
        y = tf.argmax(preds, 1)
    
    x_adv = x
    
    for t in xrange(k):
        loss_ = dl_utils.loss(model(x_adv, reuse=reuse, beta=graph_beta,
                                    update_collection=update_collection,
                                    num_classes=num_classes), y, mean=False)
        grad, = tf.gradients(loss_, x_adv)
        
        if order == 1:
            scaled_grad = grad / l1_norm_tf(grad)
            
        elif order == 2:
            scaled_grad = grad / l2_norm_tf(grad)
            
        elif order == np.inf:
            scaled_grad = tf.sign(grad)
        
        x_adv = tf.stop_gradient(x_adv + a*scaled_grad)
        
        if order in [1, 2]:
            x_adv = project_back_onto_unit_ball(x_adv, x, eps=eps, order=order)
            
        elif order == np.inf:
            x_adv = tf.clip_by_value(x_adv, x-eps, x+eps)
        
    return x_adv
    
    
def gen_adv_examples_in_sess(X, graph, sess, batch_size=100, method=fgm, num_classes=10, **kwargs):
    """Use trained model to generate adversarial examples from X within a session"""

    adv_tensor = method(graph['input_data'], graph['fc_out'], num_classes=num_classes, **kwargs)

    adv_x = np.zeros(np.shape(X))
    for i in range(0, len(X), batch_size):
        adv_x[i:i+batch_size] = sess.run(adv_tensor, feed_dict={graph['input_data']: X[i:i+batch_size]})
        
    return adv_x


def build_graph_and_gen_adv_examples(X, arch, load_dir, num_classes=10, beta=1, num_channels=3,
                                     gpu_prop=0.2, gpu_id=0, load_epoch=None, method=fgm, **kwargs):
    """Build a tensorflow graph and generate adversarial examples"""
    
    if load_epoch is None:
        load_epoch = dl_utils.latest_epoch(load_dir)
    else:
        load_epoch = np.min((dl_utils.latest_epoch(load_dir), load_epoch))
        
    tf.reset_default_graph()
    with tf.device("/gpu:%s"%(gpu_id)):
        graph = dl_utils.graph_builder_wrapper(arch, num_classes=num_classes, save_dir=load_dir, beta=beta,
                                               num_channels=num_channels, update_collection='_')

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_prop))) as sess:
            graph['saver'].restore(sess, os.path.join(load_dir, 'checkpoints', 'epoch%s'%(load_epoch)))
            return gen_adv_examples_in_sess(X, graph, sess, method=method, model=arch, graph_beta=beta, 
                                            num_classes=num_classes, **kwargs)
            

def test_net_against_adv_examples(X, Y, load_dir, arch, d=None, beta=1., num_channels=3,
                                  verbose=True, gpu_id=0, gpu_prop=0.2, load_epoch=None,
                                  fix_adv=False, num_classes=10, method=fgm, opt='momentum', **kwargs):
    """For a trained network, generate and get accuracy for adversarially-perturbed samples"""
    
    start = time.time()
        
    # Use previously fitted network which had achieved 100% training accuracy
    tf.reset_default_graph()
    with tf.device("/gpu:%s"%(gpu_id)):
        graph = dl_utils.graph_builder_wrapper(arch, num_classes=num_classes, save_dir=load_dir, beta=beta,
                                               num_channels=num_channels, update_collection='_', opt=opt)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_prop))) as sess:
            if d is None:
                if load_epoch is None:
                    load_epoch = dl_utils.latest_epoch(load_dir)
                else:
                    load_epoch = np.min((dl_utils.latest_epoch(load_dir), load_epoch))
                    
                graph['saver'].restore(sess, os.path.join(load_dir, 'checkpoints', 'epoch%s'%(load_epoch)))

            else:
                for v in tf.trainable_variables():
                    sess.run(v.assign(d[v.name]))
            
            # Generate adversarial samples and predict
            X_adv = gen_adv_examples_in_sess(X, graph, sess, method=method, model=arch, graph_beta=beta, 
                                             num_classes=num_classes, **kwargs)
            
            # Gradients for some examples will sometimes be zero.. ignore this
            if fix_adv:
                reduc_ind = tuple(xrange(1, len(X.shape)))
                mag_delta = np.sqrt(np.sum(np.square(X_adv-X), axis=reduc_ind))
                keep_inds = mag_delta > 1e-4
                if np.sum(keep_inds) > 0:
                    X_adv, Y = X_adv[keep_inds], Y[keep_inds]
            
            Yhat_adv = dl_utils.predict_labels_in_sess(X_adv, graph, sess)

    accs_adv = np.sum(Yhat_adv == Y)/float(len(Y))

    if verbose:
        print('Acc on adv examples: %.4f (%.3f s elapsed)' \
              %(accs_adv, time.time()-start))

    return accs_adv
