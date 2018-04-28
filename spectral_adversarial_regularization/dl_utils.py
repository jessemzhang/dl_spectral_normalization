from __future__ import print_function

import time
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from adversarial import fgm
from sklearn.utils import shuffle

import models.alexnet_small as model

def loss(g, Y):
    """Cross-entropy loss between labels and output of linear activation function"""
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, 
                                                                   logits=g, 
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def acc(g, Y):
    """Accuracy"""
    correct_prediction = tf.equal(Y, tf.argmax(g, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def graph_builder_wrapper(num_classes, save_dir,
                          wd=0,
                          eps=0.3,
                          order=np.inf,
                          max_save=200,
                          arch=model.alexnet,
                          update_collection=None,
                          beta=1.):
    """Wrapper for building graph and accessing all relevant ops/placeholders"""
    
    input_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='in_data')
    input_labels = tf.placeholder(tf.int64, shape=[None], name='in_labels')
    delta = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='delta') # used for adv robustness
    
    fc_out = arch(input_data-delta, num_classes, wd=wd, beta=beta, update_collection=update_collection)
    saver = tf.train.Saver(max_to_keep=max_save)
    
    # Loss and optimizer
    total_loss = loss(fc_out, input_labels)
    learning_rate = tf.Variable(0.01, name='learning_rate', trainable=False)
    opt_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)
    tf.summary.scalar('loss', total_loss)
    
    # Adv robustness using Duchi et al.'s ICLR 2018 result
    adv_loss = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_labels, logits=fc_out)
    delta_grad = tf.gradients(adv_loss, delta)[0]

    # Accuracy
    total_acc = acc(fc_out, input_labels)
    tf.summary.scalar('accuracy', total_acc)
    
    # Add histograms for trainable variables.
#     for var in tf.trainable_variables():
#         tf.summary.histogram(var.op.name, var)

    # Merge all the summaries and write them out to save_dir
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'train'))
    graph_writer = tf.summary.FileWriter(os.path.join(save_dir, 'graph'), graph=tf.get_default_graph())
    valid_writer = tf.summary.FileWriter(os.path.join(save_dir, 'validation'))

    # Fast gradient method for finding adversarial examples
    adv_x = fgm(input_data, fc_out, y=None, eps=eps, order=order, clip_min=None, clip_max=None)
    
    graph = dict(
        input_data = input_data,
        input_labels = input_labels,
        total_loss = total_loss,
        total_acc = total_acc,
        fc_out = fc_out,
        delta = delta,
        delta_grad = delta_grad,
        opt_step = opt_step,
        learning_rate = learning_rate,
        merged = merged,
        train_writer = train_writer,
        graph_writer = graph_writer,
        valid_writer = valid_writer,
        saver = saver,
        adv_x = adv_x
    )
    
    return graph


def train(Xtr, Ytr, graph, save_dir,
          val_set=None,
          adv_robustness=False,
          Ip=15,
          lambda_adv=0.5,
          lr_initial=0.01,
          seed=0,
          num_epochs=100,
          batch_size=100,
          write_every=1,
          save_every=None,
          verbose=True,
          load_epoch=-1,
          load_weights_file=None,
          early_stop_acc=None,
          early_stop_acc_num=10,
          gpu_prop=0.2):
    """Train the graph"""
    
    tf.set_random_seed(seed)
    
    if save_every is None:
        if num_epochs > 100:
            save_every = num_epochs/100
        else:
            save_every = 1
    
    start = time.time()
    training_losses, training_accs = [], []
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_prop))) as sess:

        sess.run(tf.global_variables_initializer())

        if load_weights_file is not None:
            if verbose:
                print('Loading weights from %s..'%(load_weights_file))
            graph['saver'].restore(sess,load_weights_file)

        elif load_epoch > -1:
            if verbose:
                print('Continuing training starting at epoch %s+1'%(load_epoch))
            restore_weights_file = os.path.join(save_dir, 'checkpoints', 'epoch%s'%(load_epoch))
            graph['saver'].restore(sess, restore_weights_file)

        else:
            if not os.path.exists(os.path.join(save_dir, 'checkpoints')):
                os.mkdir(os.path.join(save_dir, 'checkpoints'))
            graph['saver'].save(sess, os.path.join(save_dir, 'checkpoints', 'initial_weights'))
        
        for epoch in range(load_epoch+1, load_epoch+num_epochs+1):

            lr = lr_initial*0.95**(epoch/390.) # initial lr * decay rate ^(step/decay_steps)
            sess.run(graph['learning_rate'].assign(lr))

            t = time.time()
            training_loss = 0
            training_acc = 0
            steps = 0.
            Xtr_, Ytr_ = shuffle(Xtr, Ytr)

            if len(Xtr_)%batch_size == 0:
                end = len(Xtr_)
            else:
                end = len(Xtr_)-batch_size
            for i in range(0, end, batch_size):

                x, y = Xtr_[i:i+batch_size], Ytr_[i:i+batch_size]
                d = np.zeros(np.shape(x))

                if adv_robustness:
                    for _ in range(Ip):
                        d_ =  sess.run(graph['delta_grad'], feed_dict={graph['input_data']: x,
                                                                       graph['input_labels']: y,
                                                                       graph['delta']: d})
                        d -= lambda_adv*d_
                
                feed_dict = {graph['input_data']: x, graph['input_labels']: y, graph['delta']: d}
                training_loss_, training_acc_, _ = \
                    sess.run([graph['total_loss'], graph['total_acc'], graph['opt_step']],
                             feed_dict=feed_dict)
                training_loss += training_loss_
                training_acc += training_acc_
                steps += 1.

                if verbose:
                    print('\rEpoch %s/%s (%.3f s), batch %s/%s (%.3f s): loss %.3f, acc %.3f'
                          %(epoch+1, load_epoch+num_epochs+1, time.time()-start,steps,
                            len(Xtr_)/batch_size, time.time()-t, training_loss_, training_acc_),
                          end='')            
            
            if epoch%write_every == 0: # writing to tensorboard
                summary = sess.run(graph['merged'], feed_dict=feed_dict)
                graph['train_writer'].add_summary(summary, epoch)
                
                if val_set is not None: # make sure to keep the val_set small
                    feed_dict = {graph['input_data']: val_set['X'],
                                 graph['input_labels']: val_set['Y'],
                                 graph['delta']: np.zeros(np.shape(val_set['X']))}
                    summary = sess.run(graph['merged'], feed_dict=feed_dict)
                    graph['valid_writer'].add_summary(summary, epoch)

            if epoch%save_every == 0: # saving weights
                graph['saver'].save(sess, os.path.join(save_dir, 'checkpoints', 'epoch%s'%(epoch)))

            training_losses.append(training_loss/steps)
            training_accs.append(training_acc/steps)

            if early_stop_acc is not None and np.mean(training_accs[-early_stop_acc_num:]) >= early_stop_acc:
                if verbose:
                    print('\rMean acc >= %s for last 10 epochs. Stopping training after epoch %s/%s.'
                          %(early_stop_acc, epoch+1, load_epoch+num_epochs+1), end='')
                break

        print('\nDONE: epoch%s'%(epoch))
        if not os.path.exists(os.path.join(save_dir, 'checkpoints', 'epoch%s'%(epoch))):
            graph['saver'].save(sess, os.path.join(save_dir, 'checkpoints', 'epoch%s'%(epoch)))
            
    if verbose: print('')
    return training_losses, training_accs


def build_graph_and_train(Xtr, Ytr, save_dir, num_classes, arch=model.alexnet,
                          wd=0, gpu_id=0, seed=0, verbose=True, beta=1., **kwargs):
    """Build tensorflow graph and train"""
    
    tf.reset_default_graph()
    if seed is not None: 
        np.random.seed(seed)
        tf.set_random_seed(seed)

    if verbose: start = time.time()
    with tf.device("/gpu:%s"%(gpu_id)):
        if not os.path.exists(save_dir) or 'checkpoints' not in os.listdir(save_dir):
            graph = graph_builder_wrapper(num_classes, save_dir, arch=arch, wd=wd, beta=beta)
            tr_losses, tr_accs = train(Xtr, Ytr, graph, save_dir, **kwargs)
        else:
            graph = graph_builder_wrapper(num_classes, save_dir, arch=arch, wd=wd, update_collection='_')
            if verbose:
                print('Model already exists.. loading trained model..')
        Ytrhat = predict_labels(Xtr, graph, save_dir)
        train_acc = np.sum(Ytrhat == Ytr)/float(len(Ytr))

    if verbose:
        print('Train acc: %.2f (%.1f s elapsed)'%(train_acc, time.time()-start))

    return train_acc


def predict_labels_in_sess(X, graph, sess, batch_size=100):
    """Predict labels within a session"""
    labels = np.zeros(len(X))
    for i in range(0, len(X), batch_size):
        g_ = sess.run(graph['fc_out'], feed_dict = {graph['input_data']:X[i:i+batch_size], 
                                                    graph['delta']:np.zeros(np.shape(X[i:i+batch_size]))})
        labels[i:i+batch_size] = np.argmax(g_, 1)
    return labels


def latest_epoch(save_dir):
    """Grabs int corresponding to last epoch of weights saved in save_dir"""
    return max([int(f.split('epoch')[1].split('.')[0]) 
                for f in os.listdir(os.path.join(save_dir, 'checkpoints')) if 'epoch' in f])


def predict_labels(X, graph, save_dir, 
                   batch_size=100,
                   load_epoch=None,
                   load_weights_file=None):
    """Use trained model to predict"""
    
    # Load from checkpoint corresponding to latest epoch if none given
    if load_weights_file == None and load_epoch == None:
        load_epoch = latest_epoch(save_dir)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        if load_weights_file == None:
            graph['saver'].restore(sess, os.path.join(save_dir, 'checkpoints', 'epoch%s'%(load_epoch)))
        else:
            graph['saver'].restore(sess, load_weights_file)
        
        return predict_labels_in_sess(X, graph, sess, batch_size=batch_size)


def build_graph_and_predict(X, save_dir,
                            arch=model.alexnet,
                            Y=None,
                            num_classes=10,
                            gpu_id=0,
                            beta=1.,
                            load_epoch=None,
                            load_weights_file=None):
    """Build a tensorflow graph and predict labels"""
    
    tf.reset_default_graph()
    with tf.device("/gpu:%s"%(gpu_id)):
        graph = graph_builder_wrapper(num_classes, save_dir, arch=arch, beta=beta, update_collection='_')
        Yhat = predict_labels(X, graph, save_dir,
                              load_epoch=load_epoch,
                              load_weights_file=load_weights_file)
    if Y is None: 
        return Yhat
    return np.sum(Yhat == Y)/float(len(Y))
        

def recover_curve(X, Y, save_dir,
                  num_classes=10,
                  gpu_id=0,
                  verbose=True,
                  keyword='epoch'):
    """Evaluate performance on a dataset during training"""
    
    list_epochs = np.unique([int(f.split(keyword)[1].split('.')[0]) \
                             for f in os.listdir(os.path.join(save_dir, 'checkpoints')) if keyword in f])
    accs = np.zeros(len(list_epochs))
    load_weights_file = None
    if verbose: start = time.time()
    for i,epoch in enumerate(list_epochs):
        if keyword is not 'epoch': 
            load_weights_file= os.path.join(save_dir, 'checkpoints', '%s%s'%(keyword, epoch))
        accs[i] = build_graph_and_predict(X, save_dir,
                                          Y=Y,
                                          num_classes=num_classes,
                                          gpu_id=gpu_id,
                                          load_epoch=epoch,
                                          load_weights_file=load_weights_file)
        if verbose:
            print('\rRecovered accuracy for %s %s/%s: %.2f (%.2f s elapsed)'
                  %(keyword, i+1, len(list_epochs), accs[i], time.time()-start), end='')
    if verbose:
        print('')
    return accs


def recover_train_and_test_curves(Xtr, Ytr, Xtt, Ytt, save_dir,
                                  num_classes=10,
                                  gpu_id=0,
                                  verbose=True):
    """Recover training and test curves"""
    
    train_accs = recover_curve(Xtr, Ytr, save_dir,
                               num_classes=num_classes,
                               gpu_id=gpu_id,
                               verbose=verbose)
    test_accs = recover_curve(Xtt, Ytt, save_dir,
                              num_classes=num_classes,
                              gpu_id=gpu_id,
                              verbose=verbose)
    return train_accs,test_accs


def get_embedding_in_sess(X, graph, sess, batch_size=100):
    """Gets embedding (last layer output) within a session"""
    num_classes = graph['fc_out'].shape.as_list()[1]
    embedding = np.zeros((len(X), num_classes))
    for i in range(0, len(X), batch_size):
        embedding_ = sess.run(graph['fc_out'], feed_dict = {graph['input_data']:X[i:i+batch_size], 
                                                            graph['delta']:np.zeros(np.shape(X[i:i+batch_size]))})
        embedding[i:i+batch_size] = embedding_
    return embedding


def get_embedding(X, num_classes, save_dir, beta=1., batch_size=100, arch=model.alexnet, sn_fc=False):
    """recovers the representation of the data at the layer before the softmax layer
       Use sn_fc to indicate that last layer (should be named 'fc/weights:0') needs to be
         spectrally normalized.
    """
    
    tf.reset_default_graph()
    graph = graph_builder_wrapper(num_classes, save_dir, arch=arch, beta=beta, update_collection='_')
    load_epoch = latest_epoch(save_dir)

    if sn_fc:
        assert 'fc/weights:0' in [v.name for v in tf.global_variables()]
        W_fc_tensor = [v for v in tf.global_variables() if v.name == 'fc/weights:0'][0]
        b_fc_tensor = [v for v in tf.global_variables() if v.name == 'fc/bias:0'][0]

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        graph['saver'].restore(sess, os.path.join(save_dir, 'checkpoints', 'epoch%s'%(load_epoch)))

        # spectral normalization on last layer (fully connected)
        if sn_fc:
            W_fc, b_fc = sess.run([W_fc_tensor, b_fc_tensor])
            sigma = np.linalg.svd(W_fc.T)[1][0]
            sess.run([W_fc_tensor.assign(W_fc/sigma), b_fc_tensor.assign(b_fc/sigma)])

        return get_embedding_in_sess(X, graph, sess, batch_size=batch_size)

    
def check_weights_svs(num_classes, save_dir, arch=model.alexnet, n=2, load_epoch=None, beta=1.):    
    """Check singular value of all weights"""
    
    tf.reset_default_graph()
    graph = graph_builder_wrapper(num_classes, save_dir, arch=arch, update_collection='_', beta=beta)
    
    if load_epoch is None:
        load_epoch = latest_epoch(save_dir)
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        # Grab all weights
        graph['saver'].restore(sess, os.path.join(save_dir, 'checkpoints', 'epoch%s'%(load_epoch)))
        
        for tfvar in tf.get_collection('w_after_sn'):
            if 'weights' in tfvar.name:
                W = tfvar.eval(session=sess)
                print('%30s with shape %15s and top %s sv(s): %s' \
                      %(tfvar.name, np.shape(W), n, 
                        ', '.join(['%.2f'%(i) for i in np.linalg.svd(W.reshape(-1, np.shape(W)[-1]))[1][:n]])))
                

def print_total_number_of_trainable_params():
    """prints total number of trainable parameters according to default graph"""
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(total_parameters)

    
def extract_curve_tensorboard(tb_log_file, curve='loss'):
    """Given the name of a tensorboard event file, returns the desired curve"""
    
    values = []
    for e in tf.train.summary_iterator(tb_log_file):
        for v in e.summary.value:
            if v.tag == curve:
                values.append(v.simple_value)
    return np.array(values)


def extract_train_valid_tensorboard(save_dir, curve='accuracy', show_plot=False, only_final_value=False):
    """For a particular model, grab the tfevents training and validation curves"""

    # get train
    event_file = sorted(os.listdir(os.path.join(save_dir, 'train')))[0]
    tb_log_file = os.path.join(save_dir, 'train', event_file)
    train_values = extract_curve_tensorboard(tb_log_file, curve=curve)

    # get validation
    event_file = sorted(os.listdir(os.path.join(save_dir, 'validation')))[0]
    tb_log_file = os.path.join(save_dir, 'validation', event_file)
    valid_values = extract_curve_tensorboard(tb_log_file, curve=curve)

    if show_plot:
        plt.figure()
        plt.plot(train_values, label='training %s'%(curve))
        plt.plot(valid_values, label='validation %s'%(curve))
        plt.grid()
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel(curve)
        plt.show()
        
    if only_final_value:
        return train_values[-1], valid_values[-1]

    return train_values, valid_values


def plot_stacked_hist(v0, v1, labels=None):
    """Plots two histograms on top of one another"""
    if labels is None:
        labels = ['0', '1']
    bins = np.histogram(np.hstack((v0, v1)), bins=20)[1]
    data = [v0, v1]
    plt.hist(data, bins, label=labels, alpha=0.8, color=['r','g'],
             normed=True, edgecolor='none')
    plt.legend()


def get_margins(X, Y, save_dir, arch=model.alexnet, sn_fc=True):
    """Compute margins for X (margin = last layer difference between true label and 
       highest value that's not the true label)
    """
    
    num_classes = len(np.unique(Y))
    embeddings = get_embedding(X, num_classes, save_dir, arch=arch, sn_fc=sn_fc)
#    embeddings = np.exp(embeddings)
#    embeddings /= np.sum(embeddings, 1).reshape(-1, 1)
    margins = np.zeros(len(embeddings))
    
    print('Sanity check: accuracy is %.5f.'
          %(np.sum(np.argmax(embeddings, 1) == Y)/float(len(Y))))
    
    for i in range(len(embeddings)):
        if Y[i] == 0:
            margins[i] = np.max(embeddings[i][1:])
        elif Y[i] == len(embeddings[0])-1:
            margins[i] = np.max(embeddings[i][:-1])
        else:
            margins[i] = np.max([np.max(embeddings[i][:int(Y[i])]),
                                np.max(embeddings[i][int(Y[i])+1:])])
            
    return margins


def get_sn_weights(num_classes, save_dir, arch, print_svs=False):    
    """Grab all weights from spectrally normalized graph"""
    
    load_epoch = latest_epoch(save_dir)
    tf.reset_default_graph()
    graph = graph_builder_wrapper(num_classes, './temp/', arch=arch, update_collection='_')
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        graph['saver'].restore(sess, os.path.join(save_dir, 'checkpoints', 'epoch%s'%(load_epoch)))
        d = {v.name:sess.run(v) for v in tf.trainable_variables()}
        for v in tf.get_collection('w_after_sn'):
            key = v.name.split('_SN')[0]+':0'
            d[key] = sess.run(v)
            if print_svs:
                dim = d[key].shape[-1]
                print('%30s with shape %15s and top 2 sv(s): %s' \
                      %(key, np.shape(d[key]),
                        ', '.join(['%.2f'%(i) for i in np.linalg.svd(d[key].reshape(-1, dim))[1][:2]])))

    return d