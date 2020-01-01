from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time
import gzip

import six
from six.moves import cPickle as pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import zhusuan as zs

from examples import conf
from examples.utils import save_image_collections

# tf.autograph.set_verbosity(2)


def load_mnist_realval(path):
    f = gzip.open(path, 'rb')
    if six.PY2:
        train_set, valid_set, test_set = pickle.load(f)
    else:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    x_train, t_train = train_set[0], train_set[1]
    x_valid, t_valid = valid_set[0], valid_set[1]
    x_test, t_test = test_set[0], test_set[1]
    n_y = t_train.max() + 1
    return x_train, t_train, x_valid, t_valid, x_test, t_test



@zs.meta_bayesian_net(scope="gen", reuse_variables=True)
def build_gen(y, x_dim, z_dim, y_dim, n):
    bn = zs.BayesianNet()
    # z ~ N(0,I)
    z_mean = tf.zeros([n, z_dim])
    z = bn.normal("z", z_mean, std=1., group_ndims=1)
    
    # Concatenate z and y
    z = tf.concat(axis=1, values=[z,y])

    # x_logits = f_NN(z)
    h = tf.layers.dense(z, 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    x_logits = tf.layers.dense(h, x_dim)

    x_mean = bn.deterministic("x_mean", tf.sigmoid(x_logits))

    # x ~ Bernoulli(x_logits)
    bn.bernoulli("x", x_logits, group_ndims=1, dtype=tf.float32)
    return bn


@zs.reuse_variables(scope="q_net")
def build_q_net(x, y, z_dim, y_dim):
    bn = zs.BayesianNet()
    # concatenate x and y
    x = tf.concat(axis=1, values=[x,y])
    h = tf.layers.dense(x, 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    z_mean = tf.layers.dense(h, z_dim)
    z_logstd = tf.layers.dense(h, z_dim)
    bn.normal("z", z_mean, logstd=z_logstd, group_ndims=1)
    return bn


def main():
    # Load MNIST
    data_path = os.path.join(conf.data_dir, "mnist.pkl.gz")
    x_train, t_train, x_valid, t_valid, x_test, t_test = load_mnist_realval(data_path)


    x_train = np.random.binomial(1, x_train, size=x_train.shape).astype(np.float32)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype(np.float32)
    x_dim = x_train.shape[1]    # x_train = np.vstack([x_train, x_valid])

    # Binarize input
    y_train = to_categorical(np.array(t_train))
    y_test = to_categorical(np.array(t_test))
    y_dim = y_train.shape[1]
    y_pic = to_categorical(np.arange(10))

    # Define model parameters
    z_dim = 40

    # Build the computation graph
    # n_particles = tf.placeholder(tf.int32, shape=[], name="n_particles")
    x = tf.placeholder(tf.float32, shape=[None, x_dim], name="x")
    y = tf.placeholder(tf.float32, shape=[None, y_dim], name="y")
    n = tf.placeholder(tf.int32, shape=[], name="n")

    model = build_gen(y, x_dim, z_dim, y_dim, n)
    variational = build_q_net(x, y, z_dim, y_dim)

    lower_bound = zs.variational.elbo(model, {"x": x}, variational=variational)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    infer_op = optimizer.minimize(cost)

    # is_log_likelihood = tf.reduce_mean(
    #     zs.is_loglikelihood(model, {"x": x}, proposal=variational, axis=0))

    # Random generation
    x_gen = tf.reshape(model.observe()["x_mean"], [-1, 28, 28, 1])

    epochs = 100
    batch_size = 128
    iters = x_train.shape[0] // batch_size

    # Run the Inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            time_epoch = -time.time()
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run([infer_op, lower_bound],
                                 feed_dict={x: x_batch,
                                            y: y_batch,
                                            n: batch_size})
                lbs.append(lb)
            time_epoch += time.time()
            print("Epoch {} ({:.1f}s): Lower bound = {}".format(epoch, time_epoch, np.mean(lbs)))
            
            images = sess.run(x_gen, feed_dict={y: y_pic, n: 10})
            name = os.path.join("results", "vae.epoch.{}.png".format(epoch))
            save_image_collections(images, name, shape=(1, 10))

if __name__ == "__main__":
    main()
