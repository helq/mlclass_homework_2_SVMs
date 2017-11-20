#!/usr/bin/env python3
"""
Autoencoder to compress data

Author of original code: Aymeric Damien
Modified from: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
"""
#from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import math

# Entering data
from sklearn.model_selection import train_test_split

xs_all_ = np.array( [[float(xi) for xi in x.split()] for x in open("xtrain.txt").readlines()], dtype='int' )
xs_all = np.eye( 21 )[xs_all_].reshape( (-1,15*21) ) # converting each number into a one hot vector

test_size = 300

(xs_train, xs_test) = train_test_split(xs_all, test_size=test_size, random_state=123)

# Training Parameters
learning_rate = 0.01
num_steps = 1700*80
batch_size = 100

display_step = 5000

# batch subroutine
def next_batch_gen(xs, batch_size):
    n = xs.shape[0]
    n_batchs = math.ceil( n/batch_size )
    batch = -1

    def next_batch():
        nonlocal batch
        if n_batchs == batch:
            batch = 0
        else:
            batch += 1
        return xs[batch*batch_size : (batch+1)*batch_size]

    return next_batch

next_batch = next_batch_gen(xs_train, batch_size)

# Network Parameters
num_hidden_1 = 20 # 1st layer num features
num_hidden_2 = 10 # 2nd layer num features (the latent dim)
num_input = 15*21 # 15 features, each one as a hot vector of size 21

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    #'decoder_h2': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    #'decoder_h1': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    #'decoder_b2': tf.Variable(tf.random_normal([num_hidden_1])),
    #'decoder_b1': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    ## Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2
    #return layer_1

# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    #layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h2']), biases['decoder_b2']))
    # Decoder Hidden layer with sigmoid activation #2
    #layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h1']), biases['decoder_b1']))

    layers_1 = []
    for i in range(15):
        w_i = tf.Variable(tf.random_normal([num_hidden_2, 21]))
        b_i = tf.Variable(tf.random_normal([21]))
        #layer_1_i = tf.nn.softmax(tf.add(tf.matmul(layer_2, w_i), b_i))
        layer_1_i = tf.nn.softmax(tf.add(tf.matmul(x, w_i), b_i))
        layers_1.append( layer_1_i )
    #layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    return tf.concat( layers_1, 1 )

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x = next_batch()

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            test_loss = sess.run(loss, feed_dict={X: xs_test})
            print('Step {}: Minibatch Loss: {:f} Error in testing: {:.04g}'.format(i, l, test_loss))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    #n = 4
    #for i in range(n):
    #    # MNIST test set
    #    batch_x = next_batch()
    #    # Encode and decode the digit image
    #    g = sess.run(decoder_op, feed_dict={X: batch_x})

    #    print(batch_x[0])
    #    print(g[0])

    # Save the variables to disk.
    save_path = saver.save(sess, "autoencoder_processor/model.ckpt")
    print("Model saved in file: %s" % save_path)

#with tf.Session() as sess:
    # Restore variables from disk.
    #saver.restore(sess, "autoencoder_processor/model.ckpt")
    encoded_xs_all = sess.run(encoder_op, feed_dict={X: xs_all})

    #xs_all_nonlabeled_ = np.array( [[float(xi) for xi in x.split()] for x in open("xtrain.txt").readlines()], dtype='int' )
    #xs_all_nonlabeled  = np.eye( 21 )[xs_all_].reshape( (-1,15*21) ) # converting each number into a one hot vector

    #encoded_xs_all_nonlabeled = sess.run(encoder_op, feed_dict={X: xs_all_nonlabeled})

# saving transformed datapoints
with open("xtrain_encoded.txt", "w") as f:
    for i in range( encoded_xs_all.shape[0] ):
        for feat in encoded_xs_all[i]:
            f.write("{:10e} ".format(feat))
        f.write("\n")
