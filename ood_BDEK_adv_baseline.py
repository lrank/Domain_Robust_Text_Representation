#! /usr/bin/env python

import os
import time
import datetime
# import cPickle

import tensorflow as tf
import numpy as np

import data_helpers
from text_cnn_baseline import TextCNN

from tensorflow.contrib import learn

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("adv_lambda", 1e-3, "Robust Regularizaion lambda (default: 1e-3)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate alpha")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_train_epochs", 50, "Number of training epochs")
tf.flags.DEFINE_integer("num_tune_epochs", 50, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

print(tf.__version__)
FLAGS = tf.flags.FLAGS
FLAGS.batch_size
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Load data
print("Loading data...")
sent_length, vocab_size, num_label, num_domain, x, y, d = data_helpers.load_data()
# Randomly shuffle data
# np.random.seed(101)

score_sum = []
best_score = 0

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        intra_op_parallelism_threads=2,
        inter_op_parallelism_threads=4)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length = sent_length,
            num_classes = num_label,
            vocab_size = vocab_size,
            embedding_size = FLAGS.embedding_dim,
            filter_sizes = map(int, FLAGS.filter_sizes.split(",")),
            num_filters = FLAGS.num_filters,
            num_domains = num_domain,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            )


        # Define Training procedure
        learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        adv_lambda = tf.placeholder(tf.float32, shape=[], name="adversarial_lambda")
        
        global_step = tf.Variable(0, name="global_step", trainable=False)
        all_var_list = tf.trainable_variables()

        optimizer_n = tf.train.AdamOptimizer(
            learning_rate = learning_rate
            ).minimize(
                cnn.y_loss,
                global_step=global_step
                )

        var_d = [var for var in all_var_list if 'domain' in var.name or 'gen' in var.name]
        assert( len(var_d) == 4)
        optimizer_d = tf.train.AdamOptimizer(
            learning_rate = learning_rate
            ).minimize(
                adv_lambda * cnn.domain_loss,
                var_list=var_d
            )

        var_g = [var for var in all_var_list if var not in var_d]
        optimizer_g = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                cnn.y_loss - adv_lambda * cnn.domain_loss,
                var_list=var_g,
                global_step = global_step
                )


        def train_batch(x_batch, y_batch, d_batch, opt, adv_lbd, lr):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.input_d: d_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                adv_lambda: adv_lbd,
                learning_rate: lr,
            }
            _, step, loss, accuracy, d_l, d_a = sess.run(
                [opt, global_step, cnn.y_loss, cnn.y_accuracy, cnn.domain_loss, cnn.domain_accuracy],
                feed_dict)

        def dev_batch(x_batch, y_batch, d_batch):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.input_d: d_batch,
                cnn.dropout_keep_prob: 1.0,
                adv_lambda: 0,
            }
            step, loss, accuracy, d_l, d_a = sess.run(
                [global_step, cnn.y_loss, cnn.y_accuracy, cnn.domain_loss, cnn.domain_accuracy],
                feed_dict)
            return accuracy
        
        def dev_step(x_dev, y_dev, d_dev):
            cor = 0.
            step = 512
            for ind in range(0, len(x_dev), step):
                num_ins = min(len(x_dev) - ind, step)
                acc = dev_batch(
                    x_batch = x_dev[ind: ind + num_ins],
                    y_batch = y_dev[ind: ind + num_ins],
                    d_batch = d_dev[ind: ind + num_ins]
                    )
                cor = cor + num_ins * acc
            acc = cor / len( x_dev )
            return acc


        #data_split
        cv_iter = data_helpers.cross_validation_iter(
            data=[x, y, d], 
        )
        best_scores_pre = []
        best_scores = []

        for _ in range(1):
            x_train, y_train, d_train,\
            x_test_all, y_test_all, d_test_all = cv_iter.fetch_next()
            print("split train {} / dev {}".format(len(x_train), len(x_test_all)))
            x_test = [ [], [], [], [] ]
            y_test = [ [], [], [], [] ]
            d_test = [ [], [], [], [] ]
            for i in range( len(x_test_all) ):
                dom = np.argmax( d_test_all[i] )
                x_test[dom].append( x_test_all[i] )
                y_test[dom].append( y_test_all[i] )
                d_test[dom].append( d_test_all[i] )
            x_test = np.array( x_test )
            y_test = np.array( y_test )
            d_test = np.array( d_test )

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            # sess.run(cnn.W.assign(w2v))
            best_score_pre = np.zeros( (4) )
            best_score_cv = np.zeros( (4) )
            data_size = len(x_train)

            # Generate batches
            train_batch_iter = data_helpers.batch_iter(
                data = [x_train, y_train, d_train],
                batch_size = FLAGS.batch_size)
            
            # pre-train
            for _ in range( FLAGS.num_train_epochs * data_size / FLAGS.batch_size):
                x_batch, y_batch, d_batch = train_batch_iter.next_full_batch()

                train_batch( x_batch, y_batch, d_batch, opt=optimizer_n, adv_lbd=FLAGS.adv_lambda, lr=FLAGS.learning_rate )

                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    for dom in range( 4 ):
                        acc = dev_step( x_test[dom], y_test[dom], d_test[dom] )
                        if acc > best_score_pre[dom]:
                            best_score_pre[dom] = acc
            
            best_scores_pre.append(best_score_pre)

            # Training loop. For each batch...
            for _ in range( FLAGS.num_tune_epochs * data_size / FLAGS.batch_size ):
                x_batch, y_batch, d_batch = train_batch_iter.next_full_batch()

                train_batch( x_batch, y_batch, d_batch,
                    opt=optimizer_d, adv_lbd=FLAGS.adv_lambda, lr=FLAGS.learning_rate )
                train_batch( x_batch, y_batch, d_batch,
                    opt=optimizer_g, adv_lbd=FLAGS.adv_lambda, lr=FLAGS.learning_rate )

                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    for dom in range( 4 ):
                        acc = dev_step( x_test[dom], y_test[dom], d_test[dom] )
                        if acc > best_score_cv[dom]:
                            best_score_cv[dom] = acc
            
            best_scores.append(best_score_cv)
            print("best phase 1 score {}".format(best_score_pre))
            print("best phase 2 score {}".format(best_score_cv))

print( best_scores_pre)
print( np.average( best_scores_pre) )

print( best_scores )
print( np.average(best_scores) )
