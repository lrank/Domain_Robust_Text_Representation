import tensorflow as tf
import numpy as np


class TextCNN(object):

    def cnn(self, scope_number, embedded_chars_expanded, sequence_length, embedding_size, filter_sizes, num_filters):
        with tf.variable_scope("cnn%s" % scope_number):
        # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.get_variable(
                        name="W",
                        shape=filter_shape,
                        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
                        )
                    b = tf.get_variable(
                        name="b",
                        shape=[num_filters],
                        initializer=tf.constant_initializer(0.1)
                        )
                    conv = tf.nn.conv2d(
                        embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            pooled = tf.concat(pooled_outputs, 3)
            return tf.reshape(pooled, [-1, num_filters * len(filter_sizes)])

        
    def wx_plus_b(self, scope_name, x, size):
        with tf.variable_scope("full_connect_%s" % scope_name) as scope:
            W = tf.get_variable(
                name="W",
                shape=size,
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(
                name="b",
                shape=[size[1]],
                initializer=tf.constant_initializer(0.1, )
                )
            y = tf.nn.xw_plus_b(x, W, b, name="hidden")
            return y

    #main enter
    def __init__(self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, num_domains,
            l2_reg_lambda):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name = "input_y")
        self.input_d = tf.placeholder(tf.int32, [None, num_domains], name = "input_d")

        l2_loss = tf.constant(0.0)

        with tf.variable_scope("embedding"):
            self.emb_W = tf.get_variable(
                name="lookup_emb",
                shape=[vocab_size, embedding_size],
                initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
                trainable=False
                )
            embedded_chars = tf.nn.embedding_lookup(self.emb_W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        
        #cnn+pooling
        num_filters_total = num_filters * len(filter_sizes)
        #shared
        self.pub_h_pool = self.cnn("shared-public", self.embedded_chars_expanded, sequence_length, embedding_size, filter_sizes, num_filters)
        #private
        self.pri_h_pool = self.cnn("shared-private", self.embedded_chars_expanded, sequence_length, embedding_size, filter_sizes, num_filters)

        #final representation
        self.h_pool = tf.concat([self.pri_h_pool, self.pub_h_pool], axis=1)
        input_dim = num_filters_total * 2

        # Add dropout
        with tf.name_scope("dropout"):
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.h_drop = tf.nn.dropout(self.h_pool, self.dropout_keep_prob)
            self.pub_h_drop = tf.nn.dropout(self.pub_h_pool, self.dropout_keep_prob)
            self.pri_h_drop = tf.nn.dropout(self.pri_h_pool, self.dropout_keep_prob)

        hidden_size = 300
        with tf.variable_scope("label"):
            h1 = self.wx_plus_b(
                scope_name="h1",
                x=self.h_drop,
                size=[input_dim, hidden_size]
            )
            self.y_scores = self.wx_plus_b(
                scope_name='score',
                x = h1,
                size=[hidden_size, num_classes]
                )
            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.y_scores,
                    labels=self.input_y,
                    )
                self.y_loss = tf.reduce_mean(losses, name="task_loss")

            with tf.name_scope("accuracy"):
                self.y_pred = tf.argmax(self.y_scores, 1, name="predictions")
                cor_pred = tf.cast(
                    tf.equal( self.y_pred, tf.argmax(self.input_y, 1) ),
                    "float"
                    )
                self.y_accuracy = tf.reduce_mean( cor_pred, name="accuracy" )
        
        
        with tf.variable_scope("domain"):
            h1 = self.wx_plus_b(
                scope_name="h1",
                x=self.pub_h_drop,
                size=[num_filters_total, hidden_size]
            )
            self.domain_scores = self.wx_plus_b(
                scope_name="score",
                x=h1,
                size=[hidden_size, num_domains]
                )
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.domain_scores,
                    labels=self.input_d,
                    )
                self.domain_loss = tf.reduce_mean(losses)
            with tf.name_scope("accuracy"):
                self.domain_pred = tf.argmax(self.domain_scores, 1, name="predictions")
                cor_pred = tf.cast(
                    tf.equal(self.domain_pred, tf.argmax(self.input_d, 1) ),
                    "float"
                    )
                self.domain_accuracy = tf.reduce_mean(cor_pred, name="acc")


        with tf.variable_scope("gen"):
            h1 = self.wx_plus_b(
                scope_name="h1",
                x=self.pri_h_drop,
                size=[num_filters_total, hidden_size]
            )
            self.gen_scores = self.wx_plus_b(
                scope_name="score",
                x=h1,
                size=[hidden_size, num_domains]
                )
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.gen_scores,
                    labels=self.input_d,
                    )
                self.gen_loss = tf.reduce_mean(losses)
            with tf.name_scope("accuracy"):
                self.gen_pred = tf.argmax(self.gen_scores, 1, name="predictions")
                cor_pred = tf.cast(
                    tf.equal(self.gen_pred, tf.argmax(self.input_d, 1) ),
                    "float"
                    )
                self.gen_accuracy = tf.reduce_mean(cor_pred, name="acc")
