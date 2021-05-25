import numpy as np
import tensorflow as tf
import tflearn


class Actor(object):
    def CreateNetwork(self, inputs, n_actions):
         with tf.variable_scope('Actor_1'):
            split0 = tflearn.conv_1d(inputs[:, 0:1, :], 5, 1)
            split1 = tflearn.conv_1d(inputs[:, 1:2, :], 16, 1)
            a = split0[:, 0, :]
            b = split1[:, 0, :]
            c = inputs[:, 2, -1:]
            d = inputs[:, 3, -1:]
            e = inputs[:, 4, -1:]
            f = inputs[:, 5, -1:]
            g = inputs[:, 6, -1:]
            hidden0 = tf.concat([a, b, c, d, e, f, g], axis=1)

            l1 = tf.layers.dense(
                inputs=hidden0,
                units=128,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )
            l2 = tf.layers.dense(
                inputs=l1,
                units=128,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l2'
            )

            acts_prob = tf.layers.dense(
                inputs=l2,
                units=n_actions,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='acts_prob'
            )
            return acts_prob
         
    
    def r(self, pi_new, pi_old, acts):
        return tf.reduce_sum(tf.multiply(pi_new, acts), reduction_indices=1, keepdims=True) / \
                tf.reduce_sum(tf.multiply(pi_old, acts), reduction_indices=1, keepdims=True)
                
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features[0], n_features[1]], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.acts = tf.placeholder(tf.float32, [None, n_actions], "acts")
        self.old_pi = tf.placeholder(tf.float32, [None, n_actions], "old_pi")
        self.new_pi = tf.placeholder(tf.float32, [None, n_actions], "new_pi")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")
        self.lr = tf.placeholder(tf.float32, None, 'lr_ph')
        
        self.pi = self.CreateNetwork(inputs=self.s, n_actions=n_actions)
        #self.real_out = tf.clip_by_value(self.pi, 1e-4, 1. - 1e-4)
        self.real_out = self.pi
        self.ppo2loss = tf.minimum(self.r(self.real_out, self.old_pi, self.acts) * self.td_error, 
                            tf.clip_by_value(self.r(self.real_out, self.old_pi, self.acts), 1 - 0.2, 1 + 0.2) * self.td_error
                        )
        #self.ppo2loss = self.r(self.real_out, self.old_pi, self.acts) * self.td_error
        self.exp_v = - tf.reduce_sum(self.ppo2loss)
  
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.exp_v)  # minimize(-exp_v) = maximize(exp_v)
                
    def learn(self, s, a, p, td, lr):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.acts: a, self.old_pi: p, self.td_error: td, self.lr:lr}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)

        return exp_v

    def choose_action(self, s, eps=None):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.real_out, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel()), probs


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features[0], n_features[1]], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        self.lr = tf.placeholder(tf.float32, None, 'lr_ph')
        self.end = tf.placeholder(tf.bool, None, 'end_video')
        end_new = tf.cast(self.end, tf.float32)

        with tf.variable_scope('Critic'):
            split0 = tflearn.conv_1d(self.s[:, 0:1, :], 5, 1)
            split1 = tflearn.conv_1d(self.s[:, 1:2, :], 16, 1)
            a = split0[:, 0, :]
            b = split1[:, 0, :]
            c = self.s[:, 2, -1:]
            d = self.s[:, 3, -1:]
            e = self.s[:, 4, -1:]
            f = self.s[:, 5, -1:]
            g = self.s[:, 6, -1:]
            hidden0 = tf.concat([a, b, c, d, e, f, g], axis=1)

            l1 = tf.layers.dense(
                inputs=hidden0,
                units=128,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )
            l2 = tf.layers.dense(
                inputs=l1,
                units=128,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l2'
            )

            self.v = tf.layers.dense(
                inputs=l2,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            discount = 0.99 * 0.95

            self.td_error = discount * (self.r + 0.99 * tf.math.multiply(self.v_, (1 - end_new)) - self.v)
            self.loss = 0.5 * tf.reduce_mean(tf.square(self.td_error))    # TD_error = (r+gamma*V_next) - V_eval
        
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def learn(self, s, r, s_, end, lr):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r, self.lr: lr,
                                          self.end:end})
        return td_error


class LActor(object):
    def CreateNetwork(self, inputs, n_actions):
         with tf.variable_scope('LActor_1'):
            split0 = tflearn.conv_1d(inputs[:, 0:1, :], 5, 1)
            split1 = tflearn.conv_1d(inputs[:, 1:2, :], 16, 1)
            a = split0[:, 0, :]
            b = split1[:, 0, :]
            c = inputs[:, 2, -1:]
            d = inputs[:, 3, -1:]
            e = inputs[:, 4, -1:]
            f = inputs[:, 5, -1:]
            g = inputs[:, 6, -1:]
            hidden0 = tf.concat([a, b, c, d, e, f, g], axis=1)

            l1 = tf.layers.dense(
                inputs=hidden0,
                units=128,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )
            l2 = tf.layers.dense(
                inputs=l1,
                units=128,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l2'
            )

            acts_prob = tf.layers.dense(
                inputs=l2,
                units=n_actions,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='acts_prob'
            )
            return acts_prob
    
    def r(self, pi_new, pi_old, acts):
        return tf.reduce_sum(tf.multiply(pi_new, acts), reduction_indices=1, keepdims=True) / \
                tf.reduce_sum(tf.multiply(pi_old, acts), reduction_indices=1, keepdims=True)
    
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features[0], n_features[1]], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.acts = tf.placeholder(tf.float32, [None, n_actions], "acts")
        self.old_pi = tf.placeholder(tf.float32, [None, n_actions], "old_pi")
        self.new_pi = tf.placeholder(tf.float32, [None, n_actions], "new_pi")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")
        self.lr = tf.placeholder(tf.float32, None, 'lr_ph')
        
        self.pi = self.CreateNetwork(inputs=self.s, n_actions=n_actions)
        #self.real_out = tf.clip_by_value(self.pi, 1e-4, 1. - 1e-4)
        self.real_out = self.pi
        self.ppo2loss = tf.minimum(self.r(self.real_out, self.old_pi, self.acts) * self.td_error, 
                            tf.clip_by_value(self.r(self.real_out, self.old_pi, self.acts), 1 - 0.2, 1 + 0.2) * self.td_error
                        )
        self.exp_v = - tf.reduce_sum(self.ppo2loss)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, p, td, lr):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.acts: a, self.old_pi: p, self.td_error: td, self.lr:lr}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)

        return exp_v

    def choose_action(self, s, eps=None):
        s = s[np.newaxis, :]  # single state
        probs = self.sess.run(self.real_out, {self.s: s})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel()), probs

