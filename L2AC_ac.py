import numpy as np
import tensorflow as tf
import tflearn


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features[0], n_features[1]], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")
        self.lr = tf.placeholder(tf.float32, None, 'lr_ph')

        with tf.variable_scope('Actor_1'):
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

            self.acts_prob = tf.layers.dense(
                inputs=l2,
                units=n_actions,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td, lr):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td, self.lr:lr}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s, eps):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features[0], n_features[1]], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        self.lr = tf.placeholder(tf.float32, None, 'lr_ph')

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
            self.td_error = self.r + 0.9 * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def learn(self, s, r, s_, lr):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r, self.lr: lr})
        return td_error


class LActor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features[0], n_features[1]], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        self.lr = tf.placeholder(tf.float32, None, 'lr_ph')

        with tf.variable_scope('LActor_1'):
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
                units=256,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )
            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td, lr):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td, self.lr: lr}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s, eps):
        s = s[np.newaxis, :]  # single state
        probs = self.sess.run(self.acts_prob, {self.s: s})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())
    
    
class LActorCritic(object):
    def __init__(self, sess, n_features, n_actions_a, n_actions_la, lr_a=0.001, lr_la=0.001, lr_c=0.01):
        self.sess = sess

        
        self.s = tf.placeholder(tf.float32, [1, n_features[0], n_features[1]], "state")

        # actor
        self.a_a = tf.placeholder(tf.int32, None, "act_a")
        self.td_error_a = tf.placeholder(tf.float32, None, "td_error_a")
        self.lr_a = tf.placeholder(tf.float32, None, 'lr_ph_a')

        # critic
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        self.lr_c = tf.placeholder(tf.float32, None, 'lr_ph_c')

        # Lactor
        self.a_la = tf.placeholder(tf.int32, None, "act_la")
        self.td_error_la = tf.placeholder(tf.float32, None, "td_error_la")
        self.lr_la = tf.placeholder(tf.float32, None, 'lr_ph_la')

        with tf.variable_scope('Actor_1'):
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

            l3 = tf.layers.dense(
                inputs=hidden0,
                units=256,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l3'
            )

            self.acts_prob_a = tf.layers.dense(
                inputs=l2,
                units=n_actions_a,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='acts_prob_a'
            )

            self.acts_prob_la = tf.layers.dense(
                inputs=l3,
                units=n_actions_la,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='acts_prob_la'
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
            self.td_error = self.r + 0.9 * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval

        with tf.variable_scope('exp_v_a'):
            log_prob_a = tf.log(self.acts_prob_a[0, self.a_a])
            self.exp_v_a = tf.reduce_mean(log_prob_a * self.td_error_a)  # advantage (TD_error) guided loss

        with tf.variable_scope('exp_v_la):
            log_prob_la = tf.log(self.acts_prob_la[0, self.a_la])
            self.exp_v_la = tf.reduce_mean(log_prob_la * self.td_error_la)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op_a = tf.train.AdamOptimizer(self.lr_a).minimize(-self.exp_v_a)  # minimize(-exp_v) = maximize(exp_v)
            self.train_op_la = tf.train.AdamOptimizer(self.lr_a).minimize(-self.exp_v_la)
            self.train_op_c = tf.train.AdamOptimizer(self.lr_c).minimize(self.loss)                   
                               
    def learn(self, s, a_a, a_la, lr_a, r, s_, lr_c):
        s, s_ = s[np.newaxis, :], s_[np.newaxis,:]
        
        v_ = self.sess.run(self.v, {self.s: s_})
        td, _ = self.sess.run([self.td_error, self.train_op_c], {self.s: s, self.v_: v_, self.r: r, self.lr_a: lr_c})
        
        feed_dict_a = {self.s: s, self.a_a: a_a, self.td_error_a: td, self.lr_a:lr_a}
        _, exp_v_a = self.sess.run([self.train_op_a, self.exp_v_a], feed_dict_a)

        feed_dict_la = {self.s: s, self.a_la: a_la, self.td_error_la: td, self.lr_a:lr_a}
        _, exp_v_la = self.sess.run([self.train_op_la, self.exp_v_la], feed_dict_la)
        
        return exp_v_a, exp_v_la

    def choose_action(self, s, eps):
        s = s[np.newaxis, :]
        probs_a = self.sess.run(self.acts_prob_a, {self.s: s})   # get probabilities for all actions
        probs_la = self.sess.run(self.acts_prob_la, {self.s: s})

        result_a = np.random.choice(np.arange(probs_a.shape[1]), p=probs_a.ravel())
        result_la = np.random.choice(np.arange(probs_la.shape[1]), p=probs_la.ravel())
        return result_a, result_la

