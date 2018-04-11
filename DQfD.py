# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np
import random
import functools
from Memory import Memory
from ops import linear, conv2d
from functools import reduce
def lazy_property(func):
    attribute = '_lazy_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return wrapper


def updateNet( from_name, to_name):
    to_vars = tf.get_collection(to_name)
    from_vars = tf.get_collection(from_name)
    print("####" + str(to_vars))
    print("###"+ str(from_vars))
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

class DQfD:
    def __init__(self, name, env, config, sess, replay_memory, demo_memory=None, demo_transitions=None):
        self.sess = sess
        self.config = config
        self.name = name
        # replay_memory stores both demo data and generated data, while demo_memory only store demo data
        #Demo Data Part
        #self.replay_memory = Memory(capacity=self.config.replay_buffer_size, permanent_data=len(demo_transitions))
        #self.demo_memory = Memory(capacity=self.config.demo_buffer_size, permanent_data=self.config.demo_buffer_size)
        self.replay_memory = replay_memory
        #self.demo_memory = Memory(capacity=self.config.demo_buffer_size)

        #No Data Data
        #self.add_demo_to_memory(demo_transitions=demo_transitions)  # add demo data to both demo_memory & replay_memory

        self.time_step = 0
        self.epsilon = self.config.INITIAL_EPSILON

        self.action_dim = env.action_space.n

        self.action_batch = tf.placeholder("int32", [None])
        self.y_input = tf.placeholder("float", [None, self.action_dim])
        self.ISWeights = tf.placeholder("float", [None, 1])
        self.n_step_y_input = tf.placeholder("float", [None, self.action_dim])  # for n-step reward
        self.isdemo = tf.placeholder("float", [None])
        self.eval_input = tf.placeholder("float", [None, 84,84,3])
        self.select_input = tf.placeholder("float", [None, 84,84,3])

        self.Q_eval
        self.Q_select

        self.loss
        self.optimize
        self.update_target_net
        self.abs_errors

        self.update_local_ops

        self.sess.run(tf.global_variables_initializer())
        if(self.name  == "learner"):
            with tf.device("/device:CPU:0"):
                self.saver = tf.train.Saver()
                if(self.config.START_STEP == 0):
                    self.save_model()
                self.restore_model()

        self.dueling = False
        self.cnn_format = 'NHWC'
        self.demo_num = 0
        self.sum_abs_error = 0
        self.sum_age = 0
        self.qvalue = [0.0] * self.action_dim
        self.oldsum = 0
    def add_demo_to_memory(self, demo_transitions):
        # add demo data to both demo_memory & replay_memory
        for t in demo_transitions:
            self.demo_memory.store(np.array(t, dtype=object))
            self.replay_memory.store(np.array(t, dtype=object))
            assert len(t) == 10

    # use the expert-demo-data to pretrain
    def pre_train(self):
        print('Pre-training ...')
        for i in range(self.config.PRETRAIN_STEPS):
            self.train_Q_network(pre_train=True)
            if i % 200 == 0 and i > 0:
                print('{} th step of pre-train finish ...'.format(i))
        self.time_step = 0
        print('All pre-train finish.')

    # TODO: How to add the variable created in tf.layers.dense to the customed collection？
    # def build_layers(self, state, collections, units_1, units_2, w_i, b_i, regularizer=None):
    #     with tf.variable_scope('dese1'):
    #         dense1 = tf.layers.dense(tf.contrib.layers.flatten(state), activation=tf.nn.relu, units=units_1,
    #                                  kernel_initializer=w_i, bias_initializer=b_i,
    #                                  kernel_regularizer=regularizer, bias_regularizer=regularizer)
    #     with tf.variable_scope('dens2'):
    #         dense2 = tf.layers.dense(dense1, activation=tf.nn.relu, units=units_2,
    #                                  kernel_initializer=w_i, bias_initializer=b_i,
    #                                  kernel_regularizer=regularizer, bias_regularizer=regularizer)
    #     with tf.variable_scope('dene3'):
    #         dense3 = tf.layers.dense(dense2, activation=tf.nn.relu, units=self.action_dim,
    #                                  kernel_initializer=w_i, bias_initializer=b_i,
    #                                  kernel_regularizer=regularizer, bias_regularizer=regularizer)
    #     return dense3

    def build_layers(self, states, cnames, initializer, activation_fn, reg=None):
        self.w = {}
        self.t_w = {}
        self.dueling = True
        a_d = self.action_dim

        self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(states,
                                                     32, [8, 8], [4, 4], cnames,initializer, activation_fn,
                                                         'NHWC', name='l1')
        self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
                                                     64, [4, 4], [2, 2],  cnames, initializer, activation_fn,
                                                         'NHWC', name='l2')
        self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
                                                     64, [3, 3], [1, 1], cnames, initializer, activation_fn,
                                                         'NHWC', name='l3')

        shape = self.l3.get_shape().as_list()
        self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

        if self.dueling:
            self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = \
                linear(self.l3_flat, 512, cnames, activation_fn=activation_fn, name='value_hid')

            self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = \
                linear(self.l3_flat, 512, cnames, activation_fn=activation_fn, name='adv_hid')

            self.value, self.w['val_w_out'], self.w['val_w_b'] = \
                linear(self.value_hid, 1, cnames, name='value_out')

            self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
                linear(self.adv_hid, self.action_dim, cnames, name='adv_out')

            # Average Dueling
            self.q = self.value +  (self.advantage -  tf.reduce_mean(self.advantage, axis=1, keepdims=True))

        else:
            self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, cnames, activation_fn=activation_fn, name='l4')

            self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.action_dim, cnames, name='q')
        return self.q
    @lazy_property
    def update_local_ops(self):
        to_vars = tf.get_collection('learner_select_net_params',)
        from_vars = tf.get_collection(self.name + '_select_net_params')
        print("update_local_ops" + str(to_vars))
        return [tf.assign(e, s) for e, s in zip(from_vars, to_vars)]
    @lazy_property
    def Q_select(self):
        with tf.variable_scope(self.name+'_select_net') as scope:
            c_names = [self.name+'_select_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            initializer = tf.truncated_normal_initializer(0, 0.02)
            activation_fn = tf.nn.relu
            reg = tf.contrib.layers.l2_regularizer(scale=0.2)  # Note: only parameters in select-net need L2
            return self.build_layers(self.select_input, c_names, initializer, activation_fn, reg)

    @lazy_property
    def Q_eval(self):
        with tf.variable_scope(self.name+'_eval_net') as scope:
            c_names = [self.name+'_eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            initializer = tf.truncated_normal_initializer(0, 0.02)
            activation_fn = tf.nn.relu
            return self.build_layers(self.eval_input, c_names, initializer, activation_fn)

    def loss_l(self, ae, a):
        return 0.0 if ae == a else 0.8

    def loss_jeq(self, Q_select):
        jeq = 0.0
        for i in range(self.config.BATCH_SIZE):
            ae = self.action_batch[i]
            max_value = float("-inf")
            for a in range(self.action_dim):
                max_value = tf.maximum(Q_select[i][a] + self.loss_l(ae, a), max_value)
            jeq += self.isdemo[i] * (max_value - Q_select[i][ae])
        return jeq

    @lazy_property
    def loss(self):
        l_dq = tf.reduce_mean(tf.squared_difference(self.Q_select, self.y_input))
        l_n_dq = tf.reduce_mean(tf.squared_difference(self.Q_select, self.n_step_y_input))
        # l_n_step_dq = self.loss_n_step_dq(self.Q_select, self.n_step_y_input)
        l_jeq = self.loss_jeq(self.Q_select)
        l_l2 = tf.reduce_sum([tf.reduce_mean(reg_l) for reg_l in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)])
        return self.ISWeights * tf.reduce_sum([l * λ for l, λ in zip([l_dq, l_n_dq, l_jeq, l_l2], self.config.LAMBDA)])

    @lazy_property
    def abs_errors(self):
        return tf.reduce_sum(tf.abs(self.y_input - self.Q_select), axis=1)  # only use 1-step R to compute abs_errors

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.config.LEARNING_RATE)
        return optimizer.minimize(self.loss)  # only parameters in select-net is optimized here

    @lazy_property
    def update_target_net(self):
        select_params = tf.get_collection(self.name +'_select_net_params')
        eval_params = tf.get_collection(self.name +'_eval_net_params')
        print("update_eval_ops" + str(eval_params))
        return [tf.assign(e, s) for e, s in zip(eval_params, select_params)]

    def save_model(self):
        if(self.name == 'learner'):
            print("Model saved in : {}".format(self.saver.save(self.sess, self.config.MODEL_PATH,  global_step = self.time_step+self.config.START_STEP)))

    def restore_model(self):
        if(self.name ==  'learner'):
            self.saver.restore(self.sess, self.config.MODEL_PATH+ "-"+str(self.time_step+self.config.START_STEP))
            print("Model restored.")

    def perceive(self, transition, current_ts):
        state_batch = np.expand_dims(transition[0], axis=0)
        action_batch = np.expand_dims(transition[1], axis=0)
        reward_batch = np.expand_dims(transition[2], axis=0)
        next_state_batch  = np.expand_dims(transition[3], axis=0)
        done_batch = np.expand_dims(transition[4], axis=0)
        demo_data = np.expand_dims(transition[5], axis=0)
        y_batch = np.zeros((1, self.action_dim))

        Q_select = self.sess.run(self.Q_select, feed_dict={self.select_input: next_state_batch})
        Q_eval = self.sess.run(self.Q_eval, feed_dict={self.eval_input: next_state_batch})
        #print(Q_select.shape)
        temp = self.sess.run(self.Q_select, feed_dict={self.select_input: state_batch[0].reshape([-1, 84, 84, 3])})[0]
        # add 1-step reward
        action = np.argmax(Q_select[0])
        temp[action_batch[0]] = reward_batch[0] + (1 - int(done_batch[0])) * self.config.GAMMA * Q_eval[0][action]
        y_batch[0] = temp

        abs_errors = self.sess.run(self.abs_errors, feed_dict={self.y_input: y_batch,
                                                               self.select_input: state_batch})
        #print(abs_errors)
        #print(abs_errors.shape)

        self.replay_memory.store(np.array(transition), abs_errors, demo_data[0])
        #if (self.name == 'actor0'):
        #   print(demo)
        # epsilon->FINAL_EPSILON(min_epsilon)
        if self.replay_memory.full():
            self.epsilon = max(self.config.FINAL_EPSILON, self.epsilon * self.config.EPSILIN_DECAY)
    def train_Q_network(self, pre_train=False, update=True):
        """
        :param pre_train: True means should sample from demo_buffer instead of replay_buffer
        :param update: True means the action "update_target_net" executes outside, and can be ignored in the function
        """
        if not pre_train and not self.replay_memory.full():  # sampling should be executed AFTER replay_memory filled
            return
        self.time_step += 1

        assert self.replay_memory.full() or pre_train
        ##For the Pretrain
        #actual_memory = self.demo_memory if pre_train else self.replay_memory
        actual_memory = self.replay_memory
        tree_idxes, minibatch, ISWeights = actual_memory.sample(self.config.BATCH_SIZE)

        #print(minibatch[3][2])

        np.random.shuffle(minibatch)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        done_batch = [data[4] for data in minibatch]
        demo_data = [data[5] for data in minibatch]
        n_step_reward_batch = [data[6] for data in minibatch]
        n_step_state_batch = [data[7] for data in minibatch]
        n_step_done_batch = [data[8] for data in minibatch]
        actual_n = [data[9] for data in minibatch]
        time_steps = [data[10] for data in minibatch]
        #print(time_steps)
        #print(np.reshape(next_state_batch, [-1, 84, 84, 3]))

        # provide for placeholder，compute first
        #Q_select = self.sess.run(self.Q_select, feed_dict={self.select_input: next_state_batch})
        #Q_eval = self.sess.run(self.Q_eval, feed_dict={self.eval_input: next_state_batch})
        #n_step_Q_select = self.sess.run(self.Q_select,feed_dict={self.select_input: n_step_state_batch})
        #n_step_Q_eval = self.sess.run(self.Q_eval, feed_dict={self.eval_input: n_step_state_batch})
        #temp = self.sess.run(self.Q_select, feed_dict={self.select_input: state_batch})
        #print(state_batch)
        whole_state_bat = []
        whole_state_bat = next_state_batch + n_step_state_batch + state_batch
        whole_eval_bat = next_state_batch + n_step_state_batch
        #print(whole_state_bat.__len__())
        #Q_select, Q_eval = self.sess.run([self.Q_select, self.Q_eval], feed_dict={self.select_input: next_state_batch, self.eval_input: next_state_batch})
        #n_step_Q_select, n_step_Q_eval = self.sess.run([self.Q_select, self.Q_eval], feed_dict={self.select_input: n_step_state_batch, self.eval_input: n_step_state_batch})
        #temp = self.sess.run(self.Q_select, feed_dict={self.select_input: state_batch})
        whole_Q_select, whole_Q_eval = self.sess.run([self.Q_select, self.Q_eval], feed_dict={self.select_input : whole_state_bat, self.eval_input : whole_eval_bat})
        Q_select, n_step_Q_select, temp = whole_Q_select[0:self.config.BATCH_SIZE], whole_Q_select[self.config.BATCH_SIZE: self.config.BATCH_SIZE *2], whole_Q_select[self.config.BATCH_SIZE*2: self.config.BATCH_SIZE *3]
        Q_eval, n_step_Q_eval = whole_Q_eval[0:self.config.BATCH_SIZE], whole_Q_eval[self.config.BATCH_SIZE: self.config.BATCH_SIZE *2]
        #print('0:' +str(time.time()- start_time))
        #start_time = time.time()
        #print(temp[2][1])
        y_batch = np.zeros((self.config.BATCH_SIZE, self.action_dim))
        n_step_y_batch = np.zeros((self.config.BATCH_SIZE, self.action_dim))
        for i in range(self.config.BATCH_SIZE):
            # state, action, reward, next_state, done, demo_data, n_step_reward, n_step_state, n_step_done = t
            #print(state_batch[i].shape)
            temp_0 = np.copy(temp[i])
            # add 1-step reward
            action = np.argmax(Q_select[i])
            temp[i][action_batch[i]] = reward_batch[i] + (1 - int(done_batch[i])) * self.config.GAMMA * Q_eval[i][action]
            y_batch[i] = temp[i]
            # add n-step reward
            action = np.argmax(n_step_Q_select[i])
            q_n_step = (1 - int(n_step_done_batch[i])) * self.config.GAMMA**actual_n[i] * n_step_Q_eval[i][action]
            temp_0[action_batch[i]] = n_step_reward_batch[i] + q_n_step
            n_step_y_batch[i] = temp_0

            if(demo_data[i] == 1.0):
                self.demo_num = self.demo_num + 1
            self.qvalue += Q_select[i]
            self.sum_age = self.sum_age + time_steps[i]
        _, loss, abs_errors = self.sess.run([self.optimize, self.loss, self.abs_errors],
                                      feed_dict={self.y_input: y_batch,
                                                 self.n_step_y_input: n_step_y_batch,
                                                 self.select_input: state_batch,
                                                 self.action_batch: action_batch,
                                                 self.isdemo: demo_data,
                                                 self.ISWeights: ISWeights})

        for error in abs_errors :
            self.sum_abs_error += error
        if(self.time_step % self.config.LEARNER_TRAINING_PART == 0):
            print("Q_select : " + str(Q_select[0]))
        #print(loss)
        #print(tree_idxes)

        #print(abs_errors.shape)
        self.replay_memory.batch_update(tree_idxes, abs_errors, demo_data)  # update priorities for data in memory

        # 此例中一局步数有限，因此可以外部控制一局结束后update ，update为false时在外部控制
        if update and self.time_step % self.config.UPDATE_TARGET_NET == 0:
            self.sess.run(self.update_target_net)

    def egreedy_action(self, state):
        a= self.sess.run(self.update_local_ops)

        #sum =0
        #for i in a[5]:
        #    sum = sum + i
        #print(sum)

        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        #print('no_random')
        return np.argmax(self.sess.run(self.Q_select, feed_dict={self.select_input: [state]})[0])
    def setLocalNet(self):
        self.sess.run(self.update_local_ops)
    def getSelectNet(self):
        return self.sess.run(tf.get_collection(self.name+'_select_net_params'))








