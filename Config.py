# -*- coding: utf-8 -*-
import os


class Config:
    ENV_NAME = 'SpaceInvaders-v0'
    GAME_NAME = "spaceinvaders"
    ACTION_SET = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT',
                  'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE',
                   'DOWNLEFTFIRE']

    TRAJ_PATH = "/home/soboru963/atari_v2_release/trajectories/"
    SCREEN_PATH = '/home/soboru963/atari_v2_release/screens/'
    #TRAJ_PATH = "/home/soboru963/atari_v1/trajectories/"
    #SCREEN_PATH = '/home/soboru963/atari_v1/screens/'
    GAMMA = 0.99  # discount factor for target Q
    INITIAL_EPSILON = 1.0  # starting value of epsilon
    FINAL_EPSILON = 0.01  # final value of epsilon
    EPSILIN_DECAY = 0.999
    START_TRAINING = 1000  # experience replay buffer size
    BATCH_SIZE = 64  # size of minibatch
    UPDATE_TARGET_NET = 10000  # update eval_network params every 200 steps
    #UPDATE_TARGET_NET = 1  # update eval_network params every 200 steps
    LEARNING_RATE = 0.001
    DEMO_RATIO = 0.1
    LAMBDA = [1.0, 0.0, 1.0, 10e-5]  # for [loss_dq, loss_n_dq, loss_jeq, loss_l2]
    PRETRAIN_STEPS = 5000  # 750000
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/DQfD_model')
    ACTOR_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'actor/')
    LEARNER_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'learner/')
    GIF_STEP = 10
    human_num = 1
    actor_num = 7
    demo_buffer_size = 500 * 50
    #replay_buffer_size = demo_buffer_size * (human_num + actor_num)
    replay_buffer_size = 100

    iteration = 5
    episode = 300  # 300 games per iteration
    trajectory_n = 10  # for n-step TD-loss (both demo data and generated data)

    START_STEP = 379000
    LEARNER_TRAINING_STEP = 1500000
    LEARNER_TRAINING_PART = 1000
    ACTOR_ACTING_PART = 1000
    ACTOR_SLEEP = 0.05

class DDQNConfig(Config):
    demo_mode = 'get_demo'


class DQfDConfig(Config):
    demo_mode = 'use_demo'
demo_num = int(Config.BATCH_SIZE * Config.DEMO_RATIO)
