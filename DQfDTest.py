# -*- coding: utf-8 -*
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from gym import wrappers
import gym
import numpy as np
import pickle
from Config import Config, DDQNConfig, DQfDConfig
from DQfD import DQfD
#from DQfDDDQN import DQfDDDQN
from collections import deque
import itertools
import scipy.signal
import time
from helper import *
import threading
from Memory import Memory
from PIL import Image
import math
import csv
from datetime import datetime

from random import shuffle
def process_frame(frame):
    s = scipy.misc.imresize(frame, [84, 84, 3])
    s = s / 255.0
    return s
def process_frame(frame):
    s = scipy.misc.imresize(frame, [84, 84, 3])
    s = s / 255.0
    return s
def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="float32" )/255.0
    data = process_frame(data)
    return data

def openLog(directory, filename, rlist):
    createTime = datetime.now().strftime('%Y-%m-%d:%H:%M:%S')
    with open(directory + str(createTime) + filename + '.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(rlist)
        myfile.close()
    return str(createTime) + filename
def writeLog( directory, filename, rlist):
    with open(directory + filename + '.csv', 'a') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(rlist)
        myfile.close()



def step(i,f, episode):



    line = f.readline()
    traj = line[:-1].split(",")
    episodeEnd = False
    if not line:
        episodeEnd = True
        f.close()
        return '','', '',  '',  episodeEnd
    state = load_image(screenpath + gameName + "/" + str(episode) + "/" + str(i) + ".png")
    #print(screenpath + gameName + "/" + str(episode) + "/" + str(i) + ".png")
    #print(traj)
    #print(i)
    done = bool(int(traj[3]))

    action = int(traj[4])
    translatedAction = actionTranslator[action]
    return state, float(traj[1]), done, translatedAction, episodeEnd

def goNextEpisode(count,file, episode):


    count = 0
    file = open(trajpath + gameName + "/" + str(episode) + ".txt", 'r')
    file.readline()
    file.readline()
    return count,file

def set_n_step(container, n):
    print(container)
    t_list = list(container)
    # accumulated reward of first (trajectory_n-1) transitions
    n_step_reward = sum([t[2] * Config.GAMMA**i for i, t in enumerate(t_list[0:min(len(t_list), n) - 1])])
    for begin in range(len(t_list)):
        end = min(len(t_list) - 1, begin + Config.trajectory_n - 1)
        n_step_reward += t_list[end][2]*Config.GAMMA**(end-begin)
        # extend[n_reward, n_next_s, n_done, actual_n]
        t_list[begin].extend([n_step_reward, t_list[end][3], t_list[end][4], end-begin+1])
        n_step_reward = (n_step_reward - t_list[begin][2])/Config.GAMMA
    return t_list



def actionTranslate(gymActions, dataSetActions):
    actionTranslation = []
    length = 0
    for action in dataSetActions :
        i = 0
        for gymAction in gymActions :
            if(action == gymAction):
                actionTranslation.append(i)
            i = i+1
        if(length == actionTranslation.__len__()):
            actionTranslation.append(0)
        length = actionTranslation.__len__()
    return actionTranslation


def map_scores(dqfd_scores=None, ddqn_scores=None, xlabel=None, ylabel=None):
    if dqfd_scores is not None:
        plt.plot(dqfd_scores, 'r')
    if ddqn_scores is not None:
        plt.plot(ddqn_scores, 'b')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()
def sign(x): return 1 if x >= 0 else -1

class Learner(object):
    def __init__(self, name, agent):
        self.name = name
        self.learner = agent
    def run(self):
        global train_itr
        episode_frames = []
        train_itr = 0

        scores, e, replay_full_episode = [], 0, None
        sample_log = openLog(Config.LEARNER_DATA_PATH + 'sampleexp/', '', ['step', 'value', 'age', 'demo'])
        replay_log = openLog(Config.LEARNER_DATA_PATH + 'replaymemory/', '', ['step', 'root_priority'])

        while not self.learner.replay_memory.full() :
             time.sleep(1)
            
        for i in range(Config.LEARNER_TRAINING_STEP) :
            # print(agent.replay_memory.full())
            #print(self.learner.replay_memory.tree.data_pointer)
            start_time = time.time()

            self.learner.train_Q_network(update=False)  # train along with generation
            train_itr += 1
            replay_full_episode = replay_full_episode or e
            #(self.learner.replay_memory.tree.total_p)
            if(train_itr % 10 == 0) :
                print("learner--- %s seconds ---" % (time.time() - start_time))
            if(train_itr % Config.LEARNER_TRAINING_PART == 0):
                self.learner.save_model()
                sample_demo = float(self.learner.demo_num) / (Config.LEARNER_TRAINING_PART*Config.BATCH_SIZE)
                sample_value = math.pow(self.learner.sum_abs_error / (Config.LEARNER_TRAINING_PART*Config.BATCH_SIZE), 0.4)
                sample_age = self.learner.sum_age / (Config.LEARNER_TRAINING_PART*Config.BATCH_SIZE)
                print("learner_sample")
                print(sample_demo)
                print(sample_value)
                print(sample_age)
                print("replay_memory")
                print(self.learner.replay_memory.tree.total_p)
                writeLog(Config.LEARNER_DATA_PATH + 'sampleexp/', sample_log,
                         [str(train_itr), str(sample_value), str(sample_age), str(sample_demo)])
                writeLog(Config.LEARNER_DATA_PATH + 'replaymemory/', replay_log,
                         [str(train_itr), str(self.learner.replay_memory.tree.total_p)])

                self.learner.sum_abs_error = 0
                self.learner.demo_num = 0
                self.learner.sum_age = 0
            if train_itr % Config().UPDATE_TARGET_NET == 0:
                self.learner.sess.run(self.learner.update_target_net)
class Actor(object):
    def __init__(self, name, env, agent, local):
        self.name = name
        self.env = env
        self.learner = agent
        self.actor = local
    def run(self):
        global train_itr
        episode_frames = []
        episode_count = 0
        deleted_value = 0
        deleted_age = 0
        deleted_demo = 0
        #
        lock = threading.Lock()
        print(self.name)

        count = 0 
        scores, e, replay_full_episode = [], 0, None
        filename = ''
        if(self.name == "actor0"):
            delete_log = openLog(Config.ACTOR_DATA_PATH+'deletedexp/', 'deletedExp', ['step', 'value', 'age', 'demo'])
            episode_log = openLog(Config.ACTOR_DATA_PATH + 'episodescore/', '', ['episode', 'score'])
        while not coord.should_stop():
            done, score, n_step_reward, state = False, 0, None, self.env.reset()
            state = process_frame(state)
            t_q = deque(maxlen=Config.trajectory_n)

            while done is False:
                time.sleep(Config.ACTOR_SLEEP)
                startTime = time.time()
                # print(index + " running!")
                action = self.actor.egreedy_action(state)  # e-greedy action for train
                next_state, reward, done, _ = self.env.step(action)
                # env.render()
                episode_frames.append(next_state)
                next_state = process_frame(next_state)
                # print(next_state)
                score += reward
                reward = sign(reward) * math.log(1 + abs(reward)) if not done else sign(-100) * math.log(1 + abs(-100))
                reward_to_sub = 0. if len(t_q) < t_q.maxlen else t_q[0][2]  # record the earliest reward for the sub
                t_q.append([state, action, reward, next_state, done, 0.0])

                if len(t_q) == t_q.maxlen:
                    if n_step_reward is None:  # only compute once when t_q first filled
                        n_step_reward = sum([t[2] * Config.GAMMA ** i for i, t in enumerate(t_q)])
                    else:
                        n_step_reward = (n_step_reward - reward_to_sub) / Config.GAMMA
                        n_step_reward += reward * Config.GAMMA ** (Config.trajectory_n - 1)
                    t_q[0].extend([n_step_reward, next_state, done, t_q.maxlen, self.learner.time_step])  # actual_n is max_len here
                    value, age, demo = self.actor.perceive(t_q[0])  # perceive when a transition is completed
                    deleted_value = deleted_value + value
                    deleted_age = deleted_age + age
                    deleted_demo = deleted_demo + demo
                    #print(demo)
                    # print(t_q[0][3])
                    #print(self.learner.time_step)
                    count = count +1
                    if(count % Config.ACTOR_ACTING_PART == 0 and "actor0" == self.name):
                        print(self.name + "--- %s seconds ---" % (time.time() - startTime) + "/"+str(self.actor.replay_memory.tree.data_pointer))
                    if(count % Config.ACTOR_ACTING_PART == 0 and "actor0" == self.name ):
                        sum_value = deleted_value / Config.ACTOR_ACTING_PART
                        sum_age = deleted_age / Config.ACTOR_ACTING_PART
                        sum_demo = float(deleted_demo) / Config.ACTOR_ACTING_PART
                        print("actor_deleted")
                        print(sum_value)
                        print(sum_age)
                        print(sum_demo)
                        deleted_value = 0
                        deleted_age = 0
                        deleted_demo = 0
                        writeLog( Config.ACTOR_DATA_PATH +'deletedexp/', delete_log, [str(count), str(sum_value), str(sum_age), str(sum_demo)] )
                    if self.actor.replay_memory.full():
                        replay_full_episode = replay_full_episode or e
                    if train_itr % Config().UPDATE_TARGET_NET == 0:
                        #print("actor_update_target"+str(train_itr))
                        self.actor.sess.run(self.actor.update_target_net)
                state = next_state
            if done:
                # handle transitions left in t_q
                t_q.popleft()  # first transition's n-step is already set
                transitions = set_n_step(t_q, Config.trajectory_n, self.learner.time_step)
                for t in transitions:
                    self.actor.perceive(t)
                    if self.actor.replay_memory.full():
                        replay_full_episode = replay_full_episode or e
                if self.actor.replay_memory.full():
                    if train_itr % Config().UPDATE_TARGET_NET == 0:
                        #print("actor_update_target")
                        self.actor.sess.run(self.actor.update_target_net)
                    scores.append(score)
                if replay_full_episode is not None:
                    print("episode: {}  trained-episode: {}  score: {}  memory length: {}  epsilon: {}"
                          .format(e, e - replay_full_episode, score, len(self.actor.replay_memory), self.actor.epsilon))
                    if(self.name == "actor0"):
                        writeLog(Config.ACTOR_DATA_PATH + 'episodescore/', episode_log,
                             [str(episode_count), str(score)])

                # 주기적으로 에피소드의 gif 를 저장하고, 모델 파라미터와 요약 통계량을 저장한다.
                if episode_count % Config.GIF_STEP == 0 and episode_count != 0 and self.name == 'actor0':
                        time_per_step = 0.01
                        images = np.array(episode_frames)
                        make_gif(images, './frames/dqfd_image' + str(episode_count) + '.gif',
                                 duration=len(images) * time_per_step, true_image=True, salience=False)
                episode_count = episode_count + 1
                episode_frames = []
                # if np.mean(scores[-min(10, len(scores)):]) > 495:
                #     break
                # agent.save_model()
            e += 1
        print("actor end")
        return scores

class Human(object):
    def __init__(self, name, agent, local, episodeList):
        self.name = name
        self.learner = agent
        self.human = local
        self.episodeList = episodeList
        self.episode = self.i = self.f = None
    def run(self):
        print(self.name)
        global train_itr
        while True :
            random.shuffle(self.episodeList)
            self.episode = self.episodeList[0]
            self.i, self.f = goNextEpisode(self.i, self.f, self.episode)
            for n in range(1, episodeList.__len__()):
                done, score, n_step_reward, state = False, 0, None, np.zeros([83, 83, 3], dtype=np.float32)
                state = process_frame(state)
                episodeEnd = False
                t_q = deque(maxlen=Config.trajectory_n)
                while (not episodeEnd):

                    startTime = time.time()
                    time.sleep(Config.ACTOR_SLEEP)
                    next_state, reward, done, action, episodeEnd = step(self.i, self.f, self.episode)
                    a = self.human.sess.run(self.human.update_local_ops)
                    asum = 0
                    for i in a[5]:
                        asum = asum + i
                    #print(asum)
                    self.i = self.i + 1
                    if (episodeEnd):
                        break
                    score += reward
                    reward = sign(reward) * math.log(1 + abs(reward)) if not done else sign(-100) * math.log(1 + abs(-100))


                    reward_to_sub = 0. if len(t_q) < t_q.maxlen else t_q[0][2]  # record the earliest reward for the sub
                    t_q.append([state, action, reward, next_state, done, 1.0])
                    # print(next_state)
                    if len(t_q) == t_q.maxlen:
                        if n_step_reward is None:  # only compute once when t_q first filled
                            n_step_reward = sum([t[2] * Config.GAMMA ** i for i, t in enumerate(t_q)])
                        else:
                            n_step_reward = (n_step_reward - reward_to_sub) / Config.GAMMA
                            n_step_reward += reward * Config.GAMMA ** (Config.trajectory_n - 1)

                        t_q[0].extend([n_step_reward, next_state, done, t_q.maxlen, self.learner.time_step])  # actual_n is max_len here
                        self.human.perceive(t_q[0])  # perceive when a transition is completed
                        if(self.i % Config.ACTOR_ACTING_PART == 0):
                            print(self.name + "--- %s seconds ---" % (time.time() - startTime))
                    state = next_state
                    if train_itr % Config().UPDATE_TARGET_NET == 0:
                        #print("human_update_target")
                        self.human.sess.run(self.human.update_target_net)

                if (episodeEnd):
                    # handle transitions left in t_q
                    print("human : episode end")
                    t_q.popleft()  # first transition's n-step is already set
                    transitions = set_n_step(t_q, Config.trajectory_n, self.learner.time_step)
                    for t in transitions:
                        self.human.perceive(t)
                    if self.human.replay_memory.full():
                        if train_itr % Config().UPDATE_TARGET_NET == 0:
                            #print("human_update_target")
                            self.human.sess.run(self.human.update_target_net)
                if len(scores) >= Config.episode:
                    break
                # e += 1
                self.episode = self.episodeList[n]
                self.i, self.f = goNextEpisode(self.i, self.f, self.episode)

# extend [n_step_reward, n_step_away_state] for transitions in demo
def set_n_step(container, n, training_step):
    t_list = list(container)
    # accumulated reward of first (trajectory_n-1) transitions
    n_step_reward = sum([t[2] * Config.GAMMA**i for i, t in enumerate(t_list[0:min(len(t_list), n) - 1])])
    for begin in range(len(t_list)):
        end = min(len(t_list) - 1, begin + Config.trajectory_n - 1)
        n_step_reward += t_list[end][2]*Config.GAMMA**(end-begin)
        # extend[n_reward, n_next_s, n_done, actual_n]
        t_list[begin].extend([n_step_reward, t_list[end][3], t_list[end][4], end-begin+1, training_step])
        n_step_reward = (n_step_reward - t_list[begin][2])/Config.GAMMA
    return t_list




if __name__ == '__main__':



    # env = wrappers.Monitor(env, '/tmp/CartPole-v0', force=True)
    # ------------------------ get demo scores by DDQN -----------------------------
    # get_demo_data(env)
    # --------------------------  get DDQN scores ----------------------------------
    # ddqn_sum_scores = np.zeros(Config.episode)
    # for i in range(Config.iteration):
    #     scores = run_DDQN(i, env)
    #     ddqn_sum_scores = np.array([a + b for a, b in zip(scores, ddqn_sum_scores)])
    # ddqn_mean_scores = ddqn_sum_scores / Config.iteration
    # with open('./ddqn_mean_scores.p', 'wb') as f:
    #     pickle.dump(ddqn_mean_scores, f, protocol=2)
    #with open('./ddqn_mean_scores.p', 'rb') as f:
    #    ddqn_mean_scores = pickle.load(f)
    # ----------------------------- get DQfD scores --------------------------------
    num_threads =Config.actor_num
    num_humanThread = Config.human_num
    actors = []
    acts = []
    threads = []
    #session = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    session = tf.InteractiveSession()
    env = gym.make(Config.ENV_NAME)
    replayMemory = Memory(capacity=Config.replay_buffer_size)
    coord = tf.train.Coordinator()
    
    scores, e, replay_full_episode = [], 0, None
    gameName = Config.GAME_NAME
    gameID = Config.ENV_NAME
    dataSetAction = Config.ACTION_SET
    env = gym.make(gameID)
    gymAction = env.unwrapped.get_action_meanings()
    actionTranslator = actionTranslate(gymAction, dataSetAction)
    episodeList = os.listdir(Config().SCREEN_PATH + gameName + '/')  # dir is your directory path

    screenpath = Config.SCREEN_PATH
    trajpath = Config.TRAJ_PATH

    threads = []
    #agent = DQfD('learner', env, DQfDConfig(), session, replayMemory)
    ##env = gym.make(Config.ENV_NAME)
    #local = DQfD('actor0', env, DQfDConfig(), session, replayMemory)
    #actor = Actor('actor0' + 0, env, agent, local)
    #actor.run()
    with tf.device('/gpu:0'):
        agent = DQfD('learner', env, DQfDConfig(), session, replayMemory)
        learner = Learner('learner',  agent)
        #learner.run()
        #print(agent.getSelectNet())
        learn = lambda: learner.run()
        t = threading.Thread(target=learn)
        t.start()
        threads.append(t)
    t = act = None

    actors = []
    with tf.device('/cpu:0'):
        for i in range(num_threads):
            env = gym.make(Config.ENV_NAME)
            local = DQfD('actor' + str(i), env, DQfDConfig(), session, replayMemory)
            actor = Actor('actor' + str(i), env, agent, local)
            actors.append(actor)
        for j in range(num_threads):
            act = lambda: actors[j].run()
            t = threading.Thread(target= act)
            t.start()
            threads.append(t)
    humans = []
    with tf.device('/cpu:0'):
        for i in range(num_humanThread):
            env = gym.make(Config.ENV_NAME)
            local = DQfD('human'+str(i), env, DQfDConfig(), session, replayMemory)
            human = Human('human'+str(i), agent, local, episodeList)
            humans.append(human)
        for j in range(num_humanThread):
            teach = lambda: humans[j].run()
            t = threading.Thread(target=teach)
            t.start()
            threads.append(t)
    coord.join(threads)
    #scores = run_DQfD(0, env, agent)


    #dqfd_sum_scores = np.zeros(Config.episode)
    #for i in range(Config.iteration):
    #    scores = run_DQfD(i, env)
    #    dqfd_sum_scores = np.array([a + b for a, b in zip(scores, dqfd_sum_scores)])
    #dqfd_mean_scores = dqfd_sum_scores / Config.iteration
    #with open('./dqfd_mean_scores.p', 'wb') as f:
    #    pickle.dump(dqfd_mean_scores, f, protocol=2)

    #map_scores(dqfd_scores=dqfd_mean_scores, ddqn_scores=ddqn_mean_scores,
    #    xlabel='Red: dqfd         Blue: ddqn', ylabel='Scores')
    env.close()
# gym.upload('/tmp/carpole_DDQN-1', api_key='sk_VcAt0Hh4RBiG2yRePmeaLA')