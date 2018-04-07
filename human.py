from PIL import Image
import gym
from Config import Config, DDQNConfig, DQfDConfig
import tensorflow as tf
from DQfD import DQfD
from collections import deque
import scipy.signal
import msgpack
import msgpack_numpy as m
import time
import os
import numpy as  np
from Config import *

gameName = Config.GAME_NAME
gameID = Config.ENV_NAME
dataSetAction = Config.ACTION_SET
episodeList = os.listdir(Config().SCREEN_PATH + gameName + '/')  # dir is your directory path
episode = episodeList[0]
screen_path = Config().SCREEN_PATH
gymAction = Config.ACTION_SET
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




def step():
    global i
    global f

    line = f.readline()
    traj = line[:-1].split(",")
    episodeEnd = False
    if not line:
        episodeEnd = True
        f.close()
        return '','', '',  '',  episodeEnd
    state = load_image(screen_path + gameName + "/" + str(episode) + "/" + str(i) + ".png")
    #print(traj)
    #print(i)
    done = bool(int(traj[3]))
    i = i + 1
    action = int(traj[4])
    translatedAction = actionTranslator[action]
    return state, float(traj[1]), done, translatedAction, episodeEnd

def goNextEpisode():
    global f
    global i
    i = 0
    f = open(Config.TRAJ_PATH + gameName + "/" + str(episode) + ".txt", 'r')
    f.readline()
    f.readline()

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



