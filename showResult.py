from Config import *
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import csv

def read_lines():
    with open(Config.ACTOR_DATA_PATH +'episodescore/'+ '2018-04-07:01:16:46.csv', 'rU') as data:
        reader = csv.reader(data)
        skiprow = 1
        count =0
        for row in reader:
            count = count +1
            if(skiprow < count) :
                yield [ float(i) for i in row ]

time = []
score = []
step = 10
read_count = 0
avg_time = 0
avg_score = 0
for i in read_lines():
    avg_time += i[0]
    avg_score += i[1]
    if(read_count % 10 == 0):
        time.append(avg_time/10)
        score.append(avg_score/10)
        avg_time = 0
        avg_score = 0

    read_count = read_count + 1

plt.plot(time,score)
plt.xlabel('Time')
plt.ylabel('Load Value')
plt.title('Logged Load\n')
plt.show()