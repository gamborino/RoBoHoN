import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt

import math

print('Plot...')
file_num = len(sys.argv) - 1

timesteps = [2000, 4000, 10000, 20000, 40000, 100000, 200000]
log_timesteps = [math.log10(i) for i in timesteps]
legend = ['IRL', 'RL']

for f in range(file_num):
    v = pickle.load( open(sys.argv[f+1], "rb") )
    for i in range(5000):
        v[i+1] = 0.9*v[i] + 0.1*v[i+1]
    #plt.semilogx(timesteps, v, label=sys.argv[f+1][16:])
    plt.plot(v[:5000], label=legend[f])

#y = np.cumsum(v)

plt.legend(loc='lower right')
plt.title('Reward for Interactive Reinforcement Learning')
plt.xlabel('Episodes') 
plt.ylabel('Episode Reward')
plt.grid(True)
plt.show()
