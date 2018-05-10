# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 11:44:10 2018

@author: Vicente
"""
import random

class sarsa:
    def __init__(self, actions, epsilon, alpha, gamma):        
        #use q as a dictionary
        self.q = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions # vector [0 1 2 3 4 5 6 7] if 8 actions
    
    def getQ(self, state,action):
        #Gets the (state, action) key if key isn't available return 0.0
        return self.q.get((state,action),0.0)

    def learnQ(self, state, action, r_ps, r_rs, Qnext):
        Q = self.q.get((state,action), None)
        if Q is None:
            self.q[(state, action)] = r_ps
        else:
            self.q[(state, action)] = Q + self.alpha * (r_ps + r_rs + self.gamma*Qnext - Q)
    
    def chooseAction(self, state):
        if random.random() < self.epsilon:
            #Choose random action
            action = random.choice(self.actions)
        else:
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            #Counts how many actions give best reward
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                #Choose one random, Notice I have to change this if I want to pick 4 actions like Edwinn did
                i = random.choice(best)
            else:
                i = q.index(maxQ)
            
            action = self.actions[i]
        return action #integer of action
    
    def learn(self, s, a, r_ps, r_rs, s_, a_):
        Qnext = self.getQ(s_,a_)
        self.learnQ(s,a,r_ps, r_rs, Qnext)
