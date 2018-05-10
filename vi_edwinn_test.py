# -*- coding: utf-8 -*-
import irl_sarsa as irl
import random
from Tkinter import *
n_actions = 8

#p_action is previous action; n_action is next action; p_IFV and n_IFV is current and next state respectively
# action_id is for example "a000", "a123", ...

from utils import dlib_detect, draw_pts, show_detection
import tensorflow as tf
import numpy as np
import collections
import pickle, time
import vgg_face
import argparse
import cv2
import dlib

#For ipcam
import urllib
import time

emo_list = ['angry','contemptuous','disgusted','fearful','happy','neutral','sad','surprised']
face_det = dlib.get_frontal_face_detector()
lm_det = dlib.shape_predictor('../model/shape_predictor_68_face_landmarks.dat')
frames = 10
model = 'joint-fine-tune'

top2pref = [3, 6]
good_s = [16, 48, 128, 144, 160, 176]
neutral_s = [32]
bad_s = list(set(range(256))-set(good_s)-set(neutral_s))

class robohon:
    def __init__(self, total_steps):
        #IP cam emotion recognition
        self.lm_ph = tf.placeholder(tf.float32, [None, 51*2])
        self.img_ph = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.keep_prob = tf.placeholder(tf.float32)

        self.sess = tf.Session()
        dgn = vgg_face.DGN()
        dgn.build(self.lm_ph, self.keep_prob)
        self.pred_dgn = tf.argmax(dgn.prob, 1)

        dan = vgg_face.Vgg_face()
        dan.build(self.img_ph, self.keep_prob)
        self.pred_dan = tf.argmax(dan.prob, 1)

        if model == 'weighted-sum':

            prob_sum = tf.nn.softmax(dan.fc8+dgn.fc3)
            pred_sum = tf.argmax(prob_sum, 1)
        
        elif model == 'joint-fine-tune':

            saver = tf.train.Saver()
            saver.restore(self.sess, '../model/dgan.ckpt')
            prob_joint = tf.nn.softmax(dan.fc8+dgn.fc3)
            self.pred_joint = tf.argmax(prob_joint, 1)

        self.sess.run(tf.global_variables_initializer())
        self.reward = []
        
        #initialize state
        #Edwinn set alpha=0.3, epsilon=0.5, gamma=0.9 for first suggestion matches
        #for 4 suggestion matches, Edwinn set alpha=1, epsilon=0.1, gamma=0.9 
        self.ai = irl.sarsa(actions=range(n_actions), epsilon=0.0, alpha=0.3, gamma=0.9)
        self.p_IFV = random.choice([1, 32, 16])
        self.p_action = self.ai.chooseAction(self.p_IFV)
        self.epi_reward = 0
        self.count = 0
        self.total_steps = total_steps

        self.pref_action = 0
        
    def update(self):
        self.count += 1

	self.sarsa_to_actionid(self.p_action) #Print what robohon will perform
	#time.sleep(1.3)
	#print "React!"
	#Observe the new state from the action taken
	#n_IFV = self.getState()
        n_IFV = self.step(self.p_IFV, self.p_action)
	print "State: " + str(n_IFV)

	#Rewards
	r_ps = self.get_reward_ps(self.p_IFV, n_IFV)
	r_rs = 0.0
	self.epi_reward = r_rs + r_ps + self.epi_reward

        if self.p_action in top2pref:
            self.pref_action += 1

        n_action = self.ai.chooseAction(n_IFV)
	self.p_IFV = n_IFV
	self.p_action = n_action
        
        if self.count == 5:
            self.reward.append(self.epi_reward)
            self.p_IFV = random.choice([1, 32, 16])
            self.p_action = self.ai.chooseAction(self.p_IFV)
            self.epi_reward = 0
            self.count = 0

    def step(self, p_IFV, action):
        if action == 0:
            n_IFV = np.random.choice([16, 1], p=[0.5, 0.5])
        elif action == 1:
            n_IFV = np.random.choice([16, 1], p=[0.1, 0.9])
        elif action == 2:
            n_IFV = np.random.choice([16, 1], p=[0.7, 0.3])
        elif action == 3:
            n_IFV = np.random.choice([16, 1], p=[0.8, 0.2])
        elif action == 4:
            n_IFV = np.random.choice([16, 1], p=[0.2, 0.8])
        elif action == 5:
            n_IFV = np.random.choice([16, 1], p=[0.2, 0.8])
        elif action == 6:
            n_IFV = np.random.choice([16, 1], p=[0.9, 0.1])
        else:
            n_IFV = np.random.choice([16, 1], p=[0.2, 0.8])

        if n_IFV == 16:
            n_IFV = np.random.choice(good_s)
        elif n_IFV == 1:
            n_IFV = np.random.choice(bad_s)

        return n_IFV

    def restore(self, q_file):
        import pickle
        self.ai.q = pickle.load(open(q_file, 'rb'))
        
    #Policy Shaping Reward function: by transition from previous IFV (state) to new IFV (state)
    def get_reward_ps(self, p_IFV, n_IFV):
        if p_IFV == None:
            reward = 0
            print "Im NONE!"
        #Good -> Good    
        elif (self.good_state(p_IFV) == True) and (self.good_state(n_IFV) == True):
            reward = +0.1
        #Good -> Neutral
        elif (self.good_state(p_IFV) == True) and (n_IFV == 32):
            reward = 0.0
        #Good -> Bad
        elif (self.good_state(p_IFV) == True) and (self.good_state(n_IFV) == False):
            reward = -1.0
        #Bad -> Bad
        elif (self.good_state(p_IFV) == False) and (self.good_state(n_IFV) == False):
            reward = -0.1
        #Bad -> Good
        elif (self.good_state(p_IFV) == False) and (self.good_state(n_IFV) == True):
            reward = +1.0
        #Bad -> Neutral
        elif (self.good_state(p_IFV) == False) and (n_IFV == 32):
            reward = +0.1
        #Neutral -> Neutral
        elif (p_IFV == 32) and (n_IFV == 32):
            reward = 0.0
        #Neutral -> Good
        elif (p_IFV == 32) and (self.good_state(n_IFV) == True):
            reward = 0.1
        #Neutral -> Bad
        else:
            reward = -0.1
        return reward

    #This fucntion evaluates the state as being Good (True) or Bad (False)
    def good_state(self, state):
        if state == 128 or state == 160 or state == 176 or state == 16 or state == 48 or state == 144:
            return True
        else:
            return False

    #Reward Shaping Reward function: check if the suggested action is taken by the trainer
    def get_reward_rs(self, sarsa_action, actionid):
        # If the action learn by SARSA matches the user input
        if sarsa_action == self.actionid_to_sarsa(actionid): #sarsa_action is an int, actionid is a str
            reward = 1
        else:
            reward = 0
        return reward
    
    def getState(self):
        # Thank you Master Zih-Yun
        time.sleep(1) #TODO
        stream=urllib.urlopen('http://admin:@140.112.95.4:3333/MJPEG.CGI')
        bytes = ''
        emo_record = (np.ones(frames, dtype=int)*5).tolist()

        #TODO
        emo_buffer = collections.deque(maxlen=10)
        state_record = []

        print('Get state...')
        start_time = time.time()
        while True:
            bytes += stream.read(1024)
            a = bytes.find('\xff\xd8')
            b = bytes.find('\xff\xd9')
            if a != -1 and b != -1:
                frame = cv2.imdecode(np.fromstring(bytes[a:b+2], dtype=np.uint8), 1)
                bytes = bytes[b+2:]

                num, face, shape, shape_origin = dlib_detect(frame, 3, face_det, lm_det, 224, 224)
                if num == 1:

                    shape_norm = shape[17:]-shape[30]
                    shape_norm = shape_norm.reshape([1,51*2])
                    if model == 'dan':
                        pred = self.sess.run(self.pred_dan, feed_dict={self.img_ph: face.reshape([1,224,224,3]), self.keep_prob: 1.0})
                    elif model == 'dgn':
                        pred = self.sess.run(self.pred_dgn, feed_dict={self.lm_ph: shape_norm, self.keep_prob: 1.0})
                    elif model == 'weighted-sum':
                        pred = self.sess.run(self.pred_dgn, feed_dict={self.img_ph: face.reshape([1,224,224,3]), self.lm_ph: shape_norm, self.keep_prob: 1.0})
                    elif model == 'joint-fine-tune':
                        pred = self.sess.run(self.pred_joint, feed_dict={self.img_ph: face.reshape([1,224,224,3]), self.lm_ph: shape_norm, self.keep_prob: 1.0})
                    
                    emo_record.append(int(pred))
                    del emo_record[0]
                    ctr = collections.Counter(emo_record)

                    #TODO
                    emo_buffer.append(ctr)
                    emo_his = collections.Counter()
                    emo_his_table = np.zeros(len(emo_list))
                    emo_now_table = np.zeros(len(emo_list))
                    for c in emo_buffer:
                        emo_his += c
                    emo_his_avg = [v/float(len(emo_buffer)) for v in emo_his.values()]
                    emo_his = emo_his.items()
                    emo_his = np.array([np.array([list(emo_his[i])[0], emo_his_avg[i]]) for i in range(len(emo_his))])
                    emo_his_table[emo_his[:,0].astype(int)] = emo_his[:,1]
                    emo_now = ctr.items()
                    emo_now = np.array([np.array([list(e)[0], list(e)[1]]) for e in emo_now])
                    emo_now_table[emo_now[:,0].astype(int)] = emo_now[:,1] 
                    state = np.array([(emo_now_table[i]>=3 and (emo_now_table[i]-emo_his_table[i])>=0.0) for i in range(len(emo_list))]).astype(int)

                    state_int = 0
                    for i, j in enumerate(state):
                        state_int += j<<i

                    state_record.append(state_int)
                    print(state_int)
                    if len(state_record) == 15:
                        state_ctr = collections.Counter(state_record)
                        final_state = state_ctr.most_common()[0][0]
                        del state_record[:]
                        break
                else:
                    if time.time()-start_time > 10.0:
                        final_state = 32
                        break

        print('State: ', final_state)
        return final_state
        
    def sarsa_to_actionid(self, action):
        # This function converts the action vector into action ID's (like a000)
	# If we use always overwrite the system's decision we don't need this. 
	# But this will come handy to know which choice does the system recommend
        if action == 0:
            #Introduction: a100
            #print "Send a100 to Robohon"
            print "Introduction"
        elif action == 1:
            #Bye bye
            #print "Send a101 to Robohon"
            print "Bye bye"
        elif action == 2:
            #Dialogue/Chat: Animal ; School; Birthday
            chat = ["a102", "a103", "a104"]
            selected = random.choice(chat)
            #print "Send " + selected + " to Robohon" 
            print "Dialogue/Chat: Animal ; School; Birthday"
        elif action == 3:
            #Jokes: Sky Airplane; Ca Xiao-Ming; School; Ketchup
            jokes = ["a110", "a111", "a112", "a113"]
            selected = random.choice(jokes)
            #print "Send " + selected + " to Robohon" 
            print "Jokes: Sky Airplane; Ca Xiao-Ming; School; Ketchup"
        elif action == 4:
            #Riddle: 2 People; Right Hand; She is a bird; Foot; Monster
            riddle = ["a120", "a121", "a122", "a123", "a124"]
            selected = random.choice(riddle)
            #print "Send " + selected + " to Robohon"
            print "Riddle: 2 People; Right Hand; She is a bird; Foot; Monster"
        elif action == 5:
            #Stand Up ; Sit Down
            motion = ["a200", "a201"]
            selected = random.choice(motion)
            #print "Send " + selected + " to Robohon"
            print "Stand Up ; Sit Down"
        elif action == 6:
            #Dance: Kung-Fu; Air Guitar; Tecnho
            dance = ["a300", "a310", "a311"]
            selected = random.choice(dance)
            #print "Send " + selected + " to Robohon"
            print "Dance: Kung-Fu; Air Guitar; Tecnho"
        elif action == 7:
            #Complex: Take a photo
            photo = "a320"
            #print "Send " + photo + " to Robohon"
            print "Complex: Take a photo"
        return action

    def actionid_to_sarsa(self, actionid):
        #This function converts an action ID to it's action number (e.g Stand Up, a200 --> return 5)
        if actionid == "a100":
           #Introduction: a100
           action = 0
        elif actionid == "a101":
            #Bye bye
            action = 1
        elif actionid == "a102" or actionid == "a103" or actionid == "a104":
            #Dialogue/Chat: Animal ; School; Birthday
            action = 2
        elif actionid == "a110" or actionid == "a111" or actionid == "a112"or actionid == "a113":
            #Jokes: Sky Airplane; Ca Xiao-Ming; School; Ketchup
            action = 3
        elif actionid == "a120" or actionid == "a121" or actionid == "a122" or actionid == "a123" or actionid =="a124":
            #Riddle: 2 People; Right Hand; She is a bird; Foot; Monster
            action = 4
        elif actionid == "a200" or actionid == "a201":
            #Stand Up ; Sit Down
            action = 5
        elif actionid == "a300" or actionid == "a310" or actionid == "a311":
            #Dance: Kung-Fu; Air Guitar; Tecnho
            action = 6
        else:
            #actionid == "a320":
            #Complex: Take a photo
            action = 7
        return action
    
    def save_reward(self):
        import pickle
        pickle.dump(self.reward, open('./rewards', 'wb'))

'''
def set_action(str):
    global action_id #Be careful when using global variables
    action_id = str
    #print action_id
    
def send_btn():
    #update sarsa
    #print "This is the action_id in the Send Button: " + action_id
    robot.update(action_id)

def save_btn():
    robot.save_reward()

# Start creating the GUI    
root = Tk()
menu = Menu(root)
root.config(menu=menu)
#
##*****************************Quick*************************
quick = Menu(menu)
menu.add_cascade(label="Quick", menu=quick)

positive = Menu(quick)
quick.add_cascade(label="Positive",menu=positive)
positive.add_command(label="Can you tell me more?",command=lambda *args:set_action("a000"))
positive.add_command(label="Please say it again",command=lambda *args:set_action("a001"))
positive.add_command(label="Do not say that again",command=lambda *args:set_action("a002"))
positive.add_command(label="I'm sorry, I don't know what you are talking about",command=lambda *args:set_action("a003"))
positive.add_command(label="You can guess",command=lambda *args:set_action("a004"))
positive.add_command(label="I've heard some interesting Riddles recently",command=lambda *args:set_action("a005"))
positive.add_command(label="I will tell you, can you guess?",command=lambda *args:set_action("a006"))

reaction = Menu(quick)
quick.add_cascade(label="Reaction",menu=reaction)
reaction.add_command(label="Really?",command=lambda *args:set_action("a010"))
reaction.add_command(label="Wow!",command=lambda *args:set_action("a011"))
reaction.add_command(label="No Joking!",command=lambda *args:set_action("a012"))
reaction.add_command(label="Interesting",command=lambda *args:set_action("a013"))

q_other = Menu(quick)
quick.add_cascade(label="Other", menu=q_other)
q_other.add_command(label="Ah cannot see you! Can you Rotate me?",command=lambda *args:set_action("a020"))
q_other.add_command(label="Dui a",command=lambda *args:set_action("a021"))
q_other.add_command(label="Bu dui",command=lambda *args:set_action("a022"))
q_other.add_command(label="You a!",command=lambda *args:set_action("a023"))
q_other.add_command(label="Mei you!",command=lambda *args:set_action("a024"))
q_other.add_command(label="It must be!",command=lambda *args:set_action("a025"))

##************************************Verbal***************************
verbal = Menu(menu)
menu.add_cascade(label="Verbal",menu=verbal)

dialogue = Menu(verbal)
verbal.add_cascade(label="Dialogue",menu=dialogue)
dialogue.add_command(label="Introduction",command=lambda *args:set_action("a100"))
dialogue.add_command(label="Bye",command=lambda *args:set_action("a101"))
dialogue.add_command(label="Animal",command=lambda *args:set_action("a102"))
dialogue.add_command(label="School",command=lambda *args:set_action("a103"))
reaction.add_command(label="Birthday",command=lambda *args:set_action("a104"))

joke = Menu(verbal)
verbal.add_cascade(label="Joke", menu=joke)
joke.add_command(label="Sky Airplane",command=lambda *args:set_action("a110"))
joke.add_command(label="Ca XaioMing",command=lambda *args:set_action("a111"))
joke.add_command(label="Don't go to Class",command=lambda *args:set_action("a112"))
joke.add_command(label="Ketchup",command=lambda *args:set_action("a113"))

riddles = Menu(verbal)
verbal.add_cascade(label="Riddles", menu=riddles)
riddles.add_command(label="2 People",command=lambda *args:set_action("a120"))
riddles.add_command(label="Right Hand",command=lambda *args:set_action("a121"))
riddles.add_command(label="She is a Bird",command=lambda *args:set_action("a122"))
riddles.add_command(label="Foot",command=lambda *args:set_action("a123"))
riddles.add_command(label="Monster",command=lambda *args:set_action("a124"))

verbal.add_separator()
verbal.add_command(label="Recruitment",command=lambda *args:set_action("a130"))

##************************************Non-Verbal********************************
n_verbal = Menu(menu)
menu.add_cascade(label="Non-Verbal", menu=n_verbal)
n_verbal.add_command(label="Stand Up",command=lambda *args:set_action("a200"))
n_verbal.add_command(label="Sit Down",command=lambda *args:set_action("a201"))

##*************************************Complex*************************************
conplex = Menu(menu)
menu.add_cascade(label="Complex", menu=conplex)
conplex.add_command(label="Kung-Fu",command=lambda *args:set_action("a300"))
conplex.add_command(label="Dance - Air Guitar",command=lambda *args:set_action("a310"))
conplex.add_command(label="Dance - EDM",command=lambda *args:set_action("a311"))
conplex.add_command(label="Photo",command=lambda *args:set_action("a320"))

## Create Button and instantiate class:

robot = robohon()

b = Button(root, text="Send", command=send_btn)
b.pack(side="top", fill='both', expand=True, padx=4, pady=4)

b_2 = Button(root, text="Save", command=save_btn)
b_2.pack(side="top", fill='both', padx=4, pady=4)

root.geometry("250x150+300+300")
mainloop()
'''

import sys
total_steps = int(sys.argv[1])
robot = robohon(total_steps)
robot.restore(sys.argv[2])
for i in range(total_steps):
    robot.update()
#robot.save_reward()
#print('Reward mean: ', sum(robot.reward)/len(robot.reward))
print('Action matches: ', 100.0*robot.pref_action/total_steps)
