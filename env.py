import random

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

import threading

class Env(object):
    def __init__(self, ip, port, total_steps, emo_model):
        self.ip = ip
        self.port = port

        self.action_dim = 15
        self.epi_num = 1
        self.state_num = 0

        #Emotion recognition module
        self.emo_list = ['angry','contemptuous','disgusted','fearful',\
                'happy','neutral','sad','surprised']
        self.face_det = dlib.get_frontal_face_detector()
        self.lm_det = dlib.shape_predictor('../model/shape_predictor_68_face_landmarks.dat')
        self.frames = 10

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

        self.emo_model = emo_model

        if self.emo_model == 'weighted-sum':
            prob_sum = tf.nn.softmax(dan.fc8+dgn.fc3)
            pred_sum = tf.argmax(prob_sum, 1)
        elif self.emo_model == 'joint-fine-tune':
            saver = tf.train.Saver()
            saver.restore(self.sess, '../model/dgan.ckpt')
            prob_joint = tf.nn.softmax(dan.fc8+dgn.fc3)
            self.pred_joint = tf.argmax(prob_joint, 1)

        self.sess.run(tf.global_variables_initializer())

        self.total_steps = total_steps
        random.seed()

        ipcam_url = 'http://admin:@'+ self.ip + ':' + str(self.port) + '/MJPEG.CGI'
        self.stream=urllib.urlopen(ipcam_url)
        self.stream.close()
        self.ipCamStart = True
        ipCamThread = threading.Thread(target=self._ipCamThread)
        ipCamThread.start()

    def StartIpCam(self):
        ipcam_url = 'http://admin:@'+ self.ip + ':' + str(self.port) + '/MJPEG.CGI'
        self.stream=urllib.urlopen(ipcam_url)
        self.bytes = ''
        self.emo_record = (np.ones(self.frames, dtype=int)*5).tolist()
        self.emo_buffer = collections.deque(maxlen=10)
        self.state_record = []

    def GetInitState(self):
        #return self._getEmotion()
        self.stream.close()
        state_ctr = collections.Counter(self.state_record)
        print(state_ctr.most_common()[0][0]) #TODO: test
        return state_ctr.most_common()[0][0]

    def Step(self, state):
        #n_state = self._getEmotion()
        self.stream.close()
        state_ctr = collections.Counter(self.state_record)
        n_state = state_ctr.most_common()[0][0]
        print(n_state) #TODO: test

        reward = self._reward(state, n_state)
        t = False

        self.state_num += 1
        if self.state_num == self.total_steps:
            t = True

        return reward, n_state, t

    def _ipCamThread(self):
        while self.ipCamStart:
            self._getEmotion()

    def _getEmotion(self):
        time.sleep(1.0)
        while not self.stream.fp == None:
            self.bytes += self.stream.read(1024)
            a = self.bytes.find('\xff\xd8')
            b = self.bytes.find('\xff\xd9')
            if a != -1 and b != -1:
                frame = cv2.imdecode(np.fromstring(self.bytes[a:b+2], dtype=np.uint8), 1)
                self.bytes = self.bytes[b+2:]

                num, face, shape, shape_origin = dlib_detect(\
                        frame, 3, self.face_det, self.lm_det, 224, 224)
                if num == 1:
                    shape_norm = shape[17:]-shape[30]
                    shape_norm = shape_norm.reshape([1,51*2])
                    if self.emo_model == 'dan':
                        pred = self.sess.run(self.pred_dan, feed_dict={\
                                self.img_ph: face.reshape([1,224,224,3]), self.keep_prob: 1.0})
                    elif self.emo_model == 'dgn':
                        pred = self.sess.run(self.pred_dgn, feed_dict={\
                                self.lm_ph: shape_norm, self.keep_prob: 1.0})
                    elif self.emo_model == 'weighted-sum':
                        pred = self.sess.run(self.pred_dgn, feed_dict={\
                                self.img_ph: face.reshape([1,224,224,3]), \
                                self.lm_ph: shape_norm, self.keep_prob: 1.0})
                    elif self.emo_model == 'joint-fine-tune':
                        pred = self.sess.run(self.pred_joint, feed_dict={\
                                self.img_ph: face.reshape([1,224,224,3]), \
                                self.lm_ph: shape_norm, self.keep_prob: 1.0})
                    
                    self.emo_record.append(int(pred))
                    del self.emo_record[0]
                    ctr = collections.Counter(self.emo_record)

                    #TODO
                    self.emo_buffer.append(ctr)
                    emo_his = collections.Counter()
                    emo_his_table = np.zeros(len(self.emo_list))
                    emo_now_table = np.zeros(len(self.emo_list))
                    for c in self.emo_buffer:
                        emo_his += c
                    emo_his_avg = [v/float(len(self.emo_buffer)) for v in emo_his.values()]
                    emo_his = emo_his.items()
                    emo_his = np.array([np.array([list(emo_his[i])[0], emo_his_avg[i]]) \
                            for i in range(len(emo_his))])
                    emo_his_table[emo_his[:,0].astype(int)] = emo_his[:,1]
                    emo_now = ctr.items()
                    emo_now = np.array([np.array([list(e)[0], list(e)[1]]) for e in emo_now])
                    emo_now_table[emo_now[:,0].astype(int)] = emo_now[:,1] 
                    state = np.array([(emo_now_table[i]>=3 and \
                            (emo_now_table[i]-emo_his_table[i])>=0.0) \
                            for i in range(len(self.emo_list))]).astype(int)

                    state_int = 0
                    for i, j in enumerate(state):
                        state_int += j<<i
                    if state_int == 0:
                        state_int = 32

                    self.state_record.append(state_int)
                else:
                    self.state_record.append(32)

    def goodState(self, state):
        if state == 128 or state == 160 or state == 176 or \
                state == 16 or state == 48 or state == 144:
            return True
        else:
            return False

    def _reward(self, state, n_state):
        #Good -> Good    
        if (self.goodState(state) == True) and (self.goodState(n_state) == True):
            reward = +0.1
        #Good -> Neutral
        elif (self.goodState(state) == True) and (n_state == 32):
            reward = 0.0
        #Good -> Bad
        elif (self.goodState(state) == True) and (self.goodState(n_state) == False):
            reward = -1.0
        #Bad -> Bad
        elif (self.goodState(state) == False) and (self.goodState(n_state) == False):
            reward = -0.1
        #Bad -> Good
        elif (self.goodState(state) == False) and (self.goodState(n_state) == True):
            reward = +1.0
        #Bad -> Neutral
        elif (self.goodState(state) == False) and (n_state == 32):
            reward = +0.1
        #Neutral -> Neutral
        elif (state == 32) and (n_state == 32):
            reward = 0.0
        #Neutral -> Good
        elif (state == 32) and (self.goodState(n_state) == True):
            reward = 0.1
        #Neutral -> Bad
        else:
            reward = -0.1

        return reward
