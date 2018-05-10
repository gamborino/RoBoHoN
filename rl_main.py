import time
import random
from concurrent import futures

import sys
sys.path.insert(0, '../../grpc')
import grpc
import rohobon_message_pb2
import rohobon_message_pb2_grpc

from env import Env
from action_dict import action_dict
import irl_sarsa as irl
from det_silence import det_silence

class Servicer(rohobon_message_pb2_grpc.RoBoHoNMessageServicer):
    def RequestInfo(self, request, context):
        #Get info to send
        #print('Got a request type: ', request.request)
        return rohobon_message_pb2.desktop(info=GetInfo())

def GetInfo():
    #Combine with the UI
    global action_id
    if action_id == None:
        info = 'empty'
    else:
        info = action_id
        action_id = None
    return info

class RL(object):
    def __init__(self, args):
        self.env = Env(args.ip, args.port, args.epi_steps, args.emo_model)
        self.action_dict = action_dict()
        self.ai = irl.sarsa(actions=range(self.env.action_dim), epsilon=1.0, alpha=0.3, gamma=0.9)

        self.reward = []
        if args.restore:
            self._restore(args.q_file, args.r_file)

        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        rohobon_message_pb2_grpc.add_RoBoHoNMessageServicer_to_server(Servicer(), self.server)
        self.server.add_insecure_port('[::]:50051')
        self.server.start()

    def train(self, total_epi):
        #Default posture: sit down
        sit = True

        epi_num = 1
        action_index = range(self.env.action_dim) #TODO
        while epi_num <= total_epi:
            epi_reward = 0.0

            #Do "Introduction"
            self.env.StartIpCam()
            sub_action_num = 6
            for a in range(sub_action_num):
                global action_id
                action_id = 'ma100' + str(a+1)
                time.sleep(1.5)
                #raw_input('Please press enter to continue: ')
                #det_silence needs as argument the time (in ms) it will wait after last sound was detected, default is 100 ms
                det_silence(1000)
            
            s = self.env.GetInitState()

            #TODO
            action_num = self.ai.chooseAction(s)
            while action_num not in action_index:
                action_num = self.ai.chooseAction(s)
            action = self.action_dict.keys()[action_num]

            while True:
                #TODO
                #action = random.choice(self.action_dict.keys())

                #If needed, stand up
                if action in ['a300', 'a310', 'a311'] and sit:
                    sub_action_num = 3
                    for a in range(sub_action_num):
                        global action_id
                        action_id = 'ma200' + str(a+1)
                        time.sleep(1.5)
                        det_silence()
                    sit = False

                #Do the action
                self.env.StartIpCam()
                sub_action_num = self.action_dict[action]
                for a in range(sub_action_num):
                    global action_id
                    action_id = 'm' + str(action) + str(a+1)
                    time.sleep(1.5)
                    #raw_input('Please press enter to continue: ')
                    det_silence(2000)
                        
                #TODO
                #del self.action_dict[action]
                action_index.remove(action_num) 

                r, s2, t = self.env.Step(s)

                #TODO: Update the Q table
                self.ai.epsilon = max((10000-len(self.reward)*5)/10000.0 * self.ai.epsilon, \
                        0.0)
                a2 = self.ai.chooseAction(s2)
                while a2 not in action_index:
                    a2 = self.ai.chooseAction(s2)
                self.ai.learn(s, a, r, 0.0, s2, a2)

                epi_reward += r
                s = s2
                #TODO
                action_num = a2
                action = self.action_dict.keys()[action_num]

                if t:
                    #Do "Take a photo"
                    sub_action_num = 7
                    for a in range(sub_action_num):
                        global action_id
                        action_id = 'ma320' + str(a+1)
                        time.sleep(1.5)
                        #raw_input('Please press enter to continue: ')
                        det_silence(1000)
                        
                    #Do "Bye Bye"
                    sub_action_num = 4
                    for a in range(sub_action_num):
                        global action_id
                        action_id = 'ma101' + str(a+1)
                        time.sleep(1.5)
                        #raw_input('Please press enter to continue: ')
                        det_silence(1000)
                        
                    self.reward.append(epi_reward)
                    epi_num += 1
                    break

        self.env.ipCamStart = False
        raw_input('Please disconnect RoBoHoN and press enter to continue: ')

        self._save_reward_q()
        self.server.stop(0)

    def _restore(self, q_file, r_file):
        import pickle
        self.ai.q = pickle.load(open(q_file, 'rb'))
        self.reward = pickle.load(open(q_file, 'rb'))

    def _save_reward_q(self):
        import pickle
        pickle.dump(self.reward, open('./reward', 'wb'))
        pickle.dump(self.ai.q, open('./q', 'wb'))

if __name__ == '__main__':
    global action_id #Be careful when using global variables
    action_id = None

    #args and train
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='192.168.0.194', help='IP of the ip cam')
    parser.add_argument('--port', type=int, default=3333, help='Port of the ip cam')
    parser.add_argument('--total_epi', type=int, default=1, help='Number of episode to run')
    parser.add_argument('--epi_steps', type=int, default=5, help='Steps for each episode')
    parser.add_argument('--emo_model', type=str, default='joint-fine-tune', \
            help='Emotion recognition model: dan, dgn, weighted-sum, joint-fine-tune')
    parser.add_argument('--restore', type=bool, default=False, \
            help='Restore replay buffer or not')
    parser.add_argument('--q_file', type=str, default='./q', help='name of the q table to be restored')
    parser.add_argument('--r_file', type=str, default='./reward', help='name of the reward to be restored')

    args = parser.parse_args()

    rl = RL(args)
    raw_input('Please connect RoBoHoN and press enter to continue: ')
    name = raw_input('Please enter the name: ')
    action_id = 'n' + name
    age = raw_input('Please enter the age: ')
    action_id = 'a' + age
    gender = raw_input('Please enter the gender: ')
    action_id = 'g' + gender

    time.sleep(1.0)

    rl.train(args.total_epi)
