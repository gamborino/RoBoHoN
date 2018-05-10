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

class Demo(object):
    def __init__(self, args):
        self.env = Env(args.ip, args.port, args.epi_steps, args.emo_model)
        self.action_dict = action_dict()

        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        rohobon_message_pb2_grpc.add_RoBoHoNMessageServicer_to_server(Servicer(), self.server)
        self.server.add_insecure_port('[::]:50051')
        self.server.start()

    def ruleBasedAction(self, state):
        dialogue = [2, 4, 5]
        joke = [0, 1, 3, 6]
        riddle = [10, 11, 12, 13, 14]
        dance = [7, 8, 9]

        if self.env.goodState(state):
            action = random.sample(dialogue, 1)[0]
        elif state == 0 or state == 32:
            action = random.sample(riddle, 1)[0]
        else:
            action = random.sample(joke + dance, 1)[0]

        return action

    def train(self, total_epi):
        #Default posture: sit down
        sit = True

        epi_num = 1
        while epi_num <= total_epi:
            #TODO: Randomly choose one of the jokes, dialogues, and dances
            '''
            dialogue = random.sample([2, 4, 5], 1)[0]
            joke = random.sample([0, 1, 3, 6], 1)[0]
            riddle = random.sample([10, 11, 12, 13, 14], 1)[0]
            dance = random.sample([7, 8, 9], 1)[0]
            action_series = [joke, dialogue, dance]
            '''

            #Do "Introduction"
            sub_action_num = 6
            for a in range(sub_action_num):
                global action_id
                action_id = 'ma100' + str(a+1)
                time.sleep(1.5)
                raw_input('Please press enter to continue: ')
            
            s = self.env.GetInitState()

            #TODO
            #action_num = action_series[self.env.state_num]
            action_num = self.ruleBasedAction(s)
            action = self.action_dict.keys()[action_num]
            done_action = []
            done_action.append(action_num)

            while True:
                #If needed, stand up
                if action in ['a300', 'a310', 'a311'] and sit:
                    sub_action_num = 3
                    for a in range(sub_action_num):
                        global action_id
                        action_id = 'ma200' + str(a+1)
                        time.sleep(1.5)
                        raw_input('Please press enter to continue: ')
                    sit = False

                #Do the action
                sub_action_num = self.action_dict[action]
                for a in range(sub_action_num):
                    global action_id
                    action_id = 'm' + str(action) + str(a+1)
                    time.sleep(1.5)
                    raw_input('Please press enter to continue: ')

                r, s2, t = self.env.Step(s)

                s = s2

                if t:
                    #Do "Take a photo"
                    sub_action_num = 7
                    for a in range(sub_action_num):
                        global action_id
                        action_id = 'ma320' + str(a+1)
                        time.sleep(1.5)
                        raw_input('Please press enter to continue: ')

                    #Do "Bye Bye"
                    sub_action_num = 4
                    for a in range(sub_action_num):
                        global action_id
                        action_id = 'ma101' + str(a+1)
                        time.sleep(1.5)
                        raw_input('Please press enter to continue: ')

                    epi_num += 1
                    break

                #TODO
                #action_num = action_series[self.env.state_num]
                action_num = self.ruleBasedAction(s)
                fail = 0
                while action_num in done_action and fail < 5:
                    fail += 1
                    action_num = self.ruleBasedAction(s)
                if fail == 0:
                    done_action.append(action_num)
                action = self.action_dict.keys()[action_num]

        raw_input('Please disconnect RoBoHoN and press enter to continue: ')

        self.server.stop(0)

if __name__ == '__main__':
    global action_id #Be careful when using global variables
    action_id = None

    #args and train
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='192.168.0.194', help='IP of the ip cam')
    parser.add_argument('--port', type=int, default=3333, help='Port of the ip cam')
    parser.add_argument('--total_epi', type=int, default=1, help='Number of episode to run')
    parser.add_argument('--epi_steps', type=int, default=3, help='Steps for each episode')
    parser.add_argument('--emo_model', type=str, default='joint-fine-tune', \
            help='Emotion recognition model: dan, dgn, weighted-sum, joint-fine-tune')

    args = parser.parse_args()

    rl = Demo(args)
    raw_input('Please connect RoBoHoN and press enter to continue: ')
    name = raw_input('Please enter the name: ')
    action_id = 'n' + name
    age = raw_input('Please enter the age: ')
    action_id = 'a' + age
    gender = raw_input('Please enter the gender: ')
    action_id = 'g' + gender

    time.sleep(1.0)

    rl.train(args.total_epi)
