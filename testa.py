from det_silence import det_silence
from action_dict import action_dict

a100 = ['b', 'b', 'a', 'b', 'b', 'a']
a101 = ['a1011', 'a1012', 'a1013', 'a1014']
a102 = ['a1021', 'a1022', 'a1023', 'a1024', 'a1025']
actions = [a100, a101, a102]


#Given an action_num by AI
action_num = 0

action = actions[action_num]
for i in range(len(action)):
    action_id = 'ma100' + str(i+1)
    print(action_id)
    print(action[i])
    if action[i] is 'b':
        speech = det_silence(5000)
        while not speech:
            print("I didn't hear you") # Send a proactive response
            speech = det_silence(5000)
    else:
        speech = det_silence(5000)