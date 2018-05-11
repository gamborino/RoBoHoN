from det_silence import det_silence
from action_dict import action_dict

class test():

	def __init__(self):
		self.action_dict = action_dict()

	def do_action(self, action_key):
		action_values = self.action_dict.get(action_key)
		print(action_values)
		for i in range (len(action_values)):
			global action_id
			action_id = 'm' + action_key + str(i+1)
			#time.sleep(1.5)
            #raw_input('Please press enter to continue: ')
			if action_values[i] is 'b':
				speech = det_silence(5000)
				while not speech:
					print ("I didn't hear you") #Send a proactive response
					speech = det_silence(5000)
			else:
				det_silence(5000)


test = test()
action_key = test.action_dict.keys()[4]
test.do_action('a102')