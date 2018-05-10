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

emo_list = ['angry','contemptuous','disgusted','fearful','happy','neutral','sad','surprised']
face_det = dlib.get_frontal_face_detector()
lm_det = dlib.shape_predictor('../model/shape_predictor_68_face_landmarks.dat')
frames = 10

def main(model):

    #parser = argparse.ArgumentParser(prog='test.py', description='ref_emotion.')
    #parser.add_argument('--model', type=str, default='dgn')
    #args = parser.parse_args()

    lm_ph = tf.placeholder(tf.float32, [None, 51*2])
    img_ph = tf.placeholder(tf.float32, [None, 224, 224, 3])
    keep_prob = tf.placeholder(tf.float32)

    with tf.Session() as sess:

        dgn = vgg_face.DGN()
        # dgn.train = False
        dgn.build(lm_ph, keep_prob)
        pred_dgn = tf.argmax(dgn.prob, 1)

        dan = vgg_face.Vgg_face()
        # dan.train = False
        dan.build(img_ph, keep_prob)
        pred_dan = tf.argmax(dan.prob, 1)

        if model == 'weighted-sum':

            prob_sum = tf.nn.softmax(dan.fc8+dgn.fc3)
            pred_sum = tf.argmax(prob_sum, 1)
        
        elif model == 'joint-fine-tune':

            saver = tf.train.Saver()
            saver.restore(sess, '../model/dgan.ckpt')
            prob_joint = tf.nn.softmax(dan.fc8+dgn.fc3)
            pred_joint = tf.argmax(prob_joint, 1)

        sess.run(tf.global_variables_initializer())

        stream=urllib.urlopen('http://admin:@192.168.0.194:3333/MJPEG.CGI')
        #stream = cv2.VideoCapture('../../videos/20180226_154918.mp4')
        #stream = cv2.VideoCapture(clip)
        #if stream.isOpened() == False:
            #print('Error opening video stream of file')

        bytes = ''
        emo_record = (np.ones(frames, dtype=int)*5).tolist()

        #TODO
        '''
        emo_buffer = collections.deque(maxlen=10)
        state_record = []
        final_states = []
        '''

        import time
        time.sleep(10.0)

        while True:
        #while stream.isOpened():

            bytes += stream.read(1024)
            a = bytes.find('\xff\xd8')
            b = bytes.find('\xff\xd9')
            #ret, frame = stream.read()
            if a != -1 and b != -1:
            #if ret == True:
                frame = cv2.imdecode(np.fromstring(bytes[a:b+2], dtype=np.uint8), 1)
                bytes = bytes[b+2:]

                num, face, shape, shape_origin = dlib_detect(frame, 2, face_det, lm_det, 224, 224)
                if num == 1:

                    shape_norm = shape[17:]-shape[30]
                    shape_norm = shape_norm.reshape([1,51*2])
                    if model == 'dan':
                        pred = sess.run(pred_dan, feed_dict={img_ph: face.reshape([1,224,224,3]), keep_prob: 1.0})
                    elif model == 'dgn':
                        pred = sess.run(pred_dgn, feed_dict={lm_ph: shape_norm, keep_prob: 1.0})
                    elif model == 'weighted-sum':
                        pred = sess.run(pred_dgn, feed_dict={img_ph: face.reshape([1,224,224,3]), lm_ph: shape_norm, keep_prob: 1.0})
                    elif model == 'joint-fine-tune':
                        pred = sess.run(pred_joint, feed_dict={img_ph: face.reshape([1,224,224,3]), lm_ph: shape_norm, keep_prob: 1.0})
                    
                    emo_record.append(int(pred))
                    del emo_record[0]
                    ctr = collections.Counter(emo_record)

                    #TODO
                    '''
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
                    if len(state_record) == 10:
                        state_ctr = collections.Counter(state_record)
                        final_state = state_ctr.most_common()[0][0]
                        final_states.append(final_state)
                        del state_record[:]
                    '''

                    emotion = emo_list[ctr.most_common()[0][0]]
                    #print(emotion)
                    im_show = show_detection(frame, shape_origin, 1, emotion)

                    cv2.imshow('frame', im_show)
                else:
                    cv2.imshow('frame', frame)

                if cv2.waitKey(3) & 0xFF == ord('q'):
                    break
            #else:
                #break

        #stream.release()
        cv2.destroyAllWindows()

        #final_state_ctr = collections.Counter(final_states)
        #return final_state_ctr.most_common()[0][0]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='emotion model', type=str, default='joint-fine-tune')
    args = parser.parse_args()

    state = main(args.model)
