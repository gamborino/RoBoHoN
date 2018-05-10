from utils import dlib_detect, draw_pts, show_detection
import tensorflow as tf
import numpy as np
import collections
import pickle, time
import vgg_face
import argparse
import cv2
import dlib

emo_list = ['angry','contemptuous','disgusted','fearful','happy','neutral','sad','surprised']
face_det = dlib.get_frontal_face_detector()
lm_det = dlib.shape_predictor('../model/shape_predictor_68_face_landmarks.dat')
frames = 10

def main():

    parser = argparse.ArgumentParser(prog='test.py', description='ref_emotion.')
    parser.add_argument('--model', type=str, default='dgn')
    args = parser.parse_args()

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

        if args.model == 'weighted-sum':

            prob_sum = tf.nn.softmax(dan.fc8+dgn.fc3)
            pred_sum = tf.argmax(prob_sum, 1)
        
        elif args.model == 'joint-fine-tune':

            saver = tf.train.Saver()
            saver.restore(sess, '../model/dgan.ckpt')
            prob_joint = tf.nn.softmax(dan.fc8+dgn.fc3)
            pred_joint = tf.argmax(prob_joint, 1)

        sess.run(tf.global_variables_initializer())

        cap = cv2.VideoCapture(0)
        emo_record = (np.ones(frames, dtype=int)*5).tolist()

        while(cap.isOpened()):

            ret, frame = cap.read()
            if ret == True:

                num, face, shape, shape_origin = dlib_detect(frame, 2, face_det, lm_det, 224, 224)
                if num == 1:

                    shape_norm = shape[17:]-shape[30]
                    shape_norm = shape_norm.reshape([1,51*2])
                    if args.model == 'dan':
                        pred = sess.run(pred_dan, feed_dict={img_ph: face.reshape([1,224,224,3]), keep_prob: 1.0})
                    elif args.model == 'dgn':
                        pred = sess.run(pred_dgn, feed_dict={lm_ph: shape_norm, keep_prob: 1.0})
                    elif args.model == 'weighted-sum':
                        pred = sess.run(pred_dgn, feed_dict={img_ph: face.reshape([1,224,224,3]), lm_ph: shape_norm, keep_prob: 1.0})
                    elif args.model == 'joint-fine-tune':
                        pred = sess.run(pred_joint, feed_dict={img_ph: face.reshape([1,224,224,3]), lm_ph: shape_norm, keep_prob: 1.0})
                    
                    emo_record.append(int(pred))
                    del emo_record[0]
                    ctr = collections.Counter(emo_record)
                    emotion = emo_list[ctr.most_common()[0][0]]
                    im_show = show_detection(frame, shape_origin, 1, emotion)

                    cv2.imshow('frame', im_show)
                else:
                    cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
