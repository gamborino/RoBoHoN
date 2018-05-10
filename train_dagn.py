from utils import load_data, print_process, gen_batch, split_data
import tensorflow as tf
import numpy as np
import pickle, time
import vgg_face
import cv2
from keras.preprocessing.image import ImageDataGenerator

epoch  = 500
batch_size = 16
dropout = 0.5

def main():

    with open('../data/ex_train_list.pkl','rb') as f1:
        train_list = pickle.load(f1)
    with open('../data/ex_landmark.pkl','rb') as f2:
        landmark = pickle.load(f2)

    print (len(train_list))

    print ('Loading data...')
    x_train, label, shape = load_data(train_list, landmark)
    x_train = np.asarray(x_train)
    shape = np.asarray(shape)
    label = np.asarray(label)

    (im_train, lm_train, gt_train), (x_val, lm_val, gt_val) = split_data(x_train, shape, label, split_ratio=0.1)

    img_ph = tf.placeholder(tf.float32, [None, 224, 224, 3])
    lm_ph = tf.placeholder(tf.float32, [None, 51*2])
    label_ph = tf.placeholder(tf.float32, [None, 8])
    keep_prob = tf.placeholder(tf.float32)
    lr_ph = tf.placeholder(tf.float32)
    
    with tf.Session() as sess:

        dan = vgg_face.Vgg_face()
        dan.build(img_ph, keep_prob)
        dgn = vgg_face.DGN()
        dgn.build(lm_ph, keep_prob)

        with tf.name_scope('dan'):
            dan_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=dan.fc8,labels=label_ph)
            dan_loss = tf.reduce_mean(dan_cross_entropy)

        with tf.name_scope('dgn'):
            dgn_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=dgn.fc3,labels=label_ph)
            dgn_loss = tf.reduce_mean(dgn_cross_entropy)

        with tf.name_scope('dagn'):
            dagn_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=dan.fc8+dgn.fc3,labels=label_ph)
            dagn_loss = tf.reduce_mean(dagn_cross_entropy)

        with tf.name_scope('loss'):
            loss = dan_loss + dgn_loss + 0.1*dagn_loss
            train_step = tf.train.AdamOptimizer(lr_ph).minimize(loss)

        with tf.name_scope('acc'):
            pred = tf.nn.softmax(dan.fc8+dgn.fc3)
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(label_ph, 1))
            accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        best_acc = 0.0
        lr = 1e-4
        best_loss = 1000
        for i in range(epoch):

            if i%50 == 0 and i != 0: 
               lr = 1e-4
               print ('\nlearning rate has reset to', lr)
            lr = 0.98*lr

            cnt = 0
            for im, lm, gt in gen_batch(im_train, lm_train, gt_train, batch_size):

                tStart = time.time()
                sess.run(train_step, feed_dict={img_ph: im, lm_ph: lm, label_ph: gt, keep_prob: 1.0, lr_ph: lr})
                tEnd = time.time()
                print_process(cnt, im_train.shape[0]//batch_size, tEnd - tStart)
                if cnt == im_train.shape[0]//batch_size:
                    break
                cnt += 1

            train_acc = 0.0
            train_loss = 0.0
            for im, lm, gt in gen_batch(im_train, lm_train, gt_train, batch_size):

                acc, l = sess.run((accuracy,loss),feed_dict={img_ph: im, lm_ph: lm, label_ph: gt, keep_prob: 1.0})
                train_acc += acc
                train_loss += l

            val_acc = 0.0
            val_loss = 0.0
            for im, lm, gt in gen_batch(x_val, lm_val, gt_val, batch_size):

                acc, l = sess.run((accuracy,loss),feed_dict={img_ph: im, lm_ph: lm, label_ph: gt, keep_prob: 1.0})
                val_acc += acc
                val_loss += l

            if (best_acc == val_acc/x_val.shape[0] and best_loss > val_loss) or best_acc < val_acc/x_val.shape[0]:

                print("Epoch: %d, training accuracy %.4f, loss: %.4f, val_acc: %.4f, val_loss: %.4f     val improve from %.4f to %.4f, save model." \
                    %(i+1, train_acc/im_train.shape[0], train_loss, val_acc/x_val.shape[0], val_loss, best_acc, val_acc/x_val.shape[0]))
                best_acc = val_acc/x_val.shape[0]
                best_loss = val_loss
                saver.save(sess, '../model/dgan.ckpt')

            else:

                print("Epoch: %d, training accuracy %.4f, loss: %.4f, val_acc: %.4f, val_loss: %.4f     val_acc doesn't improve." \
                	%(i+1, train_acc/im_train.shape[0], train_loss, val_acc/x_val.shape[0], val_loss))               


if __name__ == '__main__':

    main()

        
