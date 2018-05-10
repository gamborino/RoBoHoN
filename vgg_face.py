import os
import numpy as np
import tensorflow as tf
import time

raf_mean=np.array([119.85, 136.35, 174.27])

class Vgg_face:
    def __init__(self):

        self.train = True
        self.vgg16_npy_path = './pre-train model/vgg16-save.npy'
        self.var_dict = {}

    def build(self, img, dropout):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        self.data_dict = np.load(self.vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")
        print("build model started")

        assert img.get_shape().as_list()[1:] == [224, 224, 3]

        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=img)

        img_norm = tf.concat(axis=3, values=[
            red - raf_mean[0],
            green - raf_mean[1],
            blue - raf_mean[2],
        ])

        self.conv1_1 = self.conv_layer(img_norm, 3, 64, "conv1_1", mode='fixed')
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2", mode='fixed')
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1", mode='fixed')
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2", mode='fixed')
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1", mode='fixed')
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2", mode='fixed')
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3", mode='fixed')
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1", mode='fixed')
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2", mode='fixed')
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3", mode='fixed')
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1", mode='fixed')
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2", mode='fixed')
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3", mode='fixed')
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, 25088, 2048, "fc6", mode='fixed') 
        self.relu6 = tf.nn.relu(self.fc6)
        self.relu6 = tf.nn.dropout(self.relu6, dropout)

        self.fc7 = self.fc_layer(self.relu6, 2048, 1024, "fc7", mode='fixed')
        self.relu7 = tf.nn.relu(self.fc7)
        self.relu7 = tf.nn.dropout(self.relu7, dropout)

        self.fc8 = self.fc_layer(self.relu7, 1024, 8, "fc8", mode='fine-tune')

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name, mode):

        with tf.variable_scope(name):

            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name, mode)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name, mode):
        with tf.variable_scope(name):

            weights, biases = self.get_fc_var(in_size, out_size, name, mode)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name, mode):

        if mode == 'fixed' or self.train == False:

            filters = self.data_dict[name][0]
            biases = self.data_dict[name][1]

        elif mode == 'fine-tune':

            filters = tf.Variable(self.data_dict[name][0],  name = name + '_weights')
            biases = tf.Variable(self.data_dict[name][1], name = name + '_biases')
        elif mode == 'retrain':

            filters = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001), \
                name = name + '_weights')
            biases = tf.Variable(tf.truncated_normal([out_channels], 0.0, 0.001), \
                name = name + 'biases')
        else:
            print ('Mode should be fixed/fine-tune/retrain')

        self.var_dict[(name, 0)] = filters
        self.var_dict[(name, 1)] = biases

        return filters, biases

    def get_fc_var(self, in_size, out_size, name, mode):

        if mode == 'fixed' or self.train == False:

            weights = self.data_dict[name][0]
            biases = self.data_dict[name][1]          
        elif mode == 'fine-tune':

            weights = tf.Variable(self.data_dict[name][0],  name = name + '_weights')
            biases = tf.Variable(self.data_dict[name][1], name = name + '_biases')
        elif mode == 'retrain':

            weights = tf.Variable(tf.truncated_normal([in_size, out_size], 0.0, 0.001), name = name + '_weights')
            biases = tf.Variable(tf.truncated_normal([out_size], 0.0, 0.001), name = name + 'biases')
        else:
            print ('Mode should be fixed/fine-tune/retrain')

        self.var_dict[(name, 0)] = weights
        self.var_dict[(name, 1)] = biases

        return weights, biases

    def save_npy(self, sess, npy_path="../model/vgg16-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print("file saved", npy_path)
        # return npy_path    

class DGN:
    
    def __init__(self, path='./pre-train model/dgn_model.npy'):

        self.var_dict = {}
        self.data_dict = np.load(path, encoding='latin1').item()
        self.train = True
    
    def build(self, lm, dropout):

        self.fc1 = self.fc_layer(lm, 51*2, 128, 'fc1', mode = 'fixed')
        self.relu1 = tf.nn.selu(self.fc1)
        self.drop1 = tf.nn.dropout(self.relu1, dropout)

        self.fc2 = self.fc_layer(self.drop1, 128, 256, 'fc2', mode = 'fixed')
        self.relu2 = tf.nn.selu(self.fc2)
        self.drop2 = tf.nn.dropout(self.relu2, dropout)

        self.fc3 = self.fc_layer(self.drop2, 256, 8, 'fc3', mode = 'fine-tune')
        self.prob = tf.nn.softmax(self.fc3, name='prob')

    def fc_layer(self, bottom, in_size, out_size, name, mode):
        with tf.variable_scope(name):

            weights, biases = self.get_fc_var(in_size, out_size, name, mode)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_fc_var(self, in_size, out_size, name, mode):

        if mode == 'fixed' or self.train == False:

            weights = self.data_dict[name][0]
            biases = self.data_dict[name][1]
        elif mode == 'fine-tune':

            weights = tf.Variable(self.data_dict[name][0],  name = name + '_weights')
            biases = tf.Variable(self.data_dict[name][1], name = name + '_biases')
        elif mode == 'retrain':

            weights = tf.Variable(tf.truncated_normal([in_size, out_size], 0.0, 0.001), name = name + '_weights')
            biases = tf.Variable(tf.truncated_normal([out_size], 0.0, 0.001), name = name + '_biases')
        else:
            print ('Mode should be fixed/fine-tune/retrain')

        self.var_dict[(name, 0)] = weights
        self.var_dict[(name, 1)] = biases

        return weights, biases

    def save_npy(self, sess, npy_path="./model/dgn-save.npy"):
        # assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path