'''
Sample script for running tensorflow- generating leaderboard submissions
@Author : Giridhur S
based on : https://www.tensorflow.org/versions/r0.11/tutorials/mnist/beginners/index.html\
'''

import tensorflow as tf
import cPickle as pkl
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import MiniBatchSparsePCA,PCA,IncrementalPCA
import matplotlib.pyplot as plt


RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def init_weights(shape):
    #TODO implement Xavier initialization
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden): # this network is the same as the previous one except with an extra hidden layer + dropout
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.elu(tf.matmul(X, w_h))
    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.elu(tf.matmul(h, w_h2))
    h2 = tf.nn.dropout(h2, p_keep_hidden)
    return tf.matmul(h2, w_o)


lbf = np.loadtxt('./dataset/leaderboardTest_data.csv',delimiter=',')
N = lbf.shape[0]
no_of_classes = 12
lbf_OH = np.zeros([N,no_of_classes])

#Do this only if u have done preprocessing
with open('./ipca_model.pkl',mode = 'rb') as f:
    ipca = pkl.load(f)

lbf_tf = ipca.transform(lbf)
#lbf_tf = lbf
lbf_temp = np.ones([N,lbf_tf.shape[1]+1])
lbf_temp[:,1:] = lbf_tf
lbf_tf = lbf_temp
#lbf_tf = lbf
#now we feed this to the neural net
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
ite = 500;

x_size = lbf_tf.shape[1]
y_size = no_of_classes

x = tf.placeholder(tf.float32,shape=[None,x_size])
y_ = tf.placeholder(tf.float32,shape=[None,y_size])

w_h1_size = 1024
w_h2_size = 200
#w_h3_size = 12
w_h1 = init_weights([x_size, w_h1_size])
w_h2 = init_weights([w_h1_size, w_h2_size])
#w_h3 = init_weights([w_h2_size,w_h3_size])
w_o = init_weights([w_h2_size, y_size])

p_keep_input = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)
py_x = model(x, w_h1, w_h2, w_o, p_keep_input, p_keep_hidden)
predict_op = tf.argmax(py_x, 1)

restorer = tf.train.Saver([w1_h1,w1_h2,w1_o,w2_h1,w2_h2,w2_o,alpha])
restorer.restore(sess,'./checkpoints/checkpoint_'+str(ite)+'.chk')
#now session has all variables


lbf_lbl=sess.run(predict_op, feed_dict={x: lbf_tf,p_keep_input: 1.0,p_keep_hidden: 1.0})
np.savetxt('lbd6.txt',lbf_lbl,fmt='%d')

#Visualization
ind = np.arange(1000)
np.random.shuffle(ind)
for i in ind:
    pic = trainf[i,:-1]
    pic = 0.5*pic+0.5
    pic = np.reshape(pic,[32,32,3])
    pic = sig.medfilt(pic)
    fnam = './dump/pt_'+str(i)+'.png'
    plt.imsave(fname=fnam,arr=pic)
