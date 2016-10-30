'''
Sample script for running tensorflow
@Author : Giridhur S
source : https://www.tensorflow.org/versions/r0.11/tutorials/mnist/beginners/index.html\
'''
import tensorflow as tf
import cPickle as pkl
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden): # this network is the same as the previous one except with an extra hidden layer + dropout
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))
    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))
    h2 = tf.nn.dropout(h2, p_keep_hidden)
    return tf.matmul(h2, w_o)


''' Splitting data'''
with open('train_data_pickle', mode='rb') as f:
    trainf = pkl.load(f)
trainf = pd.DataFrame.as_matrix(trainf)
labels = trainf[:,-1]
no_of_classes = 12
labels_OH = np.eyes(no_of_classes)[labels]
N = trainf.shape[0]
M = trainf.shape[1]-1 #last col was label
#adding a bias column
train = np.ones(N,M+1)
train[:,1:] = trainf[:,:-1] #preprending the column of ones

trX,teX,trY,teY = train_test_split(train,labels_OH,test_size=0.40,random_state=RANDOM_SEED)

x_size = trX.shape[1]
y_size = trY.shape[1]
x = tf.placeholder(tf.float32,shape=[None,x_size])
y_ = tf.placeholder(tf.float32,shape=[None,y_size]) #one hot prediction

#init weights of layers, w_h1 = hidden1, w_h2 = hidden2 and w_o = output
w_h1_size = 1000
w_h2_size = 500
w_h1 = init_weights([x_size, w_h1_size])
w_h2 = init_weights([w_h1_size, w_h2_size])
w_o = init_weights([w_h2_size, y_size)

#dropout params NOTE : DROPOUT WILL NOT BE PERFORMED ON THE FINAL OUTPUT LAYER!
p_keep_input = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)

py_x = model(x, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x,y_))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

MAX_ITER = 1000

with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()
    for i in xrange(MAX_ITER):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})
        #for prediction dropout rate should be set to 0, ie keep rate to 1
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX, Y: teY,
                                                         p_keep_input: 1.0,
                                                         p_keep_hidden: 1.0})))
