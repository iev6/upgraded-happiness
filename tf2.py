'''
Sample script for running tensorflow
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
from sklearn.externals import joblib
import matplotlib.pyplot as plt

RANDOM_SEED = 151195 #keep changing this
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
    #h3 = tf.nn.elu(tf.matmul(h2, w_h3))
    #h3 = tf.nn.dropout(h3, p_keep_hidden)
    return tf.matmul(h2, w_o)

''' Splitting data'''
with open('./dataset/train_data_pickle', mode='rb') as f:
    trainf = pkl.load(f)
with open('./pca_1024.pkl',mode = 'rb') as f:
    ipca = pkl.load(f)
trainf = pd.DataFrame.as_matrix(trainf)
labels = np.int32(trainf[:,-1])
no_of_classes = 12


trainf1 = ipca.transform(trainf[:,:-1]) #in memory computation
#loading PCA
#trainf1 = trainf[:,:-1]

N = trainf1.shape[0]
M = trainf1.shape[1] #last col was label


labels_OH  = np.zeros([N,no_of_classes])
labels_OH[np.arange(N),labels] = 1

#adding a bias column
train = np.ones([N,M+1])
train[:,1:] = trainf1[:,:] #preprending the column of ones
trX,teX,trY,teY = train_test_split(train,labels_OH,test_size=0.30,random_state=RANDOM_SEED)


#two parallel networks
x_size = trX.shape[1]
y_size = trY.shape[1]
#y_size = 1
x = tf.placeholder(tf.float32,shape=[None,x_size])
y_ = tf.placeholder(tf.float32,shape=[None,y_size]) #one hot prediction
#y_ = tf.placeholder(tf.int32,shape=[None]) #label

##NN1
w1_h1_size = 300
w1_h2_size = 50
w1_h1 = init_weights([x_size, w1_h1_size])
w1_h2 = init_weights([w1_h1_size, w1_h2_size])
w1_o = init_weights([w1_h2_size, y_size])

##NN 2
w2_h1_size = 700
w2_h2_size = 100
w2_h1 = init_weights([x_size, w2_h1_size])
w2_h2 = init_weights([w2_h1_size, w2_h2_size])
w2_o = init_weights([w2_h2_size, y_size])

alpha = tf.Variable(tf.random_normal([1], mean=0.5, stddev=0.01, dtype=tf.float32, seed=RANDOM_SEED)) #mixing ratio

p_keep_input = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)



##model
pry_x_1 = model(x,w1_h1,w1_h2,w1_o,p_keep_input,p_keep_hidden)
pry_x_2 = model(x,w2_h1,w2_h2,w2_o,p_keep_input,p_keep_hidden)
py_x = pry_x_1*alpha+pry_x_2*(1-alpha)

#training
global_step = tf.Variable(0, trainable=False)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x,y_))
train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-5,use_locking=False, name='Adam').minimize(cost,global_step=global_step)
predict_op = tf.argmax(py_x, 1)

k=1
MAX_ITER=901

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
start_time = dt.datetime.now()


with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()
    # restoring from previous checkpoint (keep in mind that layers shouldnt change)
    # restorer = tf.train.Saver()
    # iter = 400
    # restorer.restore(sess,'checkpoint_'+str(iter)+'.chk')
    for i in xrange(MAX_ITER):
        for start, end in zip(range(0, len(trX), 256), range(256, len(trX)+1, 256)):
            sess.run(train_op, feed_dict={x: trX[start:end], y_: trY[start:end],
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})
        if i%100==0:
            saver = tf.train.Saver([w1_h1,w1_h2,w1_o,w2_h1,w2_h2,w2_o,alpha])
            saver.save(sess, './checkpoints/checkpoint_'+str(i)+'.chk') #nn 1
        #for prediction dropout rate should be set to 0, ie keep rate to 1
        #
        print(i, np.mean(np.argmax(teY, axis=k) ==
                         sess.run(predict_op, feed_dict={x: teX, y_: teY,
                                                         p_keep_input: 1.0,
                                                         p_keep_hidden: 1.0})))
        print(i,np.mean(np.argmax(trY,axis=k)==sess.run(predict_op,feed_dict={x:trX,y_:trY,p_keep_input:1.0,p_keep_hidden:1.0})))
print "\n time taken "+str(dt.datetime.now()-start_time)
