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

RANDOM_SEED = 42 #keep changing this
tf.set_random_seed(RANDOM_SEED)

def init_weights(shape):
    #TODO implement Xavier initialization
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


<<<<<<< HEAD
def model(X, w_h1, w_h2, w_o, p_keep_input, p_keep_hidden): # this network is the same as the previous one except with an extra hidden layer + dropout
    X = tf.nn.dropout(X, p_keep_input)
    h1 = tf.nn.elu(tf.matmul(X, w_h1))
    h1 = tf.nn.dropout(h1, p_keep_hidden)
    h2 = tf.nn.elu(tf.matmul(h1, w_h2))
    h2 = tf.nn.dropout(h2, p_keep_hidden)
    #h3 = tf.nn.elu(tf.matmul(h2, w_h3))
    #h3 = tf.nn.dropout(h3, p_keep_hidden)
=======
def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden): # this network is the same as the previous one except with an extra hidden layer + dropout
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))
    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))
    h2 = tf.nn.dropout(h2, p_keep_hidden)
    # h3 = tf.nn.relu(tf.matmul(h2, w_h3))
    # h3 = tf.nn.dropout(h3, p_keep_hidden)
>>>>>>> 09a5c4e2d013cd77dc3a8f9d8c179b00385b7693
    return tf.matmul(h2, w_o)


''' Splitting data'''
with open('train_data_pickle', mode='rb') as f:
    trainf = pkl.load(f)
trainf = pd.DataFrame.as_matrix(trainf)
labels = np.int32(trainf[:,-1])
no_of_classes = 12
with open('ipca_500comp.pkl', mode='rb') as f:
# with open('pca_1024.pkl', mode='rb') as f:
    pca=pkl.load(f)
trainf1_raw = pca.transform(trainf[:,:-1])

<<<<<<< HEAD
ipca = IncrementalPCA(n_components=512,batch_size=1000)
trainf1 = ipca.fit_transform(trainf[:,:-1]) #in memory computation
#loading PCA
=======
def normalize(train):
    mean, std = train.mean(), train.std()
    train = (train - mean) / std
    # test = (test - mean) / std
    return train

trainf1 = normalize(trainf1_raw)

# ipca = IncrementalPCA(n_components=500,batch_size=1000)
# trainf1 = ipca.fit_transform(trainf[:,:-1]) #in memory computation
>>>>>>> 09a5c4e2d013cd77dc3a8f9d8c179b00385b7693
#trainf1 = trainf[:,:-1]

N = trainf1.shape[0]
M = trainf1.shape[1] #last col was label

labels_OH  = np.zeros([N,no_of_classes])
labels_OH[np.arange(N),labels] = 1

# NOTE
# 48% test - normalised pca 1024 -> 500 -> 200 -> 50 -> 12
# 48.6% test - normalised ipca 500 -> 200 -> 12

#adding a bias column
train = np.ones([N,M+1])
train[:,1:] = trainf1[:,:] #preprending the column of ones

#NOTE for using sparse_softmax_cross_entropy_with_logits, input should be labels and not labels_OH
# uncomment accordingly

#trX,teX,trY,teY = train_test_split(train,labels_OH,test_size=0.40,random_state=RANDOM_SEED)
<<<<<<< HEAD
trX,teX,trY,teY = train_test_split(train,labels_OH,test_size=0.40)

x_size = trX.shape[1]
y_size = trY.shape[1]
#y_size = 1
x = tf.placeholder(tf.float32,shape=[None,x_size])
y_ = tf.placeholder(tf.float32,shape=[None,y_size]) #one hot prediction
#y_ = tf.placeholder(tf.int32,shape=[None]) #label

#init weights of layers, w_h1 = hidden1, w_h2 = hidden2 and w_o = output
w_h1_size = 300
w_h2_size = 100
#w_h3_size = 50
w_h1 = init_weights([x_size, w_h1_size])
w_h2 = init_weights([w_h1_size, w_h2_size])
#w_h3 = init_weights([w_h2_size,w_h3_size])
w_o = init_weights([w_h2_size, y_size])

#dropout params NOTE : DROPOUT WILL NOT BE PERFORMED ON THE FINAL OUTPUT LAYER!
p_keep_input = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)

#learning_rate
global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(0.01,global_step=global_step,decay_rate=0.9,decay_steps=250,staircase=True)
py_x = model(x, w_h1, w_h2, w_o, p_keep_input, p_keep_hidden)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x,y_))
#cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(py_x,y_))
#train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
train_op = tf.train.AdadeltaOptimizer(learning_rate=0.001, rho=0.95, epsilon=1e-08, use_locking=False, name='Adadelta').minimize(cost)
train_op = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.99, beta2=0.999, epsilon=1e-2,use_locking=False, name='Adam').minimize(cost,global_step=global_step)
predict_op = tf.argmax(py_x, 1)

#k=0 #when using sparse_softmax_cross_entropy_with_logits
k=1  #when running one hot label
MAX_ITER = 2001
ID_NN = 1 #identity in stacking
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
        for start, end in zip(range(0, len(trX), 128), range(182, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={x: trX[start:end], y_: trY[start:end],
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})
        if i%100==0:
            saver = tf.train.Saver([w_h1,w_h2,w_o])
            saver.save(sess, './checkpoints/checkpoint_'+str(ID_NN)+'_'+str(i)+'.chk') #nn 1
        #for prediction dropout rate should be set to 0, ie keep rate to 1
        #
        print(i, np.mean(np.argmax(teY, axis=k) ==
                         sess.run(predict_op, feed_dict={x: teX, y_: teY,
                                                         p_keep_input: 1.0,
                                                         p_keep_hidden: 1.0})))
        print(i,np.mean(np.argmax(trY,axis=k)==sess.run(predict_op,feed_dict={x:trX,y_:trY,p_keep_input:1.0,p_keep_hidden:1.0})))
print "\n time taken "+str(dt.datetime.now()-start_time)
=======
# trX,teX,trY,teY = train_test_split(train,labels_OH,test_size=0.30,random_state=RANDOM_SEED)
from sklearn.cross_validation import KFold
kf=KFold(n=train.shape[0],n_folds=3,shuffle=True)
model_idx=0
for idx in kf:
    trX = train[idx[0]]
    trY = labels_OH[idx[0]]
    teX = train[idx[1]]
    teY = labels_OH[idx[1]]

    x_size = trX.shape[1]
    y_size = trY.shape[1]
    #y_size = 1
    x = tf.placeholder(tf.float32,shape=[None,x_size])
    y_ = tf.placeholder(tf.float32,shape=[None,y_size]) #one hot prediction
    #y_ = tf.placeholder(tf.int32,shape=[None]) #label

    #init weights of layers, w_h1 = hidden1, w_h2 = hidden2 and w_o = output
    w_h1_size = 200
    w_h2_size = 50
    w_h3_size = 50
    w_h1 = init_weights([x_size, w_h1_size])
    w_h2 = init_weights([w_h1_size, w_h2_size])
    # w_h3 = init_weights([w_h2_size,w_h3_size])
    w_o = init_weights([w_h2_size, y_size])

    #dropout params NOTE : DROPOUT WILL NOT BE PERFORMED ON THE FINAL OUTPUT LAYER!
    p_keep_input = tf.placeholder(tf.float32)
    p_keep_hidden = tf.placeholder(tf.float32)

    #learning_rate
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(0.01,global_step=global_step,decay_rate=0.9,decay_steps=250,staircase=True)
    py_x = model(x, w_h1, w_h2, w_o, p_keep_input, p_keep_hidden)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x,y_))
    #cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(py_x,y_))
    #train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,use_locking=False, name='Adam').minimize(cost,global_step=global_step)
    predict_op = tf.argmax(py_x, 1)

    #k=0 #when using sparse_softmax_cross_entropy_with_logits
    k=1  #when running one hot label
    MAX_ITER = 300

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    start_time = dt.datetime.now()

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # you need to initialize all variables
        tf.initialize_all_variables().run()
        # restoring from previous checkpoint (keep in mind that layers shouldnt change)
        # restorer = tf.train.Saver()
        # iter = 200
        # restorer.restore(sess,'checkpoint_'+str(iter)+'.chk')
        for i in xrange(MAX_ITER):
            for start, end in zip(range(0, len(trX), 256), range(256, len(trX)+1, 256)):
                sess.run(train_op, feed_dict={x: trX[start:end], y_: trY[start:end],
                                              p_keep_input: 0.8, p_keep_hidden: 0.5})
            # if i%100==0:
            #     saver = tf.train.Saver([w_h1,w_h2,w_o])
            #     # saver = tf.train.Saver([w_h1,w_h2,w_h3,w_o])
            #     saver.save(sess, 'checkpoint_'+str(i)+'.chk')
            #for prediction dropout rate should be set to 0, ie keep rate to 1
            #
            print('Model-'+str(model_idx))
            print(i,'Test', np.mean(np.argmax(teY, axis=k) ==
                             sess.run(predict_op, feed_dict={x: teX, y_: teY,
                                                             p_keep_input: 1.0,
                                                             p_keep_hidden: 1.0})))
            print(i,'Train',np.mean(np.argmax(trY,axis=k)==sess.run(predict_op,feed_dict={x:trX,y_:trY,p_keep_input:1.0,p_keep_hidden:1.0})))
        saver = tf.train.Saver([w_h1,w_h2,w_o])
        saver.save(sess,'models/model_'+str(model_idx)+'.chk')
        model_idx = model_idx+1
print("\n time taken "+str(dt.datetime.now()-start_time))
>>>>>>> 09a5c4e2d013cd77dc3a8f9d8c179b00385b7693
