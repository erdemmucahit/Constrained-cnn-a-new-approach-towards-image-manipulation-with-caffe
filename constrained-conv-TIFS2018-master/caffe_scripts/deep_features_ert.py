"""
             _       _                          
  _ __ ___  (_) ___ | | multimedia &                
 | '_ ` _ \ | |/ __|| | information
 | | | | | || |\__ \| | security
 |_| |_| |_||_||___/|_| lab
 __________________________________________________
|__________________________________________________|

 misl.ece.drexel.edu

 DEPT. OF ELECTRICAL & COMPUTER ENGINEERING
 DREXEL UNIVERSITY
"""


import os,math
import sys
import caffe
import cv2
import numpy as np
from numpy import *
import lmdb
from sklearn import svm
caffe_root = '/home/erdem/anaconda2/lib/python2.7/site-packages/caffe'



MODEL_FILE = '/home/erdem/constrained-conv-TIFS2018-master/caffe_scripts/deploy_mislnet.prototxt'  #Make sure about the path where you saved your prototxt file
PRETRAINED = '/home/erdem/constrained-conv-TIFS2018-master/caffe_scripts/mislnet_six_classes.caffemodel'  #Make sure about the path where you saved your caffe model


net = caffe.Net(MODEL_FILE, PRETRAINED,caffe.TEST)
caffe.set_device(0)
caffe.set_mode_gpu()


lmdb_env = lmdb.open('/home/erdem/data/tifs_dresden_data/train_lmdb/')

lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
count = 0
correct = 0

p_label = []
t_label = []
feat_tr = []
soft = []

#In case you need to train & test the ERT classifier with smaller patches, e.g., 64x64
########################################
n = 64
i1_start = (256-n)/2  
i1_stop = i1_start + n 
i2_start = (256-n)/2
i2_stop = i2_start + n 
########################################

idx = 0


for key, value in lmdb_cursor:
	print "Count:"
	print count
	count = count + 1
	datum = caffe.proto.caffe_pb2.Datum()
	datum.ParseFromString(value)
	t_label.append(int(datum.label))
	image = caffe.io.datum_to_array(datum)
	image = image.astype(np.uint8)
	im = image#[0,i1_start:i1_stop, i2_start:i2_stop] #use indexing for retaining central patches of smaller sizes
	out = net.forward_all(data=np.asarray([im]))
	p_label.append(out['prob'][0].argmax(axis=0))
	soft.append(out['prob'][0]) #get softmax output if needed
	feat_tr.append(net.blobs['fc7_res'].data[0].tolist())
	print("Label is class " + str(int(datum.label)) + ", predicted class is " + str(out['prob'][0].argmax(axis=0)))





lmdb_env = lmdb.open('/home/erdem/data/tifs_dresden_data/test_lmdb/')


lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
count = 0
correct = 0

p_y = []
t_y = []
feat_tt = []



for key, value in lmdb_cursor:
	print "Count:"
	print count
	count = count + 1
	datum = caffe.proto.caffe_pb2.Datum()
	datum.ParseFromString(value)
	t_y.append(int(datum.label))
	image = caffe.io.datum_to_array(datum)
	image = image.astype(np.uint8)
	im = image#[0,i1_start:i1_stop, i2_start:i2_stop]
	out = net.forward_all(data=np.asarray([im]))
	p_y.append(out['prob'][0].argmax(axis=0))
	feat_tt.append(net.blobs['fc7_res'].data[0].tolist())
	print("Label is class " + str(int(datum.label)) + ", predicted class is " + str(out['prob'][0].argmax(axis=0)))



#####################################Softmax confusion matrix and testing accuracy
from sklearn.metrics import confusion_matrix

cmat = confusion_matrix(t_y, p_y)
nbr = cmat.sum(1)
nbr = np.array(nbr, dtype = 'f')
M = cmat/nbr
np.set_printoptions(suppress=True)
M = np.around(M*100, decimals=2) #Set confusion matrix to two decimals

binary = [t_y[i]==p_y[i] for i in range(len(p_y))]
acc_sft = binary.count(True)/float(count)

print 'The softmax-based CNN testing accuracy is ' + str(acc_sft)



#####################################Extremely Randomized Trees (ERT) confusion matrix and testing accuracy
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(n_estimators=800, max_depth=None, min_samples_split=3, random_state=0, n_jobs=8)
et.fit(feat_tr, t_label)
cet = et.predict(feat_tt)

from sklearn.metrics import confusion_matrix

cmat5 = confusion_matrix(t_y, cet.tolist())
nbr5 = cmat5.sum(1)
nbr5 = np.array(nbr5, dtype = 'f')
M5 = cmat5/nbr5
np.set_printoptions(suppress=True)
M5 = np.around(M5*100, decimals=2) ##Set confusion matrix to two decimals

binary = [t_y[i]==cet.tolist()[i] for i in range(len(t_y))]
acc_ert = binary.count(True)/float(count)

print 'The ERT-based CNN testing accuracy is ' + str(acc_ert)



