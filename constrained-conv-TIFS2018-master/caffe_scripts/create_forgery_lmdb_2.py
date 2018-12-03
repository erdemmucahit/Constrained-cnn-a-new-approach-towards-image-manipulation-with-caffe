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
# coding=utf-8

import os
from os import listdir
from random import shuffle
import cv2
from numpy import *
import numpy as np


import lmdb
import caffe
import scipy.io as sio

from scipy import misc
import cPickle



if __name__ == '__main__':
	os.chdir('/home/erdem/constrained-conv-TIFS2018-master/caffe_scripts')
	ll = cPickle.load(open('imglst_test_2.dmp', 'r')) #load the list of images that have been used for the testing. These images have never been used for training
	print(ll)
	os.chdir('/home/erdem/foto_makine_test') #Change work directory to where you saved your Dresden images
	n = 1280 #cropping height 256x5
	m= 1280 # cropping width 256x5
	
	X = np.zeros((1500,1,256,256), dtype=np.uint8) #Initialize image data with zeros 
	y = np.zeros(1500, dtype=np.int64) #Initialize image labels with zeros
	count = 0
	k = 0
	for p in ll:


		img=cv2.imread(p)
		
		k+= 1
		i1_start = (img.shape[0]-n)/2 #Find the coordinates of the central 1280f1280 sub-region
		i1_stop = i1_start + n
		i2_start = (img.shape[1]-m)/2
		i2_stop = i2_start + m 
		img = img[i1_start:i1_stop, i2_start:i2_stop,:] #Retain the central 1280x1280 sub-region to create 25 central patches later
		cv2.imwrite('img_CV2_70.jpg', img[:,:,1], [int(cv2.IMWRITE_JPEG_QUALITY), 70]) #JPEG QF=70
		tmprs = misc.imresize(img[:,:,1], 1.5) #resampling with scaling 150% of the green channel
		i1_start = (tmprs.shape[0]-n)/2 #Find the new coordinates of the central 1280x1280 sub-region 
		i1_stop = i1_start + n
		i2_start = (tmprs.shape[1]-m)/2
		i2_stop = i2_start + m
		tmprs = tmprs[i1_start:i1_stop, i2_start:i2_stop] #Retain the central 1280x1280 image patch
		tmpj= cv2.imread('img_CV2_70.jpg')
		os.remove('img_CV2_70.jpg') #Delete the saved JPEG image from dresden images folder
		tmp = img[:(img.shape[0]/256)*256,:(img.shape[1]/256)*256,1] #Retain the green channel
		tmprs = tmprs[:(tmprs.shape[0]/256)*256,:(tmprs.shape[1]/256)*256]
		tmpj = tmpj[:(tmpj.shape[0]/256)*256,:(tmpj.shape[1]/256)*256,1] #Retain the green channel
		del(img)
		tmpm = cv2.medianBlur(tmp,5) #Median filtering with 5x5 kernel
		tmpg = cv2.GaussianBlur(tmp,(5,5),0) #Gaussian blur with default sigma=1.1 and kernel size 5x5
		awgn = 2.0*np.random.randn(tmp.shape[0],tmp.shape[1]) #Additive While Gaussian Noise with sigma=2
		tmpw = (tmp+awgn)
		tmpw = np.clip(tmpw,0,255) #Keep image pixel values within [0,255] range
		tmpw = tmpw.astype(np.uint8)
		vblocks = np.vsplit(tmp, tmp.shape[0]/256) #split image patch into vertical blocks
		shuffle(vblocks)
		vblocks = vblocks[:len(vblocks)]
		imcount = 0
		for v in vblocks:
			hblocks = np.hsplit(v, v.shape[1]/256) #split each vertical block into horizantal blocks
			shuffle(hblocks)
			hblocks = hblocks[:len(hblocks)]
			for h in hblocks:
				X[count-1] = h.reshape((1,1,256,256))
				y[count-1] = 0
				imcount += 1
				if count == 50000:
					break
				count += 1
			if count == 50000:
				break
		print 'OR ' + str(imcount)
		# DONT DO REST
		continue

		if count == 50000:
			break
		vblocks = np.vsplit(tmpm, tmpm.shape[0]/256)
		shuffle(vblocks)
		vblocks = vblocks[:len(vblocks)]
		imcount = 0
		for v in vblocks:
			hblocks = np.hsplit(v, v.shape[1]/256)
			shuffle(hblocks)
			hblocks = hblocks[:len(hblocks)]
			for h in hblocks:
				X[count-1] = h.reshape((1,1,256,256))
				y[count-1] = 1
				imcount += 1
				if count == 50000:
					break
				count += 1
			if count == 50000:
				break
		print 'MF ' + str(imcount)
		if count == 50000:
			break
		vblocks = np.vsplit(tmpg, tmpg.shape[0]/256)
		shuffle(vblocks)
		vblocks = vblocks[:len(vblocks)]
		imcount = 0
		for v in vblocks:
			hblocks = np.hsplit(v, v.shape[1]/256)
			shuffle(hblocks)
			hblocks = hblocks[:len(hblocks)]
			for h in hblocks:
				X[count-1] = h.reshape((1,1,256,256))
				y[count-1] = 2
				imcount += 1
				if count == 50000:
					break
				count += 1
			if count == 50000:
				break
		print 'GB ' + str(imcount)
		if count == 50000:
			break
		vblocks = np.vsplit(tmpw, tmpw.shape[0]/256)
		shuffle(vblocks)
		vblocks = vblocks[:len(vblocks)]
		imcount = 0
		for v in vblocks:
			hblocks = np.hsplit(v, v.shape[1]/256)
			shuffle(hblocks)
			hblocks = hblocks[:len(hblocks)]
			for h in hblocks:
				X[count-1] = h.reshape((1,1,256,256))
				y[count-1] = 3
				imcount += 1
				if count == 50000:
					break
				count += 1
			if count == 50000:
				break
		print 'WGN ' + str(imcount)
		if count == 50000:
			break
		vblocks = np.vsplit(tmprs, tmprs.shape[0]/256)
		shuffle(vblocks)
		vblocks = vblocks[:len(vblocks)]
		imcount = 0
		for v in vblocks:
			hblocks = np.hsplit(v, v.shape[1]/256)
			shuffle(hblocks)
			hblocks = hblocks[:len(hblocks)]
			for h in hblocks:
				X[count-1] = h.reshape((1,1,256,256))
				y[count-1] = 4
				imcount += 1
				if count == 50000:
					break
				count += 1
			if count == 50000:
				break
		print 'RS ' + str(imcount)
		if count == 50000:
			break
		vblocks = np.vsplit(tmpj, tmpj.shape[0]/256)
		shuffle(vblocks)
		vblocks = vblocks[:len(vblocks)]
		imcount = 0
		for v in vblocks:
			hblocks = np.hsplit(v, v.shape[1]/256)
			shuffle(hblocks)
			hblocks = hblocks[:len(hblocks)]
			for h in hblocks:
				X[count-1] = h.reshape((1,1,256,256))
				y[count-1] = 5
				imcount += 1
				if count == 50000:
					break
				count += 1
			if count == 50000:
				break
		print 'JPG ' + str(imcount)
		print 'image ' + str(k) + ' out of ' + str(len(ll)) + ' images is processed and the number of patches is ' + str(count)
		if count == 50000:
			break	
	
	from sklearn.utils import shuffle
	X, y = shuffle(X, y, random_state=0) #Shuffling data and labels with the same order. This is optional for testing and validation datasets
	
	
	os.chdir('/home/erdem/test_lmdb_file3/') #Change the work directory under where you are going to save the test_lmdb data
	
	N = X.shape[0]
	
	map_size = X.nbytes * 10
	
	env = lmdb.open('test_lmdb', map_size=map_size) #Create test_lmdb folder
	
	with env.begin(write=True) as txn:
		for i in range(N):
			datum = caffe.proto.caffe_pb2.Datum()
			datum.channels = X.shape[1]
			datum.height = X.shape[2]
			datum.width = X.shape[3]
			datum.data = X[i].tobytes()
			datum.label = int(y[i])
			str_id = '{:08}'.format(i)
			txn.put(str_id.encode('ascii'), datum.SerializeToString())
			print i+1
