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
import lmdb

#caffe_root = '/home/erdem/pytorch/build/caffe'
caffe_root="/home/erdem/anaconda2/lib/python2.7/site-packages/caffe"
print("*****")

if __name__ == '__main__':
        #caffe.set_device(0)
        #caffe.set_mode_gpu()
        #solver = caffe.SGDSolver('/home/belhassen/caffe_scripts/solver_mislnet.prototxt') #Make sure to add the correct path
	solver = caffe.SGDSolver('/home/erdem/constrained-conv-TIFS2018-master/caffe_scripts/solver_mislnet.prototxt')
        print("*****")
        for i in range(675200):
                sys.stdout.flush()
                tmp = solver.net.params['convF'][0].data*10000 #Scale by 10k otherwise you may encounter numerical issues while normalizing
                tmp[:,:,2,2] = 0 #Set central value of all filters to zero in order to exclude it in the normalization step
                tmp = tmp.reshape((3,1,1,25)) #Vectorize each convolutional filter for the element-wise division
                tmp = tmp/tmp.sum(3).reshape((3,1,1,1)) #Element-wise division by the sum
                tmp = tmp.reshape((3,1,5,5)) #Reshape back filters to the original dimension
                tmp[:,:,2,2] = -1 #Set central value of all filters to -1
                solver.net.params['convF'][0].data[...] = tmp
                solver.net.params['convF'][1].data[...] = 0
                solver.step(1)
                print solver.net.params['convF'][0].data[0].sum() #Sum of the first filter weights. The sum of all filter weights in each filter should be very close to zero
                print 'iteration ' + str(i+1) + ' is done'
