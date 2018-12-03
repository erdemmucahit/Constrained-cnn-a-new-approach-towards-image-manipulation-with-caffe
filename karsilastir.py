#! /usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import shutil
okunan=open("/home/erdem/Masaüstü/imglst_test.dmp","r")
b=okunan.read()
print type(b)
print(b)
print("-------")
b=b.split("\n")
listOfImageNames=[]
for a in b:
    try:
        listOfImageNames.append(a.split("aS")[1].split("'")[1])
    except:
        pass
print type(listOfImageNames)
print listOfImageNames
DIRECTORY="/home/erdem/shared_folder/Dresden/"
count=1
for i in os.listdir(DIRECTORY):
    if os.path.splitext(i)[1].lower() in ('.jpg', '.jpeg'):
        a=os.path.join(i)
    for x in listOfImageNames:
        if a==x:
            shutil.copy(DIRECTORY+a, "/home/erdem/Masaüstü/yeni_fotolar")
            print "match", count
            count+=1
        else:
            pass   
okunan.close()
 docker cp Masaüstü/yeni_fotolar/. tubitak/ccnn:~/Masaüstü

