# Source code for MISLnet: "Constrained Convolutional Neural Networks: A New approach Towards General Purpose Image Manipulation Detection"
by Belhassen Bayar and Matthew C. Stamm <br/>
Deparment of Electrical and Computer Engineering <br/>
Drexel University - Philadelphia, PA, USA

## About

This repository contains pycaffe scripts for general purpose image manipulation detection using constrained convolutional neural network (CNN), i.e., MISLnet architecture. The functions within this repository are described below:

- **training_mislnet.py:** Train a CNN architecture associated with a constrained convolutional layer. This script uses caffe prototxt files, namely `solver_mislnet.prototxt` (training hyper-parameters) and `train_val_mislnet.prototxt` (CNN layers).

- **testing_mislnet.py**: Test a trained CNN using the caffe MISLnet model `mislnet_six_classes.caffemodel` that we provide along with `deploy_mislnet.prototxt` file. The MISLnet caffe model has been trained with 1.2M image patches of size 256x256x1 pixels (green channel) created by the five different types of image processing defined in the paper. Class 0 corresponds to unaltered patches and the remaining five classes are labeled with respect to the order we followed in Table III of the paper. Also, this script returns the confusion matrix of the CNN.

- **create_forgery_lmdb.py:** Create lmdb testing data using the list of 334 images `imglst_test.dmp` from Dresden Image Database that we selected to create 50K testing image patches as decribed in the paper. This code creates 256x256x1 (green channel) image patches along with their corresponding altered patches using the five different types of editing operations used in the paper. The testing `test_lmdb` data will be saved under the root folder `caffe_scripts`. The list of images in `imglst_test.dmp` can be excluded from the Dresden Image Database to create the training and validation data using the same code.

- **deep_features_ert.py:** Extract training and testing deep features (second fully-connected layer after activation) from MISLnet caffe model `mislnet_six_classes.caffemodel` using the `deploy_mislnet.prototxt` file. Next, train and test an extremely randomized trees (ERT) classifier using the extracted deep features. This script returns also the testing accuracy along with the confusion matrix for both SoftMax-based and ERT-based CNNs.



## Installation

1. Download the repository .zip  or clone via git. 
2. Extract `caffe_scripts` folder.

## Requirements

python2.7 with packages:

- [caffe](https://github.com/BVLC/caffe)
- lmdb
- opencv2
- numpy
- scipy
- sklearn
- random
- cPickle

## Examples of running python scripts

Most of the provided scripts and files contain paths to data or prototxt file. <b/>
Make sure to add the appropriate paths in every file as needed.

Train MISLnet architecture by running the following command line under caffe_scripts directory:
```
python training_mislnet.py 2>&1 | tee -a mislnet_train_date.log
```
Create image manipulation testing lmdb data by running the following command line under caffe_scripts directory:
```
python create_forgery_lmdb.py 2>&1 | tee -a data_creation.log
```
## Useful tips

##### Running scripts in the background mode
If you would like to run your code in the background mode in order to close your terminal or SSH session, you should follow these steps:
1. Press ``Ctrl+Z``: This will temporarly suspend your program.
2. ``bg`` (Followed by "Enter" button): This will resume running your program in the background mode.
3. ``disown -h`` (Followed by "Enter" button): This removes the process from the shell's job control, but it still leaves it connected to the terminal.

Later you can check the progress of your program by accessing the log file (e.g., mislnet_train_date.log) using your vi/vim editor.

##### Display validation accuracy
To display the validation accuracy recorded in the log file while training the CNN, you can use the following command line:
```
grep -nrH ' accuracy = ' mislnet_train_date.log 
```

##### Testing using the caffe c++ code
After you saved your caffe model you can run the testing without the deploy file using the caffe c++ code under the caffe root folder as follow:
```
./build/tools/caffe test --model=/path/to/caffe_scripts/train_val_mislnet.prototxt --weights=/path/to/caffe_scripts/mislnet_six_classes.caffemodel -gpu 0 -iterations 2000
```
Make sure to set the correct path to the `test_lmdb` data for the 'TEST' phase in your `train_val_mislnet.prototxt` file. The argument 2000 corresponds to the number of testing iterations given the batch size you chose in your prototxt file.


## Citing this Code

If you are using this code for scholarly or academic research, please cite this paper:

Belhassen Bayar, and Matthew C. Stamm. "Constrained Convolutional Neural Networks: A New approach Towards General Purpose Image Manipulation Detection." IEEE Transactions on Information Forensics and Security (2018).

bibtex:

```
@article{bayar2018mislnet,
  title={Constrained Convolutional Neural Networks: A New Approach Towards General Purpose Image Manipulation Detection},
  author={Bayar, Belhassen and Stamm, Matthew C},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={13},
  number={11},
  pages={2691--2706},
  year={2018},
  publisher={IEEE}
}
```