"""
This is a script for finding the best performing neural network architecture among
a set of potential architectures. While you can run this script directly from the
command line, the idea here is to have a Bayesian model select the architecture
most likely to improve on the past experiences. Through many repeated
experiments the Bayesian model finds a high-performing network
and the space weather person is responsible for interpreting the
best performing network.

The network architectures searchable by this script follow
the form:

INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC

Unpacking this statement, the architecture starts with the inputs, then
passes through a convolutional layer with a rectified linear unit N times.
Next the layers are max pooled. The combination of convolutions
and pooling is repeated M times, before passing through K fully connected
layers having a rectified linear unit activation. The final layer of the
nework is a fully connected layer having a single output.

Layer parameters can be specified via the command line.
Each of the parameters have defaults so you can just start the script and watch it
train. If you want to specify parameters from the command line, they are specified
positionally, so to run with the default parameters you would run:
`python architectures.py 2 2 1 8 1 1 1 2 2 1 4 4 4 1 .3 128`

Before running the script, we recommend you start the tensorboard server so you
can track the progress.

`tensorboard --logdir=/tmp/sdo_128`

"""

#####################################
#        Importing Modules          #
#####################################

# Neural network specification
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.models import Model
from keras import backend as K

# Linear algebra library within Python
import numpy as np

# Deep learning training library
from keras.callbacks import TensorBoard

# Utilities for this script
import os
import random
import datetime
import argparse
import sys

#####################################
#        Specifying Network         #
#####################################

"""
These parameters specify the network architecture and are set from the
command line either by you (the user) or by a program that is
searching/optimizing the structure of the architecture.

You can change these values from the command line, or you can
modify the soure code to have hard-coded values. We generally
recommend you use the command line as stated above.
"""
arguments = [2, 2, 1, 8, 1, 1, 1, 2, 2, 1, 4, 4, 4, 1, .3, 128]  # Defaults
if len(sys.argv) > 2:
    arguments = sys.argv[1:]

# Set all pooling parameters to 1 to skip pooling layer
pool_1_width = int(arguments[0])  # [1,*2*,4,8,16,32,64,128]
pool_1_height = int(arguments[1])  # [*1*,2,4,8,16,32,64,128]
pool_1_stride = int(arguments[2]) # [*1*,2,4]

# Set conv_1_channels to 0 to not include this layer
conv_1_channels = int(arguments[3])  # [0,1,2,3,4,5,6,7,*8*]
conv_1_width = int(arguments[4])  # [*1*,2,3,4,5,6,7]
conv_1_height = int(arguments[5])  # [*1*,2,3,4,5,6,7]
conv_1_stride = int(arguments[6])  # [*1*,2,4]
conv_1_activation = "relu" # Not available initially

# Set all pooling parameters to 1 to skip pooling layer
pool_2_width = int(arguments[7])  # [1,*2*,4,8,16,32,64,128]
pool_2_height = int(arguments[8])  # [1,*2*,4,8,16,32,64,128]
pool_2_stride = int(arguments[9]) # [*1*,2,4]

# Set conv_2_channels to 0 to not include this layer
conv_2_channels = int(arguments[10])  # [1,2,3,*4*,5,6,7,8]
conv_2_width = int(arguments[11])  # [1,2,3,*4*,5,6,7]
conv_2_height = int(arguments[12])  # [1,2,3,*4*,5,6,7]
conv_2_stride = int(arguments[13])  # [*1*,2,4]
conv_2_activation = "relu" # Not available initially

dropout_rate = float(arguments[14])  # [0,.1,.2,*.3*,.4,.5,.6,.7]

dense_1_count = int(arguments[15])  # [4,8,16,32,64,*128*,256]
dense_1_activation = "relu" # Not available for search initially

# Final output for regression
dense_2_count = 1
dense_2_activation = "linear"


#####################################
#         SPECIFYING DATA           #
#####################################

data_directory = "/home/smcgregor/projects/solar-forecast/datasets/sdo_128/bin/"
tensorboard_log_data_path = "/tmp/sdo_128"
seed = 0
random.seed(seed)
input_channels = 8
input_width = 128
input_height = 128
input_image = Input(shape=(input_width, input_height, input_channels))
x = input_image

#####################################
#     Constructing Architecture     #
#####################################

if pool_1_width != 1 or pool_1_height != 1 or pool_1_stride != 1:
    input_width = (input_width - pool_1_width)/pool_1_stride + 1
    input_height = (input_height - pool_1_height)/pool_1_stride + 1
    x = MaxPooling2D((pool_1_width, pool_1_height), padding='same', strides=pool_1_stride)(x)

if conv_1_channels is not 0:
    x = Conv2D(conv_1_channels, (conv_1_width, conv_1_height), activation=conv_1_activation, padding='same')(x)

if pool_2_width != 1 or pool_2_height != 1 or pool_2_stride != 1:
    x = MaxPooling2D((pool_2_width, pool_2_height), padding='same', strides=pool_2_stride)(x)

if conv_2_channels is not 0:
    x = Conv2D(conv_2_channels, (conv_2_width, conv_2_height), activation=conv_2_activation, padding='same')(x)
    
x = Flatten()(x)
x = Dropout(dropout_rate)(x)
x = Dense(dense_1_count, activation=dense_1_activation)(x)
prediction = Dense(1, activation=dense_2_activation)(x)

forecaster = Model(input_image, prediction)
forecaster.compile(optimizer='adadelta', loss='mean_absolute_error')

"""
Debugging code:
  Uncomment to plot the network architecture.
"""
#from keras.utils import plot_model
#plot_model(forecaster, to_file='model3.png', show_shapes=True)
#exit()

#####################################
#          LOADING DATA             #
#####################################

# get a directory listing of the sdo mnist data
filenames = os.listdir(data_directory)

# remove a random subset of the file name list and make them the test set
random.shuffle(filenames)
train_files = filenames[:-10]
test_files = filenames[-10:]

def sdo_summary(filename):
    """
    Get a scalar summary of the SDO disk data, used in prediction.
    """
    path = data_directory + filename
    data = np.memmap(path, dtype='uint8', mode='r', shape=(128,128,8))
    return np.mean(data)
    
def get_files(paths):
    """
    Wrap the data up for training/testing.
    todo: make this use an generator for efficiency.
    """

    def result_file_name(path):
        """
        Find the file result file for the current starting file. Return None
        if there is no result file.
        """
        current_index = int(path.split("_")[2].split(".")[0])  # sdo* -> ['sdo', 'multichannel', '201507010000.bin'] -> #####
        year = current_index / 100000000
        month = current_index % 100000000 / 1000000
        day = current_index % 1000000 / 10000
        next_date = datetime.date(year, month, day) + datetime.timedelta(days=1)
        part_1 = str(next_date.year)
        if next_date.month < 10:
            part_2 = "0" + str(next_date.month)
        else:
            part_2 = str(next_date.month)
        if day < 10:
            part_3 = "0" + str(next_date.day * 10000)
        else:
            part_3 = str(next_date.day * 10000)
        next_index_string = part_1 + part_2 + part_3
        next_file_name = "sdo_multichannel_" + next_index_string + ".bin"
        if os.path.isfile(data_directory + next_file_name):
            return next_file_name
        else:
            return None
    
    ret_x = []
    ret_y = []
    for idx, f in enumerate(paths):
        res_name = result_file_name(f)
        if not res_name:
            print "no result found: " + f
            continue
        data_x = np.memmap(data_directory + f, dtype='uint8', mode='r', shape=(128,128,8))
        ret_x.append(data_x[:].copy())
        ret_y.append(sdo_summary(res_name))
    return (np.asarray(ret_x), np.asarray(ret_y))

# pack the x_train from the train set
x_train, y_train = get_files(train_files)
    
x_train = x_train.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 128, 128, 8))  # adapt this if using `channels_first` image data format

#####################################
#    Optimizig the Neural Network   #
#####################################

forecaster.fit(x_train, (np.asarray([[1]*len(x_train)])).reshape(len(x_train),1),
               epochs=3,
               batch_size=1,
               shuffle=True,
               callbacks=[TensorBoard(log_dir=tensorboard_log_data_path)])
