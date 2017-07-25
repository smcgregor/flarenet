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
train. If you want to specify parameters from the command line, they are specified with
flags. So to run with the default parameters you would run:
`python architectures.py -pool_1_width 2 -pool_1_height 2 -pool_1_stride 1 -conv_1_channels 8 -conv_1_width 1 -conv_1_height 1 -conv_1_stride 1 -pool_2_width 2 -pool_2_height 2 -pool_2_stride 1 -conv_2_channels 4 -conv_2_width 4 -conv_2_height 4 -conv_2_stride 1 -dropout_rate .3 -dense_1_count 128``

If you don't specify these parameters at the command line, then the default value
will be used.

Before running the script, we recommend you start the tensorboard server so you
can track the progress.

`tensorboard --logdir=/tmp/version1`

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
from corona_callbacks import CoronaCallbacks

# Utilities for this script
import os
import random
import datetime
import argparse
import sys
import psutil

# Library for parsing arguments
import argparse

# Uncomment to force training to take place on the CPU
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

parser = argparse.ArgumentParser(description='Train a neural network.')

parser.add_argument('ignore', metavar='N', type=str, nargs='*',
                    help='ignore this argument. It is used to accumulate positional arguments from SMAC')

# Set all pooling parameters to 1 to skip pooling layer
parser.add_argument('-pool_1_width', type=int, nargs="?", default=4)
parser.add_argument('-pool_1_height', type=int, nargs="?", default=4)
parser.add_argument('-pool_1_stride', type=int, nargs="?", default=4)

# Set conv_1_channels to 0 to not include this layer
parser.add_argument('-conv_1_channels', type=int, nargs="?", default=8)
parser.add_argument('-conv_1_width', type=int, nargs="?", default=4)
parser.add_argument('-conv_1_height', type=int, nargs="?", default=4)
parser.add_argument('-conv_1_stride', type=int, nargs="?", default=1)
conv_1_activation = "relu" # Not available initially

# Set all pooling parameters to 1 to skip pooling layer
parser.add_argument('-pool_2_width', type=int, nargs="?", default=4)
parser.add_argument('-pool_2_height', type=int, nargs="?", default=4)
parser.add_argument('-pool_2_stride', type=int, nargs="?", default=4)

# Set conv_2_channels to 0 to not include this layer
parser.add_argument('-conv_2_channels', type=int, nargs="?", default=8)
parser.add_argument('-conv_2_width', type=int, nargs="?", default=2)
parser.add_argument('-conv_2_height', type=int, nargs="?", default=2)
parser.add_argument('-conv_2_stride', type=int, nargs="?", default=1)
conv_2_activation = "relu" # Not available initially

parser.add_argument('-dropout_rate', type=float, nargs="?", default=.3)
parser.add_argument('-dense_1_count', type=int, nargs="?", default=16)
dense_1_activation = "relu" # Not available for search initially

# Final output for regression
dense_2_count = 1
dense_2_activation = "linear"

args = parser.parse_args()

#####################################
#         SPECIFYING DATA           #
#####################################

data_directory = "/data/sw/version1/x/"
results_path = "/data/sw/version1/y/Y_GOES_XRAY_201401.csv"
tensorboard_log_data_path = "/tmp/version1/"
seed = 0
random.seed(seed)
input_channels = 8
input_width = 1024
input_height = 1024
input_image = Input(shape=(input_width, input_height, input_channels))
validation_steps = 30
steps_per_epoch = 100
samples_per_step = 32  # batch size
epochs = 10
x = input_image

#####################################
#     Constructing Architecture     #
#####################################

print "constructing network in the Keras functional API"

if args.pool_1_width != 1 or args.pool_1_height != 1 or args.pool_1_stride != 1:
    x = MaxPooling2D((args.pool_1_width, args.pool_1_height), padding='same', strides=args.pool_1_stride)(x)

if args.conv_1_channels is not 0:
    x = Conv2D(args.conv_1_channels, (args.conv_1_width, args.conv_1_height), activation=conv_1_activation, padding='same')(x)

if args.pool_2_width != 1 or args.pool_2_height != 1 or args.pool_2_stride != 1:
    x = MaxPooling2D((args.pool_2_width, args.pool_2_height), padding='same', strides=args.pool_2_stride)(x)

if args.conv_2_channels is not 0:
    x = Conv2D(args.conv_2_channels, (args.conv_2_width, args.conv_2_height), activation=conv_2_activation, padding='same')(x)
    
x = Flatten()(x)
x = Dropout(args.dropout_rate)(x)
x = Dense(args.dense_1_count, activation=dense_1_activation)(x)
prediction = Dense(1, activation=dense_2_activation)(x)

forecaster = Model(input_image, prediction)
forecaster.compile(optimizer='adadelta', loss='mean_absolute_error')

print forecaster.summary()

# Do not allow a configuration with more than 150 million parameters
if forecaster.count_params() > 150000000:
    print "exiting since this network architecture will contain too many paramters"
    print "Result for SMAC: SUCCESS, 0, 0, 999999999, 0" #  todo: figure out the failure string within SMAC
    exit()

"""
Debugging code:
  Uncomment to plot the network architecture.
"""
#from keras.utils import plot_model
#plot_model(forecaster, to_file='model3.png', show_shapes=True)
#exit()

#####################################
#        GENERATING DATA            #
#####################################

print "defining data generators"

# get a directory listing of the sdo data
filenames = os.listdir(data_directory)
random.shuffle(filenames)
train_files = filenames[:-validation_steps]
test_files = filenames[-validation_steps:]

print "loading results file"
y_dict = {}
with open(results_path, "rb") as f:
    for line in f:
        split_y = line.split(",")
        y_dict[split_y[0]] = float(split_y[1])

def get_y(filename, y_dict=y_dict):
    """
    Get the true forecast result for the current filename.
    """
    split_filename = filename.split("_")
    k = split_filename[0] + "_" + split_filename[1]
    return y_dict[k]

#  Dictionary caching filenames to their normalized in-memory result
cache = {}
def generator(training=True):
    """
    Generate samples
    """

    def available_cache(training):
        """
        If there is enough main memory, add the object to the cache.
        Always add the object to the cache if it is in the validation
        set.
        """
        vm = psutil.virtual_memory()
        return vm.percent < 75 or not training

    if training:
        files = train_files
    else:
        files = test_files

    x_mean_vector = [2.2832, 10.6801, 226.4312, 332.5245, 174.1384, 27.1904, 4.7161, 67.1239]
    x_standard_deviation_vector = [12.3858, 26.1799, 321.5300, 475.9188, 289.4842, 42.3820, 10.3813, 72.7348]

    data_x = []
    data_y = []
    i = 0
    while 1:

        # The current file
        f = files[i]

        # Get the sample from the cache or load it from disk
        if f in cache:
            data_x_sample = cache[f][0]
            data_y_sample = cache[f][1]
        else:
            data_x_sample = np.load(data_directory + f)
            data_x_sample = ((data_x_sample.astype('float32').reshape((input_width*input_height, input_channels)) - x_mean_vector) / x_standard_deviation_vector).reshape((input_width, input_height, input_channels)) # Standardize to [-1,1]
            data_y_sample = get_y(f)

            if available_cache(training):
                cache[f] = [data_x_sample, data_y_sample]

        data_x.append(data_x_sample)
        data_y.append(data_y_sample)

        if i == len(files):
            i = 0
            random.shuffle(files)

        if samples_per_step == len(data_x):
            ret_x = np.reshape(data_x, (len(data_x), input_width, input_height, input_channels))
            ret_y = np.reshape(data_y, (len(data_y)))
            yield (ret_x, ret_y)
            data_x = []
            data_y = []

print "loading image dataset"

#####################################
#   Optimizing the Neural Network   #
#####################################

tensorboard_callbacks = TensorBoard(log_dir=tensorboard_log_data_path)
corona_callbacks = CoronaCallbacks("argument-search-results", args)

history = forecaster.fit_generator(generator(training=True),
                                   steps_per_epoch,
                                   epochs=epochs,
                                   validation_data=generator(training=False),
                                   validation_steps=validation_steps,
                                   callbacks=[tensorboard_callbacks, corona_callbacks])

# Loss on the training set
print history.history['loss']

# Loss on the validation set
if 'val_loss' in history.history.keys():
    print history.history['val_loss']

# Print the performance of the network for the SMAC algorithm
print "Result for SMAC: SUCCESS, 0, 0, %f, 0" % history.history['loss'][-1]
