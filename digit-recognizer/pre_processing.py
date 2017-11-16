"""
This file will have utility functions for all pre processing functionality
 Also have the code to extract data and labels from data set and save it to np array.
Also there is subroutine which will save the image as .jpg format in TRAIN-IMAGE and TEST-IMAGE directory
"""
### Import all the modules
from sklearn.utils import resample
from zipfile import ZipFile
import gzip
import os
from urllib.request import urlretrieve
import csv
import numpy as np
from PIL import Image
import tensorflow as tf
from scipy.misc import imsave
from sklearn.model_selection import train_test_split


SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'mnist'
TRAIN_IMAGE = 'mnist/train-image/'
TEST_IMAGE = 'mnist/test-image/'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10

def helpi():
    print('help')
    return

# Download script to download datasets if not already
"""
code to download the nmist data set if not already downloded Data set will be downloded at "currnt dirctory/mnist" dirctory unless we modify the WORK_DIRECTORY.
"""
def maybe_download(filename):
  print('downloading',filename)
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
    print('...')
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath

"""
Extract routine to extract data and labels from dowloded files and save it into numpy array
"""
def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].
  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    
    # Load image data as 1 dimensional array
    data = np.array(data, dtype=np.float32).flatten()
    # reshape the image into [image imdex, image_size*image_size*channels]
    data = data.reshape(num_images,IMAGE_SIZE*IMAGE_SIZE*1)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    labels = labels.reshape(num_images,1)
  return labels

# Call download routing to download the mnist data files
def get_raw_data():
    print('get_data')
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract train_data_filename and save into np arrays upto 60000 images.
    train_data = extract_data(train_data_filename, 60000)
#    display(train_data.shape)
    # Extract train_labels_filename and save into np arrays upto 60000 images.
    train_labels = extract_labels(train_labels_filename, 60000)
#    display(train_labels.shape)

    # Extract test_data_filename and save into np arrays upto 10000 images.
    test_data = extract_data(test_data_filename, 10000)
    # Extract test_labels_filename and save into np arrays upto 10000 images.
    test_labels = extract_labels(test_labels_filename, 10000)

#    display(test_data[0:])
    # Wait until you see that all features and labels have been uncompressed.
    print('All features and labels uncompressed.')
   # data_save(train_data, train_labels, test_data, test_labels)
    
    return train_data, train_labels, test_data, test_labels

"""
Save the digit in .jpg format in TRAIN-IMAGE and TEST-IMAGE directory Also creating a .csv file to save lables
"""
def data_save(train_data, train_labels, test_data, test_labels):
    if not os.path.isdir(TRAIN_IMAGE):
       os.makedirs(TRAIN_IMAGE)

    if not os.path.isdir(TEST_IMAGE):
       os.makedirs(TEST_IMAGE)

    # process train data
    with open("mnist/train-labels.csv", 'w') as csvFile:
      writer = csv.writer(csvFile, delimiter=',', quotechar='"')
      for i in range(len(train_data)):
        img = train_data[i]
        img = img.reshape(IMAGE_SIZE, IMAGE_SIZE)
       # print (img.shape)
        imsave(TRAIN_IMAGE + str(i) + ".jpg", img)
        writer.writerow(["train-images/" + str(i) + ".jpg", train_labels[i]])
    print('training-data saving done')
    # repeat for test data
    with open("mnist/test-labels.csv", 'w') as csvFile:
      writer = csv.writer(csvFile, delimiter=',', quotechar='"')
      for i in range(len(test_data)):
        img = test_data[i]
        img = img.reshape(IMAGE_SIZE, IMAGE_SIZE)
        imsave(TEST_IMAGE + str(i) + ".jpg", img)
        writer.writerow(["test-images/" + str(i) + ".jpg", test_labels[i]])
    print('test-data saving done')
    
### Label encoding routine.
def label_encoding(x):
    y = np.zeros((len(x), 10))
    for i in range(len(x)):
        y[i,x[i]] = 1
    return y

"""
This routine is to get the training, cross validation and test data as well as encoded label 
from mnist dataset
"""
def get_processed_data():
    print ('data processing')
    train_data, train_labels, test_data, test_labels = get_raw_data()
    train_label_encoded = label_encoding(train_labels)
    test_label_encoded = label_encoding(test_labels)
    # Get randomized datasets for training and validation
    train_features, valid_features, train_labels, valid_labels = train_test_split(
        train_data,
        train_label_encoded,
        test_size=0.05,
        random_state=0)
#    display(train_features)

    print('Training features and labels randomized and split.')
#    display(test_label_encoded)
    return train_features, valid_features, train_labels, valid_labels, test_data, test_label_encoded

