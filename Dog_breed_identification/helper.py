### Import all relevent lib
import csv
import numpy as np
from PIL import Image
import os, os.path
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 64,64
CHANNEL = 3
SIZE = 64
PIXEL_DEPTH = 255
### Dir
LABELS = 'data/labels.csv'
DATA_FILE = 'data/data.csv'
TRAIN_IMAGE_DIR = 'data/train/'
TEST_IMAGE_DIR = 'data/test/'

def load_image_in_array( indirname ) :
    data = []
    with open("./data/tmp.csv", 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='"')
    for root, dirs, fileList in os.walk(indirname, topdown=True):
         for fname in fileList:
#            for i in range(5):
#                if count > 5 :
#                    break
            f = os.path.join(root,fname)
            img = Image.open(f)
            img.load()
            img = img.resize(IMAGE_SIZE, Image.ANTIALIAS)
            tmp = np.asarray( img, dtype="int32" )
            data.append(tmp)
            with open("./data/tmp.csv", 'a', newline='') as csvFile:
                writer = csv.writer(csvFile, delimiter=',', quotechar='"')
                writer.writerow([str(fname)])

#            display(img)
#            display(len(data))
            
    data = np.array(data)       
    return data 

def save_feature_in_csv(outfile, indata):
    read = list(csv.reader(open("data/tmp.csv", "r"), delimiter=" "))

    with open(outfile, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='"')
        writer.writerow(["Index" , "Feature", "Name"]) 
    with open(outfile, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='"')
        for i in range(indata.shape[0]):
            a = indata[i]
            b = str(read[i])
            read[i] = b[2:len(b)-6]
 
            writer.writerow([i+1,a,str(read[i])])


def label_encoding(labels):
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)

    label_dict = dict(zip(labels, integer_encoded))
    #for i in range(len(integer_encoded)):
    #    label_dict.update({integer_encoded[i]:labels[i]})
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_label_encoded = onehot_encoder.fit_transform(integer_encoded)
    
    print(onehot_label_encoded[0])
    return onehot_label_encoded, label_dict

def feature_scaling(values):
    # Values are rescaled from [0, 255] down to [-0.5, 0.5]
    for i in range(values.shape[0]):
        values[i] = (values[i] - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    return values

def organise_data():
    # load the image data in to np array of size (no of image, 32, 32, 3)
    data = load_image_in_array(TRAIN_IMAGE_DIR)
    #display(data.shape)
    
    data = np.array(data, dtype=np.int32).flatten()
    #display(data.shape)
    # reshape the image into [image index, image_size*image_size*channels]
    jpgCounter = len(glob.glob1(TRAIN_IMAGE_DIR,"*.jpg"))
    
    data = data.reshape(jpgCounter,SIZE*SIZE*CHANNEL)
    #display(data.shape)
    
    # save the data in /data/data.csv file
    save_feature_in_csv(DATA_FILE, data)
    
    return data

def get_feature_label():
    labels = np.loadtxt(open(LABELS, "rb"), dtype=str, delimiter=",", skiprows=1, usecols = 1)
    features = organise_data()
    features = features.reshape(features.shape[0], SIZE, SIZE, CHANNEL)
    #features = np.loadtxt(open("data/data.csv", "rb"), dtype=int, delimiter=",", skiprows=1)
    display(labels.shape)
    onehot_label_encoded, label_dict = label_encoding(labels)
    scaled_features = feature_scaling(features)
    display(onehot_label_encoded.shape)
    return scaled_features, onehot_label_encoded, label_dict

def get_train_cv_test_data():
    features, labels, label_dict = get_feature_label()
    #features = features.reshape(features.shape[0], SIZE, SIZE, CHANNEL)
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=0.01,
        random_state=42)
    train_features, valid_features, train_labels, valid_labels = train_test_split(
        train_features,
        train_labels,
        test_size=0.1,
        random_state=42)
    
    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels, label_dict


