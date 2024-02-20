import tensorflow as tf
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import Sequential, Model
from tensorflow.keras.activations import softmax

def get_image_path_and_id(img_names, img_path): 
    # Purpose: to get the FULL path of the images and their name id
    
    # first set two empty lists
    img = []
    img_id = []
    
    # for each image name, append the paths and ids
    for i in img_names:
        image_path = img_path+'/'+i
        img_id.append(i.split('.')[0])
        img.append(image_path)
        ## used to read the image directly but not good
        # img.append(load_image(image_path)) 
        
    return img, img_id

def load_image(image_path):
    # Purpose: to load, resize and preprocess images with InceptionV3 structure
    # this function refers https://www.tensorflow.org/tutorials/text/image_captioning
    
    # load and read RGB channels
    img = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3) 
    # print(type(img))
    
    # resize pictures to 299x299x3
    img = tf.image.resize(img, (299, 299))
    
    # preprocess, this would normalize the pixel values to (-2,0) or some other ranges
    img = tf.keras.applications.inception_v3.preprocess_input(img) 
    
    # return bothe the image array and image path
    return img, image_path

def load_cnn():
    # Purpose: create a tf.keras model where the output layer is the last convolutional layer in the InceptionV3
    
    # load model and include the top avg_pool layer because we want the final features to be in the shape of (None, 2048)
    image_model = tf.keras.applications.InceptionV3(weights='imagenet') 

    # the shape of input is (None, 299, 299, 3)
    new_input = image_model.input
    
    # the shape of the output of this layer is (None, 2048)
    hidden_layer = image_model.layers[-2].output

    # create the model
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    # image_features_extract_model.summary()
    
    return image_features_extract_model


def extract_features(img):
    # Purpose: create a dataset for one set of images (like train, test, dev, val) and extract features
    
    # create a dataset first
    image_dataset = tf.data.Dataset.from_tensor_slices(img) 
    
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel
    image_dataset = image_dataset.map(load_image, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    
    # the dataset is small so set the batch size as 25, I decided this personally
    batch_size = 25
    # this batch is only used for the feature calculation, and should be irrelevant to the training process of LSTM
    image_dataset = image_dataset.batch(batch_size)   
    
    # load CNN which is InceptionV3 here
    image_features_extract_model = load_cnn()
    
    # now extract features for the dataset and use this as the input for the LSTM
    # for img, path in image_dataset: ## if tqdm does not work
    for img, path in tqdm(image_dataset):   
        batch_features = image_features_extract_model(img)
        # print('batch features shape: ',batch_features.shape)
        # batch features shape: (25, 2048)

        for batch_feature, path_of_feature in zip(batch_features, path):
            # save each image's feature to disk 
            np.save(path_of_feature.numpy().decode("utf-8"), batch_feature.numpy())

            
def combine_features(mode, img, img_path, features_path):            
    # we have saved each image feature array to the image folder
    # but we want all images' training features to be in the same array, so does test images...
    
    # get the data length / image number, so that we can define the array size
    num = len(img)
    features_npy = np.zeros(shape=(num, 2048))

    # combine each images 
    for i,j in zip(img, range(num)):
        name = i.split('/')[-1]
        tmp = np.load(img_path + '/' + name + '.npy')
        features_npy[j] = np.array(tmp)

    print('feature shape for '+mode+' data: ',np.shape(features_npy))
    # type(features_npy)

    # save the feature array to the colab data path so that next time we can directly load it
    features_path = features_path+'/'+mode+'_features.npy'
    print('path to save '+mode+' data: ', features_path)
    np.save(features_path, features_npy)
    return features_npy