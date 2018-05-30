import pickle
import sys

import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2
from pycocotools.coco import COCO

from places_utils import preprocess_input


# Don't hog GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
keras.backend.set_session(sess)
    
# Source for vgg16 implementation : https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3


def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='dense1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='dense2'))
    model.add(Dropout(0.5))
    model.add(Dense(365, activation='softmax'))

    model.load_weights(weights_path)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model


def predict(model, labels, img_path, ret_dict):
    im = cv2.resize(cv2.imread(img_path), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    # Test pretrained model
    out = model.predict(im)
    out  = reversed(np.argsort(out)[0])
    results = []
    for x in out:
        results.append(labels[x])
    ret_dict['place'] = results[:5]
    return results[:5]


def get_features(data_dir, version_d, vgg_model):
    coco_d = dict()
    imgIds_d = dict()

    model = Model(vgg_model.input, vgg_model.get_layer('dense2').output)

    for k in version_d.keys():
        instance_fn_d = '{}/annotations/instances_{}.json'.format(data_dir, version_d[k])
        # initialize COCO api for instance annotations
        coco_d[k] = COCO(instance_fn_d)
        # get all images containing given categories, select one at random
        imgIds_d[k] = coco_d[k].getImgIds(catIds=[])
        print("Number of images: ", len(imgIds_d[k]))
        for i, img_id in enumerate(imgIds_d[k], start=0):
            img = coco_d[k].loadImgs(img_id)[0]
            # transformation
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = model.predict(x)

            # DEBUG
            print features
            print features.shape
            print features[features!=0]
            print features[features!=0].shape
            if i > 10:
                break


if __name__ == "__main__":
    DATA_DIR = '../MSCOCO'
    version_d = dict()
    version_d['train'] = 'train2014'
    version_d['valid'] = 'val2014'

    print "Loaded place module"
    keras.backend.set_image_dim_ordering('th')
    vgg_model = VGG_16('models/places/places_vgg_keras.h5')
    labels = pickle.load(open('models/places/labels.pkl','rb'))

    get_features(DATA_DIR, version_d, vgg_model)


