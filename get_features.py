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
import skimage.io as io
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

    res_mat = np.zeros((581929, 4096))

    for k in version_d.keys():
        instance_fn_d = '{}/annotations/instances_{}.json'.format(data_dir, version_d[k])
        # initialize COCO api for instance annotations
        coco_d[k] = COCO(instance_fn_d)
        # get all images containing given categories, select one at random
        imgIds_d[k] = coco_d[k].getImgIds(catIds=[])
        print("Number of images: ", len(imgIds_d[k]))
        for i, img_id in enumerate(imgIds_d[k], start=0):
            # print img_id
            img_dat = coco_d[k].loadImgs(img_id)[0]
            # print img
            pad_img_id = "%012d" % img_id
            img_fn = '{}/{}/COCO_{}_{}.jpg'.format(data_dir, version_d[k], version_d[k], pad_img_id)


            img = io.imread(img_fn)
            img = cv2.resize(img, (224, 224))
            # transformation
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            if x.all() == -1:
                continue
            if x.ndim != 4:
                continue
            feat_vec = model.predict(x).reshape(-1)
            res_mat[img_id] = feat_vec

            # DEBUG
            # print feat_vec
            # print feat_vec.shape
            # if i > 5:
            #     break
            if i % 100 == 0:
                print i

    # print res_mat
    # print res_mat.shape
    
    return res_mat


if __name__ == "__main__":
    DATA_DIR = '../MSCOCO'
    version_d = dict()
    version_d['train'] = 'train2014'
    version_d['valid'] = 'val2014'

    print "Loaded place module"
    keras.backend.set_image_dim_ordering('th')
    vgg_model = VGG_16('models/places/places_vgg_keras.h5')
    labels = pickle.load(open('models/places/labels.pkl','rb'))

    res_mat = get_features(DATA_DIR, version_d, vgg_model)
    # print res_mat[9]

    with open("coco_imgs", 'w') as f: 
        np.save(f, res_mat)


