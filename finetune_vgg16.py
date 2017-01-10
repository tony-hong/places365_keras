# Fix for Tensorflow import error while using Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
import cv2, numpy as np
import pickle
import sys
import random
import os

# Source for vgg16 implementation : https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

def ready_model(weights_path, n_classes):
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
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(365, activation='softmax'))

	model.load_weights(weights_path)
	# Freeze all layers except the last two
	model.pop()
	model.pop()
	model.add(Dropout(0.5))
	model.add(Dense(n_classes, activation='softmax'))
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
	return model


def process_image(img_path):
	im = cv2.resize(cv2.imread(img_path), (224, 224)).astype(np.float32)
	im[:,:,0] -= 103.939
	im[:,:,1] -= 116.779
	im[:,:,2] -= 123.68
	im = im.transpose((2,0,1))
	return im


def train_test_set(image_filenames, n_classes, ratio = 0.8):
	X = []
	y = []
	random.shuffle(image_filenames.keys())
	for img in image_filenames.keys():
		X.append(process_image(img))
		y.append(image_filenames[img])
	split_ratio = int(len(X) * ratio)
	X_train, y_train = X[:split_ratio], y[:split_ratio]
	X_test, y_test = X[split_ratio:], y[split_ratio:]
	# to_categorical
	X_train, y_train =  np.array(X_train), np_utils.to_categorical(np.array(y_train), n_classes)
	X_test, y_test =  np.array(X_test), np_utils.to_categorical(np.array(y_test), n_classes)
	return (X_train, y_train), (X_test, y_test)


def ready_data(base_dir):
	# Link class names with labels
	name_index_mapping = {}
	index_name_mapping = {}
	file_name_mapping = {}
	labels = []
	for root, subdirs, files in os.walk(base_dir):
		labels = subdirs[:]
		break
	for class_index, class_name in enumerate(labels):
		name_index_mapping[class_name] = class_index
		index_name_mapping[class_index] = class_name
	# Walk through directory, gathering names of images with their labels
	n_classes = len(name_index_mapping)
	for root, subdirs, files in os.walk(base_dir):
		for filename in files:
			file_path = os.path.join(root, filename)
			assert file_path.startswith(base_dir)
			suffix = file_path[len(base_dir):]
			suffix = suffix.lstrip("/")
			label = suffix.split("/")[0]
			file_name_mapping[file_path] = name_index_mapping[label]
	return file_name_mapping, n_classes


def fine_tune(model, n_classes, file_name_map):
	n_eopchs = 10
	batch_size = 32
	(X_train, y_train), (X_test, y_test) = train_test_set(file_name_map, n_classes)
	model.fit(X_train,y_train,
				nb_epoch = n_eopchs,
				batch_size = batch_size,
				callbacks = [TensorBoard(log_dir='/tmp/places_vgg16_finetune')])
	loss_and_metrics = model.evaluate(X_test, y_test, batch_size = 32)
	model.save("finetuned_keras.h5")
	return loss_and_metrics


if __name__ == "__main__":
	file_name_map, n_classes = ready_data(sys.argv[1])
	model = ready_model('models/places/places_vgg_keras.h5', n_classes)
	acc = fine_tune(model, n_classes, file_name_map)
	print "Accuracy was ", acc
