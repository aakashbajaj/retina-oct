import tensorflow as tf
import numpy as np
import json
import os

import tensorflow.gfile as tf_reader

tf.logging.set_verbosity(tf.logging.INFO)

# def dataset_input_fn(
# 	filenames,
# 	labels,
# 	image_size=(224,224,1),
# 	shuffle=False,
# 	batch_size=64,
# 	num_epochs=None,
# 	buffer_size=4096,
# 	prefetch_buffer_size=None):

# 	dataset = tf.data.TFRecordDataset(filenames)
# 	num_classes = len(labels)

# 	# parser function for reading stored tfrecords
# 	def tfr_parser(data_record):
# 		feature_def = {
# 			'filename': tf.FixedLenFeature([], tf.string),
# 			'image': tf.FixedLenFeature([], tf.string),
# 			'label': tf.FixedLenFeature([], tf.int64),
# 			'classname': tf.FixedLenFeature([], tf.string),
# 		}

# 		sample = tf.parse_single_example(data_record, feature_def)
			
# 		img_arr = tf.decode_raw(sample['image'], tf.float32)
# 		img_arr = tf.reshape(img_arr, image_size)
# 		label = tf.cast(sample['label'], tf.int64)

# 		filename = sample['filename']
# 		classname = sample['classname']

# 		return (img_arr, tf.one_hot([label], num_classes), filename, classname)

# 	dataset = dataset.map(tfr_parser, num_parallel_calls=os.cpu_count())
	
# 	if shuffle and num_epochs:
# 		dataset = dataset.shuffle(buffer_size).repeat(num_epochs)
# 	elif shuffle:
# 		dataset = dataset.shuffle(buffer_size)
# 	elif num_epochs:
# 		dataset = dataset.repeat(num_epochs)
	
# 	dataset = dataset.batch(batch_size)
# 	dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
	
# 	return dataset

def tfr_parser(data_record):
	feature_def = {
		'filename': tf.FixedLenFeature([], tf.string),
		'image': tf.FixedLenFeature([], tf.string),
		'label': tf.FixedLenFeature([], tf.int64),
		'classname': tf.FixedLenFeature([], tf.string),
	}

	sample = tf.parse_single_example(data_record, feature_def)
	
	image_size=(256,256,1)
	num_classes = 4

	img_arr = sample['image']
	img_arr = tf.decode_raw(sample['image'], tf.float32)
	img_arr = tf.reshape(img_arr, image_size)
	label = tf.cast(sample['label'], tf.int64)

	filename = sample['filename']
	classname = sample['classname']

	# return (img_arr, tf.one_hot(label, num_classes), filename, classname)
	return (img_arr, label, filename, classname)


if __name__ == '__main__':

	TFR_DIR = "gs://kfp-testing/retin_oct/conv_256_10may/tfrecords"
	LABEL_LIST = "gs://kfp-testing/retin_oct/conv_256_10may/labels.json"
	# TFR_DIR = "/home/aakashbajaj5311/conv_data_256/conv_256_10may/tfrecords/"
	# LABEL_LIST = "/home/aakashbajaj5311/conv_data_256/conv_256_10may/labels.json"

	train_path = os.path.join(TFR_DIR, "test")

	training_filenames = []

	if tf_reader.IsDirectory(train_path):
		for filename in tf.gfile.ListDirectory(train_path):
			filepath = os.path.join(train_path, filename)
			training_filenames.append(filepath)
	else:
		print("Invalid training directory. Exiting.......\n")
		exit(0)

	# training_filenames = ["/home/techno/oct_data/retin_oct_conv_9may_tfrecords_test_test-00000-of-00005"]
	training_filenames = [training_filenames[0]]
	print(training_filenames)

	dataset = tf.data.TFRecordDataset(training_filenames)
	dataset = dataset.map(tfr_parser)
	iter = dataset.make_one_shot_iterator()

	next_elem = iter.get_next()

	import cv2
	with tf.Session() as sess:
		abc = []
		for i in range(10):
			features, label, filename, classname = sess.run(next_elem)
			print(features.shape)

			print(np.argmax(label))
			print(label)
			abc.append(label)
			print(filename)
			print(classname)
			cv2.imshow("image{}".format(i), features)

		cv2.waitKey()
		# print(np.array(abc))
		# print(np.argmax(abc, axis=1))
		# cv2.waitKey()
