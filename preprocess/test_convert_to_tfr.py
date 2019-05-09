from __future__ import print_function

import argparse
import datetime
import os
import shutil
import subprocess
import sys
# import apache_beam as beam
import tensorflow as tf
import numpy as np
import cv2

import csv
import random
import json

import tensorflow.gfile as tf_reader

import logging

from tensorflow.python.keras.preprocessing.image import img_to_array

tf.logging.set_verbosity(tf.logging.INFO)


def _int64_feature(value):
	"""Wrapper for inserting int64 features into Example proto."""
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
	"""Wrapper for inserting bytes features into Example proto."""
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(filename, image_buffer, label_int, label_str, height, width):
	"""Build an Example proto for an example.

	Args:
	filename: string, path to an image file, e.g., '/path/to/example.JPG'
	image_buffer: string, JPEG encoding of RGB image
	label_int: integer, identifier for ground truth (0-based)
	label_str: string, identifier for ground truth, e.g., 'daisy'
	height: integer, image height in pixels
	width: integer, image width in pixels
	Returns:
	Example proto
	"""
	# colorspace = 'RGB'
	# channels = 1
	# image_format = 'JPEG'

	example = tf.train.Example(
		features=tf.train.Features(
			feature={
				'filename': _bytes_feature(filename.encode('utf-8')),
				'image': _bytes_feature(image_buffer),
				'label': _int64_feature(int(label_int)),  # model expects 1-based
				'classname': _bytes_feature(label_str.encode('utf-8')),
				# 'image/height': _int64_feature(height),
				# 'image/width': _int64_feature(width),
				# 'image/colorspace': _bytes_feature(colorspace),
				# 'image/channels': _int64_feature(channels),
				# 'image/format': _bytes_feature(image_format),
				}))

	return example




class ImageCoder(object):
	"""Helper class that provides TensorFlow image coding utilities."""

	def __init__(self, height=224, width=224, channels=1):
		# Create a single Session to run all image coding calls.
		self._sess = tf.Session()

		# Initializes function that decodes RGB JPEG data.
		self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
		self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=channels)
		# self._batch_img = tf.expand_dims(self._decode_jpeg, 0)
		self._resize_area = tf.image.resize_images(self._decode_jpeg, size=[height, width], method=tf.image.ResizeMethod.BICUBIC)

	def decode_jpeg(self, image_data):
		image = self._sess.run(self._resize_area, feed_dict={self._decode_jpeg_data: image_data})
		
		print("through tf: ", image.shape)
		assert len(image.shape) == 3
		assert image.shape[2] == 1
		return image

	def __del__(self):
		self._sess.close()


def _get_image_data(filename, coder):
	"""Process a single image file.

	Args:
	filename: string, path to an image file e.g., '/path/to/example.JPG'.
	coder: instance of ImageCoder to provide TensorFlow image coding utils.
	Returns:
	image_buffer: string, JPEG encoding of RGB image.
	height: integer, image height in pixels.
	width: integer, image width in pixels.
	"""
	# Read the image file.
	with tf.gfile.GFile(filename, 'rb') as ifp:
		image_data = ifp.read()

	#decode(image_data)
	#resize(image_data)

	# Decode the RGB JPEG.
	# nparr = np.fromstring(image_data, np.uint8)
	# # print(nparr)
	# img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
	# print(img_np[0])
	# # nparr = nparr/255.
	
	image = coder.decode_jpeg(image_data)
	im = np.array(img_to_array(image))/255.
	print(im[0][:10])
	print("Shape: ", im.shape)

	# Check that image converted to RGB
	height = image.shape[0]
	width = image.shape[1]

	return im, height, width

if __name__ == '__main__':

	NUM_SHARDS = 2
	HEIGHT = 224
	WIDTH = 224

	# LABEL_LIST = "gs://kfp-testing/retin_oct/conv_rskfp/labels.json"

	# try:
	# 	with tf_reader.GFile(LABEL_LIST, 'rb') as fl:
	# 		labels_bytes = fl.read()
	# 		labels_json = labels_bytes.decode('utf8')

	# 		labels = json.loads(labels_json)
	# 		print(labels)
	# except Exception as e:
	# 	print(str(e))
	# 	exit(1)

	# reverse_labels = dict()

	# for k in labels:
	# 	classname = labels[k]
	# 	reverse_labels[classname] = k

	input_csv = "gs://kfp-testing/retin_oct/conv_rskfp/file_list_csv/test_list.csv"

	coder = ImageCoder()
	from PIL import Image

	with tf.gfile.GFile(input_csv, 'r') as fl:
		for i in range(3):
			filename, classname = fl.readline().split(',')
			print("\n",filename)
			print(classname)
			data, ht, wt = _get_image_data(filename, coder)
			print(data.shape)
			print(data.dtype)
			# if i == 2:
			# 	cv2.imshow("image", data)
			# 	cv2.waitKey()