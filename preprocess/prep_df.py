from __future__ import print_function

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import apache_beam as beam
import tensorflow as tf

import csv
import random
import json

import tensorflow.gfile as tf_reader

import logging

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
		self._resize_area = tf.image.resize_images(self._decode_jpeg, size=[height, width], method=tf.image.ResizeMethod.AREA)

	def decode_jpeg(self, image_data):
		image = self._sess.run(self._resize_area, feed_dict={self._decode_jpeg_data: image_data})
		
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
	image = coder.decode_jpeg(image_data)

	# Check that image converted to RGB
	assert len(image.shape) == 3
	height = image.shape[0]
	width = image.shape[1]
	assert image.shape[2] == 1

	return image.tostring(), height, width

def convert_to_example(csvline, categories, resize_image_dims):
	"""Parse a line of CSV file and convert to TF Record.

	Args:
	csvline: line from input CSV file
	categories: list of labels
	Yields:
	serialized TF example if the label is in categories
	"""
	# logging.info(csvline.encode('ascii', 'ignore'))
	# filename, label = csvline.encode('ascii', 'ignore').split(',')
	filename, label = csvline.split(',')
	# print(filename, label)
	logging.info("{} with {}".format(filename, label))

	# fl1 = tf_reader.GFile('gs://kfp-testing/retin_oct/debug/log1.txt', 'a')
	# fl1.write("{} with {}".format(filename, label))
	# fl1.close()
	
	filename = filename.rstrip()
	label = label.rstrip()

	is_image_file = filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".tif")
	
	if label in categories and is_image_file:
		logging.info("processed {} with {}".format(filename, label))
		
		# fl2 = tf_reader.GFile('gs://kfp-testing/retin_oct/debug/log2.txt', 'a')
		# fl2.write("{} with {}".format(filename, label))
		# fl2.close()
		
		# ignore labels not in categories list
		coder = ImageCoder()
		image_buffer, im_height, im_width = _get_image_data(filename, coder)
		del coder
		example = _convert_to_example(filename, image_buffer,
										categories[label], label, resize_image_dims[0], resize_image_dims[1])
		yield example.SerializeToString()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("--train-csv", required=True)
	parser.add_argument("--test-csv")
	parser.add_argument("--input-dir", required=True)
	parser.add_argument("--output-dir", required=True)
	parser.add_argument("--label-list", required=True, help="Location of labels json file")
	parser.add_argument("--project-id",	required=True)
	parser.add_argument("--runner",	default=None)
	parser.add_argument("--num-shards", type=int, default=5)
	parser.add_argument("--split-flag", type=int, default=2)
	parser.add_argument("--height", type=int, default=224)
	parser.add_argument("--width", type=int, default=224)


	args = parser.parse_args()
	arguments = args.__dict__

	SPLIT_FLAG = arguments['split_flag']
	
	if SPLIT_FLAG == 0:
		targets = ['train']

	elif SPLIT_FLAG == 2:
		targets = ['train', 'test']
		if not (arguments['train_csv'] and arguments['test_csv']):
			print("Train and Test CSVs required.")
			exit(1)
	else:
		print("Invalid split flag value. Exiting....")
		exit(1)


	NUM_SHARDS = arguments['num_shards']
	HEIGHT = arguments['height']
	WIDTH = arguments['width']

	JOBNAME = ('preprocess-images-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S'))

	PROJECT = arguments['project_id']
	OUTPUT_DIR = arguments['output_dir']

	LABEL_LIST = arguments['label_list']

	try:
		with tf_reader.GFile(LABEL_LIST, 'rb') as fl:
			labels_bytes = fl.read()
			labels_json = labels_bytes.decode('utf8')

			labels = json.loads(labels_json)
			print(labels)
	except Exception as e:
		print(str(e))
		exit(1)

	reverse_labels = dict()

	for k in labels:
		classname = labels[k]
		reverse_labels[classname] = k

	# set RUNNER using command-line arg or based on output_dir path
	INPUT_DIR = arguments['input_dir']

	on_cloud = OUTPUT_DIR.startswith('gs://') or INPUT_DIR.startswith('gs://')

	if arguments['runner'] and arguments['runner']!="None":
		RUNNER = arguments['runner']
	else:
		RUNNER = 'DataflowRunner' if on_cloud else 'DirectRunner'

	print("Runner: {}".format(RUNNER))

	# clean-up output directory since Beam will name files 0000-of-0004 etc.
	# and this could cause confusion if earlier run has 0000-of-0005, for eg
	if on_cloud:
		try:
			subprocess.check_call('gsutil -m rm -r {}'.format(OUTPUT_DIR).split())
		except subprocess.CalledProcessError:
			pass
	else:
		shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
		os.makedirs(OUTPUT_DIR)

	# set up Beam pipeline to convert images to TF Records
	options = {
		'staging_location': os.path.join(OUTPUT_DIR, 'tmp', 'staging'),
		'temp_location': os.path.join(OUTPUT_DIR, 'tmp'),
		'job_name': JOBNAME,
		'project': PROJECT,
		'teardown_policy': 'TEARDOWN_ALWAYS',
		'save_main_session': True,
		'machine_type': 'n1-standard-8',
		'max_num_workers': 20,
		'num_workers': 4
		}
	opts = beam.pipeline.PipelineOptions(flags=[], **options)

	with beam.Pipeline(RUNNER, options=opts) as p:
		# BEAM tasks
		for step in targets:
			print(arguments['{}_csv'.format(step)])
			_ = (
				p
				| '{}_read_csv'.format(step) >> beam.io.ReadFromText(
				arguments['{}_csv'.format(step)])
				| '{}_convert'.format(step) >>
				beam.FlatMap(lambda line: convert_to_example(line, reverse_labels, (HEIGHT, WIDTH)))
				| '{}_write_tfr'.format(step) >> 
				beam.io.tfrecordio.WriteToTFRecord(os.path.join(OUTPUT_DIR, (step+"/{}").format(step)), num_shards=NUM_SHARDS))
