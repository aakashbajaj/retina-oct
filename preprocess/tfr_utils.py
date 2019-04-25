import tensorflow as tf
import os
import cv2
import numpy as np
import random
from tqdm import tqdm

import io
import logging
from xml.dom import minidom

import tensorflow.gfile as tf_reader
from tensorflow.python.keras.preprocessing.image import img_to_array

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _load_image(image_path, height, width, ann_dir):
	try:
		with tf_reader.GFile(image_path, 'rb') as fl:
			image_bytes = fl.read()
			
			image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

			if ann_dir != "":
				image = _get_ann_images(path, image, ann_dir)
			image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
			im = np.array(img_to_array(image) / 255.)
			return im


	except Exception as e:
		print("Error Processing Image: %s\n %s" % (image_path, str(e)))
		return

def _get_ann_images(filepath, img_arr, ann_dir):
	image_name = (filepath.split('/')[-1]).split(".")[-2]
	breed_folder = filepath.split('/')[-2]

	ann_filepath = os.path.join(ann_dir, breed_folder, image_name)
	ann_xml = minidom.parse(ann_filepath)
	xmin = int(ann_xml.getElementsByTagName('xmin')[0].firstChild.nodeValue)
	ymin = int(ann_xml.getElementsByTagName('ymin')[0].firstChild.nodeValue)
	xmax = int(ann_xml.getElementsByTagName('xmax')[0].firstChild.nodeValue)
	ymax = int(ann_xml.getElementsByTagName('ymax')[0].firstChild.nodeValue)

	new_img_arr = img_arr[ymin:ymax, xmin:xmax, :]

	return new_img_arr

def build_example_list_tf(input_dir, seed):
	examples = []
	labels = {}
	class_cnt = 0

	for classname in tf.gfile.ListDirectory(input_dir):
		class_dir = os.path.join(input_dir, classname)
		
		if tf.gfile.IsDirectory(class_dir):	
			for filename in tf.gfile.ListDirectory(class_dir):
				filepath = os.path.join(class_dir, filename)
				example = {
					'classname': classname,
					'path': filepath,
					'label': class_cnt
				}
				examples.append(example)

			labels[class_cnt] = classname
			class_cnt = class_cnt + 1

	random.seed(seed)
	random.shuffle(examples)
	return examples, labels


def get_example_share(examples, train_split):
	example_len = len(examples)
	training_len = int(example_len*train_split)

	return np.split(examples, [training_len])

def split_list(tar_list, wanted_parts=1):	
	length = len(tar_list)
	return [tar_list[i*length//wanted_parts : (i+1)*length//wanted_parts] for i in range(wanted_parts)]

def _write_tf_records(examples, output_filename, image_dims, ann_dir):
	writer = tf.python_io.TFRecordWriter(output_filename)
	cnt = 0
	for example in tqdm(examples):
		try:
			cnt += 1	
			image = _load_image(example['path'], height=image_dims[0], width=image_dims[1], ann_dir=ann_dir)
			if image is not None:
				im_str = image.tostring()

				g_label = example['label']
				
				tf_example = tf.train.Example(features = tf.train.Features(feature = {
					'filename': tf.train.Feature(bytes_list = tf.train.BytesList(value = [example['path'].encode('utf-8')])),
					'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [im_str])),
					'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [g_label])),
					'classname': tf.train.Feature(bytes_list = tf.train.BytesList(value = [example['classname'].encode('utf-8')]))
				}))

				writer.write(tf_example.SerializeToString())
		except Exception as e:
			print(e)
			pass
	writer.close()


# not being used
def _write_sharded_tfrecords(examples, num_shards, output_dir, image_dims, is_training=True):
	sharded_examples = _split_list(examples, num_shards)

	for count, shard in tqdm(enumerate(sharded_examples, start=1)):
		output_filename = '{0}_{1:02d}_of_{2:02d}.tfrecord'.format('train' if is_training else 'test', count, num_shards)
		out_filepath = os.path.join(output_dir, output_filename)
		_write_tf_records(shard, out_filepath, image_dims)