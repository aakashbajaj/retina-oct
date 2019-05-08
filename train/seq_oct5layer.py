import tensorflow as tf
import numpy as np
import json
import os

import tensorflow.gfile as tf_reader

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
	input_layer = tf.reshape(features, [-1,224,224,1])

	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=256,
		kernel_size=[3,3],
		padding="valid",
		activation=tf.nn.relu)

	bn1 = tf.layers.batch_normalization(inputs=conv1, training=(mode == tf.estimator.ModeKeys.TRAIN))
	mp1 = tf.layers.max_pooling2d(inputs=bn1, pool_size=[2,2], strides=2)
	drp1 = tf.layers.dropout(
		inputs=mp1, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

	conv2 = tf.layers.conv2d(
		inputs=drp1,
		filters=128,
		kernel_size=[3,3],
		padding="valid",
		activation=tf.nn.relu)

	bn2 = tf.layers.batch_normalization(inputs=conv2, training=(mode == tf.estimator.ModeKeys.TRAIN))
	mp2 = tf.layers.max_pooling2d(inputs=bn1, pool_size=[2,2], strides=2)
	drp2 = tf.layers.dropout(
		inputs=mp2, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

	conv3 = tf.layers.conv2d(
		inputs=drp2,
		filters=64,
		kernel_size=[3,3],
		padding="valid",
		activation=tf.nn.relu)

	bn3 = tf.layers.batch_normalization(inputs=conv3, training=(mode == tf.estimator.ModeKeys.TRAIN))
	drp3 = tf.layers.dropout(
		inputs=bn3, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

	flt = tf.layers.flatten(inputs=drp3)

	dns1 = tf.layers.dense(inputs=flt, units=32, activation=tf.nn.relu)
	# dense_bn = tf.layers.batch_normalization(inputs=dns1, training=(mode == tf.estimator.ModeKeys.TRAIN))
	drp4 = tf.layers.dropout(
		inputs=dns1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

	logits = tf.layers.dense(inputs=drp4, units=4)

	predictions = {
		"classes": tf.argmax(input=logits, axis=1),
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	loss = tf.nn.softmax_cross_entropy_with_logits_v2(
		labels=labels, logits=logits)

	# accuracy = tf.metrics.accuracy(
	# 	labels=labels, predictions=predictions["classes"])

	# logging_hook = tf.train.LoggingTensorHook(
	# 	{"loss" : loss, "accuracy" : accuracy}, every_n_iter=10)

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer()
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())

	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
		# training_hooks = [logging_hook]
		)

	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
		labels=labels,
		predictions=predictions["classes"])
	}
	
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def dataset_input_fn(
	filenames,
	labels,
	image_size=(224,224,1),
	shuffle=False,
	batch_size=64,
	num_epochs=None,
	buffer_size=4096,
	prefetch_buffer_size=None):

	dataset = tf.data.TFRecordDataset(filenames)
	num_classes = len(labels)

	# parser function for reading stored tfrecords
	def tfr_parser(data_record):
		feature_def = {
			'filename': tf.FixedLenFeature([], tf.string),
			'image': tf.FixedLenFeature([], tf.string),
			'label': tf.FixedLenFeature([], tf.int64),
			'classname': tf.FixedLenFeature([], tf.string),
		}

		sample = tf.parse_single_example(data_record, feature_def)
			
		img_arr = tf.decode_raw(sample['image'], tf.float32)
		img_arr = tf.reshape(img_arr, image_size)
		label = tf.cast(sample['label'], tf.int64)

		return (img_arr, tf.one_hot([label], num_classes))

	dataset = dataset.map(tfr_parser, num_parallel_calls=os.cpu_count())
	
	if shuffle and num_epochs:
		dataset = dataset.shuffle(buffer_size).repeat(num_epochs)
	elif shuffle:
		dataset = dataset.shuffle(buffer_size)
	elif num_epochs:
		dataset = dataset.repeat(num_epochs)
	
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
	
	return dataset

if __name__ == '__main__':

	TFR_DIR = "gs://kfp-testing/retin_oct/conv_rskfp/tfrecords"
	LABEL_LIST = "gs://kfp-testing/retin_oct/conv_rskfp/labels.json"

	train_path = os.path.join(TFR_DIR, "train")

	training_filenames = []

	if tf_reader.IsDirectory(train_path):
		for filename in tf.gfile.ListDirectory(train_path):
			filepath = os.path.join(train_path, filename)
			training_filenames.append(filepath)
	else:
		print("Invalid training directory. Exiting.......\n")
		exit(0)

	try:
		with tf_reader.GFile(LABEL_LIST, 'rb') as fl:
			labels_bytes = fl.read()
			labels_json = labels_bytes.decode('utf8')

			labels = json.loads(labels_json)
			print(labels)
	except Exception as e:
		print(str(e))
		exit(1)

	strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=2)
	config = tf.estimator.RunConfig(train_distribute=strategy)

	oct_classifier = tf.estimator.Estimator(
		model_fn=cnn_model_fn,
		model_dir="/tmp/oct_classifier",
		config=config)

	tensors_to_log = {"probabilities": "softmax_tensor"}

	# logging_hook = tf.train.LoggingTensorHook(
	# 	tensors=tensors_to_log, every_n_iter=50)

	oct_train_in = lambda:dataset_input_fn(
		training_filenames,
		labels,
		shuffle=True,
		# batch_size=BATCH_SIZE,
		# buffer_size=2048,
		# num_epochs=EPOCHS,
		# prefetch_buffer_size=PREFETCH
		)

	oct_classifier.train(
    	input_fn=oct_train_in,
    	steps=1000,
    	# hooks=[logging_hook]
    	)