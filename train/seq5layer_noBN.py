import tensorflow as tf
import numpy as np
import json
import os
import logging
from functools import partial

import tensorflow.gfile as tf_reader

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
	image_tensor = features["img_tensor"]

	input_layer = tf.reshape(image_tensor, [-1,256,256,1])

	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=256,
		kernel_size=[3,3],
		padding="valid",
		activation=tf.nn.relu)

	# bn1 = tf.layers.batch_normalization(inputs=conv1, training=(mode == tf.estimator.ModeKeys.TRAIN))
	mp1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
	drp1 = tf.layers.dropout(inputs=mp1, rate=0.25, training=(mode == tf.estimator.ModeKeys.TRAIN))

	conv2 = tf.layers.conv2d(
		inputs=drp1,
		filters=128,
		kernel_size=[3,3],
		padding="valid",
		activation=tf.nn.relu)

	# bn2 = tf.layers.batch_normalization(inputs=conv2, training=(mode == tf.estimator.ModeKeys.TRAIN))
	mp2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
	drp2 = tf.layers.dropout(inputs=mp2, rate=0.25, training=(mode == tf.estimator.ModeKeys.TRAIN))

	conv3 = tf.layers.conv2d(
		inputs=drp2,
		filters=64,
		kernel_size=[3,3],
		padding="valid",
		activation=tf.nn.relu)

	# bn3 = tf.layers.batch_normalization(inputs=conv3, training=(mode == tf.estimator.ModeKeys.TRAIN))
	drp3 = tf.layers.dropout(inputs=conv3, rate=0.25, training=(mode == tf.estimator.ModeKeys.TRAIN))

	flt = tf.layers.flatten(inputs=drp3)

	dns1 = tf.layers.dense(inputs=flt, units=32, activation=tf.nn.relu)
	# dense_bn = tf.layers.batch_normalization(inputs=dns1, training=(mode == tf.estimator.ModeKeys.TRAIN))
	drp4 = tf.layers.dropout(inputs=dns1, rate=0.5, training=(mode == tf.estimator.ModeKeys.TRAIN))

	logits = tf.layers.dense(inputs=drp4, units=4)

	predictions = {
		"classes": tf.argmax(input=logits, axis=1),
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
		# "accuracy": tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=tf.argmax(input=logits, axis=1))
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(
			mode=mode,
			predictions=predictions,
			export_outputs={
				'predict': tf.estimator.export.PredictOutput(predictions)}
			)

	# fix_labels = tf.stop_gradient(labels)

	# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=fix_labels, logits=logits))
	# loss = (tf.nn.softmax_cross_entropy_with_logits_v2(labels=fix_labels, logits=logits))
	# loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

	# accuracy = tf.metrics.accuracy(
	# 	labels=labels, predictions=predictions["classes"])

	# logging_hook = tf.train.LoggingTensorHook({"loss" : loss, "accuracy" : accuracy[1]}, every_n_iter=50)

	if mode == tf.estimator.ModeKeys.TRAIN:
		accuracy =  tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
		tf.summary.scalar('accuracy', accuracy[1])
		eval_train_metrics = {
			"accuracy": accuracy
		}
		optimizer = tf.train.AdamOptimizer()
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		train_op = tf.group([train_op, update_ops])

		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_train_metrics,
			# training_hooks = [logging_hook]
			)

	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
	}
	
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def dataset_input_fn(
	filenames,
	labels,
	image_size=(256,256,1),
	shuffle=False,
	batch_size=32,
	num_epochs=None,
	buffer_size=320,
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

		# return (img_arr, (tf.one_hot(label, num_classes)))
		return ({"img_tensor":img_arr}, label)

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

def _parse_function(filename):
	image_string = tf.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_string, channels=1)
	image_decoded = tf.image.resize_images(image_decoded, (256, 256))
	# image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
	# image_decoded = image_decoded/255.
	image_decoded = tf.cast(image_decoded, tf.float32) / 255.
	image_decoded.set_shape([256, 256, 1])
	return {"img_tensor":image_decoded}
	# return {"input_1": image_decoded}

def predict_input_fn(image_path):
	img_filenames = tf.constant(image_path)
	dataset = tf.data.Dataset.from_tensor_slices(img_filenames)
	dataset = dataset.map(_parse_function)
	# dataset = dataset.repeat(1)
	# dataset = dataset.batch(32)
	
	return dataset

# def serving_input_receiver_fn():
# 	input_ph = tf.placeholder(tf.string, shape=[None], name='image_binary')
# 	images = tf.map_fn(partial(tf.image.decode_image, channels=1), input_ph, dtype=tf.uint8)
# 	images = tf.cast(images, tf.float32) / 255.
# 	# images = tf.image.resize_images(images, (256, 256))
# 	images.set_shape([None, 256, 256, 1])

# 	tf.print(images)

# 	return tf.estimator.export.ServingInputReceiver({"img_tensor": images}, {'bytes': input_ph})


def _img_string_to_tensor(image_string, image_size=(256, 256)):
	image_decoded = tf.image.decode_jpeg(image_string, channels=1)
	# Convert from full range of uint8 to range [0,1] of float32.
	image_decoded_as_float = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
	# Resize to expected
	image_resized = tf.image.resize_images(image_decoded_as_float, size=image_size)
    
	return image_resized

def serving_input_receiver_fn():
    
    feature_spec = {
        'img_tensor': tf.FixedLenFeature([], dtype=tf.string)
    }
    
    default_batch_size = 1
    
    serialized_image = tf.placeholder(dtype=tf.string, shape=[None], name='inp_img_byte_string')
    
    received_tensors = { 'images': serialized_image }
    # features = tf.parse_example(serialized_tf_example, feature_spec)
    
    fn = lambda image: _img_string_to_tensor(image)
    
    img_features = tf.map_fn(fn, serialized_image, dtype=tf.float32)
    
    return tf.estimator.export.ServingInputReceiver({'img_tensor': img_features}, received_tensors)

# estimator.export_savedmodel('export', serving_input_receiver_fn)


if __name__ == '__main__':

	TFR_DIR = "/home/aakashbajaj5311/conv_data_256/conv_256_10may/tfrecords"
	LABEL_LIST = "/home/aakashbajaj5311/conv_data_256/conv_256_10may/labels.json"

	train_path = os.path.join(TFR_DIR, "train")
	test_path = os.path.join(TFR_DIR, "test")

	training_filenames = []
	testing_filenames = []

	if tf_reader.IsDirectory(train_path):
		for filename in tf.gfile.ListDirectory(train_path):
			filepath = os.path.join(train_path, filename)
			training_filenames.append(filepath)
	else:
		print("Invalid training directory. Exiting.......\n")
		exit(0)

	if tf_reader.IsDirectory(test_path):
		for filename in tf.gfile.ListDirectory(test_path):
			filepath = os.path.join(test_path, filename)
			testing_filenames.append(filepath)

	try:
		with tf_reader.GFile(LABEL_LIST, 'rb') as fl:
			labels_bytes = fl.read()
			labels_json = labels_bytes.decode('utf8')

			labels = json.loads(labels_json)
			print(labels)
	except Exception as e:
		print(str(e))
		exit(1)

	print("Training:", training_filenames)
	print("Testing:", testing_filenames)

	strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=2)
	config = tf.estimator.RunConfig(train_distribute=strategy)
	# config = tf.estimator.RunConfig()

	oct_classifier = tf.estimator.Estimator(
		model_fn=cnn_model_fn,
		model_dir="/home/aakashbajaj5311/train_models/oct_classifier_logging_test",
		config=config)

	tensors_to_log = {"probabilities": "softmax_tensor", "train_acc": "accuracy_op"}

	# logging_hook = tf.train.LoggingTensorHook(
	# 	tensors=tensors_to_log, every_n_iter=50)

	oct_train_in = lambda:dataset_input_fn(
		training_filenames,
		labels,
		shuffle=True,
		# batch_size=BATCH_SIZE,
		# buffer_size=2048,
		num_epochs=1,
		# prefetch_buffer_size=PREFETCH
		)

	# oct_classifier.train(
    # 	input_fn=oct_train_in,
	# 	max_steps=7000,
    # 	# steps=1000,
    # 	# hooks=[logging_hook]
    # 	)

	oct_test_in = lambda: dataset_input_fn(
		testing_filenames,
		labels,
		batch_size=10)
	res = oct_classifier.evaluate(input_fn=oct_test_in, steps=100)
	print(res)
	
	print(labels)

	predict_filenames = []

	predict_path = "/home/aakashbajaj5311/datasets/OCT2017/test/DRUSEN"
	
	if tf_reader.IsDirectory(predict_path):
		for filename in tf.gfile.ListDirectory(predict_path):
			filepath = os.path.join(predict_path, filename)
			predict_filenames.append(filepath)
	else:
		print("\nWrong directory")

	# print("Files:")
	# print(predict_filenames)

	pred_in=lambda: predict_input_fn(predict_filenames)

	pred_results = list(oct_classifier.predict(input_fn=pred_in))

	# print(pred_results)

	res_class = []

	for k in pred_results:
		res_class.append(k["classes"])

	c0 = res_class.count(0)
	c1 = res_class.count(1)
	c2 = res_class.count(2)
	c3 = res_class.count(3)

	print(c0)
	print(c1)
	print(c2)
	print(c3)

	oct_classifier.export_savedmodel(
		"/home/aakashbajaj5311/train_models/oct_classifier_logging_test/saved",
		serving_input_receiver_fn=serving_input_receiver_fn
	)