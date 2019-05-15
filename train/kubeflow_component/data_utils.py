import tensorflow as tf
import os

def gen_input_fn(image_size, num_classes):

	def dataset_input_fn(
		filenames,
		image_size=image_size,
		shuffle=False,
		batch_size=32,
		num_epochs=None,
		buffer_size=320,
		prefetch_buffer_size=None):

		dataset = tf.data.TFRecordDataset(filenames)
		# num_classes = len(labels)

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

			return (img_arr, label)

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

	return dataset_input_fn

def get_serving_input_receiver_fn(image_size=(256,256,1)):
	def serving_input_receiver_fn():
		inputs = {
			INPUT_FEATURE: tf.placeholder(tf.float32, [None, image_size[0], image_size[1], image_size[2]]),
		}
		return tf.estimator.export.ServingInputReceiver(inputs, inputs)

	return serving_input_receiver_fn