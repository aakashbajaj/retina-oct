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

			return ({"image":img_arr}, label)

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


def get_serving_input_receiver_fn(image_size):
	def _img_string_to_tensor(image_string, image_size):
		image_decoded = tf.image.decode_jpeg(image_string, channels=image_size[2])
		image_decoded_as_float = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
		image_resized = tf.image.resize_images(image_decoded_as_float, size=(image_size[0], image_size[1]))
		return image_resized

	def serving_input_receiver_fn():
		feature_spec = {
			'image': tf.FixedLenFeature([], dtype=tf.string)
		}
		default_batch_size = 1
		serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[default_batch_size], name='input_image_tensor')
		received_tensors = { 'images': serialized_tf_example }
		features = tf.parse_example(serialized_tf_example, feature_spec)
		fn = lambda image: _img_string_to_tensor(image, (image_size))
		features['image'] = tf.map_fn(fn, features['image'], dtype=tf.float32)
		
		return tf.estimator.export.ServingInputReceiver(features, received_tensors)

	return serving_input_receiver_fn


def get_predict_fn(image_size):
	def _parse_function(filename):
		image_string = tf.read_file(filename)
		image_decoded = tf.image.decode_jpeg(image_string, channels=image_size[2])
		image_decoded = tf.image.resize_images(image_decoded, (image_size[0], image_size[1]))
		image_decoded = tf.cast(image_decoded, tf.float32) / 255.
		image_decoded.set_shape(image_size)

		return {"image":image_decoded}

	def predict_input_fn(image_path):
		img_filenames = tf.constant(image_path)
		dataset = tf.data.Dataset.from_tensor_slices(img_filenames)
		dataset = dataset.map(_parse_function)
		
		return dataset