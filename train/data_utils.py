import tensorflow as tf
import os

def dataset_input_fn(filenames, labels, 
	image_size=(224,224),
	shuffle=False,
	batch_size=64,
	num_epochs=None,
	buffer_size=4096,
	prefetch_buffer_size=None
	):

	dataset = tf.data.TFRecordDataset(filenames)
	num_classes = len(labels)

	def parser(data_record):
		feature_def = {
			'filename': tf.FixedLenFeature([], tf.string),
			'image': tf.FixedLenFeature([], tf.string),
			'label': tf.FixedLenFeature([], tf.int64),
			'classname': tf.FixedLenFeature([], tf.string),
		}

		sample = tf.parse_single_example(data_record, feature_def)
			
		img_arr = tf.decode_raw(sample['image'], tf.float32)
		img_arr = tf.reshape(img_arr, (image_size[0], image_size[1], 3))
		# label = tf.decode_raw(sample['label'], tf.int64)
		label = tf.cast(sample['label'], tf.int64)

		return (img_arr, tf.one_hot([label], num_classes))

	if num_epochs is not None and shuffle:
		dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size, num_epochs))
	elif shuffle:
		dataset = dataset.shuffle(buffer_size)
	elif num_epochs is not None:
		dataset = dataset.repeat(num_epochs)

	dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parser,
								batch_size=batch_size,
								num_parallel_calls=os.cpu_count()))
	dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

	return dataset
