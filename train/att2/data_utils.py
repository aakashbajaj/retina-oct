import tensorflow as tf
import numpy as np

def dataset_input_fn(filenames, labels, image_size=(HEIGHT,WIDTH,1),shuffle=False, batch_size=64, num_epochs=None,
		buffer_size=4096, prefetch_buffer_size=None):
	dataset = tf.data.TFRecordDataset(filenames)

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
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)

    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
 
    return features, labels