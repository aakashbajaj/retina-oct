import tensorflow as tf

def tfr_parser(data_record):
	feature_def = {
		'filename': tf.FixedLenFeature([], tf.string),
		'image': tf.FixedLenFeature([], tf.string),
		'label': tf.FixedLenFeature([], tf.int64),
		'classname': tf.FixedLenFeature([], tf.string),
	}

	sample = tf.parse_single_example(data_record, feature_def)

	img_arr = tf.decode_raw(sample['image'], tf.float32)
	img_arr = tf.reshape(img_arr, (224,224,3))
                # label = tf.decode_raw(sample['label'], tf.int64)
	label = tf.cast(sample['label'], tf.int64)

	return (img_arr, tf.one_hot([label], 4))
