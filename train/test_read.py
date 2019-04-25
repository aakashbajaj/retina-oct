import tensorflow as tf
import os

def tfr_parser(data_record):
	feature_def = {
		'filename': tf.FixedLenFeature([], tf.string),
		'image': tf.FixedLenFeature([], tf.string),
		'label': tf.FixedLenFeature([], tf.int64),
		'classname': tf.FixedLenFeature([], tf.string),
	}

	sample = tf.parse_single_example(data_record, feature_def)

	img_arr = tf.decode_raw(sample['image'], tf.float32)
	img_arr = tf.reshape(img_arr, (224,224))
    #             # label = tf.decode_raw(sample['label'], tf.int64)
	label = tf.cast(sample['label'], tf.int64)

	return (img_arr, label)

tr_files = ['/home/aakashbajaj5311/oct/conv/train/data_01_of_02.tfrecord']

dataset = tf.data.TFRecordDataset(tr_files)
dataset = dataset.map(tfr_parser)
iterator = dataset.make_one_shot_iterator()

next_elem = iterator.get_next()

with tf.Session() as sess:
    try:
        while True:
            features, label = sess.run(next_elem)
            print(features.shape)
            print("****************\n")
            # print(label)
            # print(label.shape)
    except Exception as e:
        print(str(e))
