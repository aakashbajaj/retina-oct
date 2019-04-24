import tensorflow as tf
import os
import time

import tensorflow.gfile as tf_reader

from tensorflow.python.client import device_lib

def get_available_gpus():
	local_device_protos = device_lib.list_local_devices()
	dev_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
	return len(dev_list)

from argparse import ArgumentParser
parser = ArgumentParser()



parser.add_argument("--tfr-dir", required=True, dest="TFR_DIR", help="Folder containing converted records")
parser.add_argument("--ckpt-dir", dest="CKPT_DIR", help="Folder for model checkpointing", default="/tmp/model_ckpt/")
parser.add_argument("--label-list", required=True, dest="LABEL_LIST", help="Location of labels json file")
parser.add_argument("--num-shards", type=int, dest="NUM_SHARDS", default=2)
parser.add_argument("--split-flag", type=int, dest="SPLIT_FLAG", default=1)
parser.add_argument("--height", type=int, dest="HEIGHT", default=224)
parser.add_argument("--width", type=int, dest="WIDTH", default=224)

args = parser.parse_args()





tf.logging.set_verbosity(tf.logging.INFO)

class TimeHistory(tf.train.SessionRunHook):
	def begin(self):
		self.times = []
	def before_run(self, run_context):
		self.iter_time_start = time.time()
	def after_run(self, run_context, run_values):
		self.times.append(time.time() - self.iter_time_start)

def dataset_input_fn(filenames, labels, 
	image_size=(224,224,1),
	shuffle=False,
	batch_size=64,
	num_epochs=None,
	buffer_size=4096,
	prefetch_buffer_size=None):

	dataset = tf.data.TFRecordDataset(filenames)
	num_classes = len(labels)

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

	# if num_epochs is not None and shuffle:
	# 	dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size, num_epochs))
	# elif shuffle:
	# 	dataset = dataset.shuffle(buffer_size)
	# elif num_epochs is not None:
	# 	dataset = dataset.repeat(num_epochs)

	# dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=tfr_parser,
	# 							batch_size=batch_size,
	# 							num_parallel_calls=os.cpu_count()))
	# dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
	if shuffle:
		dataset = dataset.shuffle(buffer_size)
	elif num_epochs is not None:
		dataset = dataset.repeat(num_epochs)
	
	dataset = dataset.map(tfr_parser)
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
	
	return dataset

train_path = "/home/aakashbajaj5311/oct/conv/train"
test_path = "/home/aakashbajaj5311/oct/conv/test"

training_filenames = []
testing_filenames = []

if tf_reader.IsDirectory(train_path):
	for filename in tf.gfile.ListDirectory(train_path):
		filepath = os.path.join(train_path, filename)
		training_filenames.append(filepath)

if tf_reader.IsDirectory(test_path):
	for filename in tf.gfile.ListDirectory(test_path):
		filepath = os.path.join(test_path, filename)
		testing_filenames.append(filepath)

import json
labels_path = "/home/aakashbajaj5311/oct/conv/labels.json"
with open(labels_path, 'r') as fl:
	labels = json.load(fl)

keras_vgg = tf.keras.applications.VGG16(input_shape=(224,224,3), include_top=False)

output = keras_vgg.output
output = tf.keras.layers.Flatten()(output)
prediction = tf.keras.layers.Dense(len(labels), activation=tf.nn.softmax)(output)

model = tf.keras.Model(inputs=keras_vgg.input, outputs=prediction)

for layers in keras_vgg.layers[:-4]:
	layers.trainable = False

model.compile(loss='categorical_crossentropy',
		optimizer=tf.train.AdamOptimizer(),
		metrics=['accuracy'])

NUM_GPUS = get_available_gpus()


strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
config = tf.estimator.RunConfig(train_distribute=strategy)
estimator = tf.keras.estimator.model_to_estimator(model, config=config)

BATCH_SIZE = 32
EPOCHS = 1

time_hist = TimeHistory()
#logging_hook = tf.train.LoggingTensorHook({"loss" : loss, "accuracy" : accuracy}, every_n_iter=10)

oct_train_in = lambda:dataset_input_fn(training_filenames, labels, shuffle=True, batch_size=BATCH_SIZE, buffer_size=2048, num_epochs=EPOCHS, prefetch_buffer_size=4)
train_spec = tf.estimator.TrainSpec(input_fn=oct_train_in, max_steps=10000)

# estimator.train(input_fn=oct_train_in, hooks=[time_hist])

oct_test_in = lambda:dataset_input_fn(testing_filenames, labels, shuffle=False, batch_size=BATCH_SIZE, buffer_size=1024, num_epochs=EPOCHS)
eval_spec = tf.estimator.EvalSpec(input_fn=oct_test_in)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
#estimator.evaluate(input_fn=oct_test_in)
