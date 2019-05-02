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
parser.add_argument("--model-dir", dest="MODEL_DIR", help="Folder for model checkpointing", default="/tmp/model_ckpt/")
parser.add_argument("--label-list", required=True, dest="LABEL_LIST", help="Location of labels json file")
parser.add_argument("--epochs", type=int, dest="EPOCHS", default=1)
parser.add_argument("--batch", type=int, dest="BATCH_SIZE", default=64)
parser.add_argument("--train-steps", type=int, dest="TRAIN_STEPS", default=10000)
parser.add_argument("--prefetch", type=int, dest="PREFETCH", default=-1)
parser.add_argument("--height", type=int, dest="HEIGHT", default=224)
parser.add_argument("--width", type=int, dest="WIDTH", default=224)

args = parser.parse_args()

TFR_DIR = args.TFR_DIR
MODEL_DIR = args.MODEL_DIR
LABEL_LIST = args.LABEL_LIST
EPOCHS = int(args.EPOCHS)
BATCH_SIZE = int(args.BATCH_SIZE)
PREFETCH = int(args.PREFETCH)
HEIGHT = int(args.HEIGHT)
WIDTH = int(args.WIDTH)
TRAIN_STEPS = int(args.TRAIN_STEPS)

# if prefetch_buffer_size is None then TensorFlow will use an optimal prefetch buffer size automatically
if PREFETCH == -1:
	PREFETCH = None

tf.logging.set_verbosity(tf.logging.INFO)

class TimeHistory(tf.train.SessionRunHook):
	def begin(self):
		self.times = []
	def before_run(self, run_context):
		self.iter_time_start = time.time()
	def after_run(self, run_context, run_values):
		self.times.append(time.time() - self.iter_time_start)

time_hist = TimeHistory()

def dataset_input_fn(filenames, labels, image_size=(HEIGHT,WIDTH,1), shuffle=False, batch_size=64, num_epochs=None,
		buffer_size=4096, prefetch_buffer_size=None):

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

	if shuffle:
		dataset = dataset.shuffle(buffer_size)
	elif num_epochs is not None:
		dataset = dataset.repeat(num_epochs)
	
	dataset = dataset.map(tfr_parser, num_parallel_calls=os.cpu_count())
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
	
	return dataset

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

import json
# try:
# 	with open(LABEL_LIST, 'r') as fl:
# 		labels = json.load(fl)
# except Exception as e:
# 	print(str(e))
# 	exit(1)

try:
	with tf_reader.GFile(LABEL_LIST, 'rb') as fl:
		labels_bytes = fl.read()
		labels_json = labels_bytes.decode('utf8')

		labels = json.loads(labels_json)
		print(labels)
except Exception as e:
	print(str(e))
	exit(1)

# we will retrain last 5 layers of VGG16 model
keras_vgg = tf.keras.applications.VGG16(input_shape=(HEIGHT, WIDTH, 3), include_top=False)

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
print("\n{0} GPUs available".format(NUM_GPUS))

strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
config = tf.estimator.RunConfig(train_distribute=strategy, model_dir=MODEL_DIR)
estimator = tf.keras.estimator.model_to_estimator(model, config=config)

#logging_hook = tf.train.LoggingTensorHook({"loss" : loss, "accuracy" : accuracy}, every_n_iter=10)

oct_train_in = lambda:dataset_input_fn(training_filenames, labels, shuffle=True, batch_size=BATCH_SIZE, buffer_size=2048, num_epochs=EPOCHS, prefetch_buffer_size=PREFETCH)
train_spec = tf.estimator.TrainSpec(input_fn=oct_train_in, max_steps=TRAIN_STEPS)

# estimator.train(input_fn=oct_train_in, hooks=[time_hist])

oct_test_in = lambda:dataset_input_fn(testing_filenames, labels, shuffle=False, batch_size=BATCH_SIZE, buffer_size=1024, num_epochs=EPOCHS)
eval_spec = tf.estimator.EvalSpec(input_fn=oct_test_in)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
#estimator.evaluate(input_fn=oct_test_in)

# def serving_input_receiver_fn():
# 	inputs = {
# 		'image': tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, 1]),
# 		}
# 	return tf.estimator.export.ServingInputReceiver(inputs, inputs)

# estimator.export_savedmodel(
# 	MODEL_DIR,
# 	serving_input_receiver_fn=serving_input_receiver_fn)
