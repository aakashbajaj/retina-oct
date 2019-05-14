import tensorflow as tf
import os
import time

import tensorflow.gfile as tf_reader

from tensorflow.python.client import device_lib

from cnn_model import gen_cnn_model_fn
import data_utils

tf.logging.set_verbosity(tf.logging.INFO)

def get_available_gpus():
	local_device_protos = device_lib.list_local_devices()
	dev_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
	return len(dev_list)

from argparse import ArgumentParser
parser = ArgumentParser()

# Flags
parser.add_argument("--train-flag", type=int, dest="TRAIN_FLAG", default=1)
parser.add_argument("--evaluate-flag", type=int, dest="EVALUATE_FLAG", default=1)
parser.add_argument("--save-model-flag", type=int, dest="SAVE_MODEL_FLAG", default=1)
parser.add_argument("--distribute", type=int, dest="DISTRIBUTE_FLAG", default=1)

# parameters
parser.add_argument("--tfr-dir", required=True, dest="TFR_DIR", help="Folder containing converted records")
parser.add_argument("--model-dir", dest="MODEL_DIR", help="Folder for model checkpointing", default="/tmp/model_ckpt/")
parser.add_argument("--save-model-dir", dest="SAVE_MODEL_DIR", help="Folder for exporting saved model", default="/tmp/model_ckpt/saved")
parser.add_argument("--label-list", required=True, dest="LABEL_LIST", help="Location of labels json file")
parser.add_argument("--num-epochs", type=int, dest="NUM_EPOCHS", default=1)
parser.add_argument("--batch-size", type=int, dest="BATCH_SIZE", default=32)
parser.add_argument("--train-steps", type=int, dest="TRAIN_STEPS", default=10000)
parser.add_argument("--max-train-steps", type=int, dest="MAX_TRAIN_STEPS", default=10000)
parser.add_argument("--prefetch-buffer", type=int, dest="PREFETCH", default=-1)
parser.add_argument("--height", type=int, dest="HEIGHT", default=256)
parser.add_argument("--width", type=int, dest="WIDTH", default=256)
parser.add_argument("--channels", type=int, dest="CHANNELS", default=1)
parser.add_argument("--learning-rate", type=int, dest="LEARNING_RATE", default=0.001)

args = parser.parse_args()
arguments = args.__dict__

TRAIN_FLAG = args.TRAIN_FLAG
EVALUATE_FLAG = args.EVALUATE_FLAG
SAVE_MODEL_FLAG = args.SAVE_MODEL_FLAG
DISTRIBUTE_FLAG = args.DISTRIBUTE_FLAG

TFR_DIR = args.TFR_DIR
MODEL_DIR = args.MODEL_DIR
SAVE_MODEL_DIR = args.SAVE_MODEL_DIR
LABEL_LIST = args.LABEL_LIST
NUM_EPOCHS = int(args.NUM_EPOCHS)
BATCH_SIZE = int(args.BATCH_SIZE)
TRAIN_STEPS = int(args.TRAIN_STEPS)
MAX_TRAIN_STEPS = int(args.MAX_TRAIN_STEPS)
PREFETCH = int(args.PREFETCH)
HEIGHT = int(args.HEIGHT)
WIDTH = int(args.WIDTH)
CHANNELS = int(args.CHANNELS)
LEARNING_RATE = int(args.LEARNING_RATE)

# if prefetch_buffer_size is None then TensorFlow will use an optimal prefetch buffer size automatically
if PREFETCH == -1:
	PREFETCH = None


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
try:
	with tf_reader.GFile(LABEL_LIST, 'rb') as fl:
		labels_bytes = fl.read()
		labels_json = labels_bytes.decode('utf8')

		labels = json.loads(labels_json)
		print(labels)
except Exception as e:
	print(str(e))
	exit(1)

model_est_fn = gen_cnn_model_fn(image_size=(HEIGHT, WIDTH, CHANNELS), num_classes=len(labels), opt_learn_rate=LEARNING_RATE)

NUM_GPUS = get_available_gpus()
print("\n{0} GPUs available".format(NUM_GPUS))

if DISTRIBUTE_FLAG:
	strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
	config = tf.estimator.RunConfig(train_distribute=strategy)
else:
	config = tf.estimator.RunConfig()

model_classifier = tf.estimator.Estimator(
	model_fn=model_est_fn,
	config=config,
	model_dir=MODEL_DIR
)

dataset_input_fn = data_utils.gen_input_fn(image_size=(HEIGHT, WIDTH, CHANNELS), num_classes=len(labels))

if TRAIN_FLAG:
	oct_train_in = lambda:dataset_input_fn(
		training_filenames,
		shuffle=True,
		batch_size=BATCH_SIZE,
		buffer_size=BATCH_SIZE*10,
		num_epochs=NUM_EPOCHS,
		prefetch_buffer_size=PREFETCH
	)

	model_classifier.train(
    	input_fn=oct_train_in,
		max_steps=MAX_TRAIN_STEPS,
    	steps=TRAIN_STEPS,
    	# hooks=[logging_hook]
    	)

if EVALUATE_FLAG:
	oct_test_in = lambda: dataset_input_fn(
		testing_filenames,
		batch_size=10)
	res = model_classifier.evaluate(input_fn=oct_test_in, steps=100)
	print(res)

if SAVE_MODEL_FLAG:
	serving_input_receiver_fn = data_utils.get_serving_input_receiver_fn(image_size=(HEIGHT, WIDTH, CHANNELS))

	model_classifier.export_savedmodel(
		SAVE_MODEL_DIR,
		serving_input_receiver_fn=serving_input_receiver_fn
	)