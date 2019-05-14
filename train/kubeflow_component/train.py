import tensorflow as tf
import os
import time

import tensorflow.gfile as tf_reader

from tensorflow.python.client import device_lib

from cnn_model import gen_cnn_model_fn
from data_utils import gen_input_fn

tf.logging.set_verbosity(tf.logging.INFO)

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
parser.add_argument("--batch", type=int, dest="BATCH_SIZE", default=32)
parser.add_argument("--train-steps", type=int, dest="TRAIN_STEPS", default=10000)
parser.add_argument("--prefetch", type=int, dest="PREFETCH", default=-1)
parser.add_argument("--height", type=int, dest="HEIGHT", default=256)
parser.add_argument("--width", type=int, dest="WIDTH", default=256)

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

model_est_fn = gen_cnn_model_fn(num_classes=len(labels))

NUM_GPUS = get_available_gpus()
print("\n{0} GPUs available".format(NUM_GPUS))

strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
config = tf.estimator.RunConfig(train_distribute=strategy)

model_classifier = tf.estimator.Estimator(
	model_fn=model_est_fn,
	config=config,
	model_dir=MODEL_DIR
)

oct_train_in = lambda:dataset_input_fn(
	training_filenames,
	labels,
	shuffle=True,
	batch_size=BATCH_SIZE,
	buffer_size=2048,
	num_epochs=2,
	prefetch_buffer_size=PREFETCH
)