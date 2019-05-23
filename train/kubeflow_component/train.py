import tensorflow as tf
import os
import json
import time

import tensorflow.gfile as tf_reader
from tensorflow.python.lib.io import file_io

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
# parser.add_argument("--save-model-flag", type=int, dest="SAVE_MODEL_FLAG", default=1)
parser.add_argument("--distribute", type=int, dest="DISTRIBUTE_FLAG", default=1)

# parameters
parser.add_argument("--conv-dir", required=True, dest="CONV_DIR", help="Folder containing converted files")
# parser.add_argument("--tfr-dir", required=True, dest="TFR_DIR", help="Folder containing converted records")
# parser.add_argument("--label-list", required=True, dest="LABEL_LIST", help="Location of labels json file")
parser.add_argument("--model-dir", required=True, dest="MODEL_DIR", help="Folder for model checkpointing")
parser.add_argument("--save-model-dir", dest="SAVE_MODEL_DIR", help="Folder for exporting saved model", default="")
parser.add_argument("--num-epochs", type=int, dest="NUM_EPOCHS", default=1)
parser.add_argument("--batch-size", type=int, dest="BATCH_SIZE", default=32)
# parser.add_argument("--train-steps", type=int, dest="TRAIN_STEPS", default=10000)
parser.add_argument("--max-train-steps", type=int, dest="MAX_TRAIN_STEPS", default=10000)
parser.add_argument("--eval-steps", type=int, dest="EVAL_STEPS", default=500)
parser.add_argument("--prefetch-buffer", type=int, dest="PREFETCH", default=-1)
parser.add_argument("--height", type=int, dest="HEIGHT", default=256)
parser.add_argument("--width", type=int, dest="WIDTH", default=256)
parser.add_argument("--channels", type=int, dest="CHANNELS", default=1)

args = parser.parse_args()
arguments = args.__dict__

TRAIN_FLAG = args.TRAIN_FLAG
EVALUATE_FLAG = args.EVALUATE_FLAG
# SAVE_MODEL_FLAG = args.SAVE_MODEL_FLAG
DISTRIBUTE_FLAG = args.DISTRIBUTE_FLAG

CONV_DIR = args.CONV_DIR
# TFR_DIR = args.TFR_DIR
# LABEL_LIST = args.LABEL_LIST
MODEL_DIR = args.MODEL_DIR
SAVE_MODEL_DIR = args.SAVE_MODEL_DIR
NUM_EPOCHS = int(args.NUM_EPOCHS)
BATCH_SIZE = int(args.BATCH_SIZE)
# TRAIN_STEPS = int(args.TRAIN_STEPS)
MAX_TRAIN_STEPS = int(args.MAX_TRAIN_STEPS)
EVAL_STEPS = int(args.EVAL_STEPS)
PREFETCH = int(args.PREFETCH)
HEIGHT = int(args.HEIGHT)
WIDTH = int(args.WIDTH)
CHANNELS = int(args.CHANNELS)

# if prefetch_buffer_size is None then TensorFlow will use an optimal prefetch buffer size automatically
if PREFETCH == -1:
	PREFETCH = None

# if save dir is not explicitly mentioned, it will be saved in model_checkpoint_dir/saved/ 
if SAVE_MODEL_DIR == "":
	SAVE_MODEL_DIR = os.path.join(MODEL_DIR, "saved")

TFR_DIR = os.path.join(CONV_DIR, "tfrecords")
LABEL_LIST = os.path.join(CONV_DIR, "labels.json")

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

try:
	with tf_reader.GFile(LABEL_LIST, 'rb') as fl:
		labels_bytes = fl.read()
		labels_json = labels_bytes.decode('utf8')

		labels = json.loads(labels_json)
		print(labels)
except Exception as e:
	print(str(e))
	exit(1)

print("Found %d training file records" % (len(training_filenames)))
print("Found %d testing file records" % (len(testing_filenames)))

print("Training:", training_filenames)
print("Testing:", testing_filenames)

model_est_fn = gen_cnn_model_fn(image_size=(HEIGHT, WIDTH, CHANNELS), num_classes=len(labels))

NUM_GPUS = get_available_gpus()
print("\n{0} GPUs available".format(NUM_GPUS))

if DISTRIBUTE_FLAG:
	print("Using Distributed Strategy.....")
	strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
	config = tf.estimator.RunConfig(train_distribute=strategy, eval_distribute=strategy)
else:
	config = tf.estimator.RunConfig()

model_classifier = tf.estimator.Estimator(
	model_fn=model_est_fn,
	model_dir=MODEL_DIR,
	config=config,
)

dataset_input_fn = data_utils.gen_input_fn(image_size=(HEIGHT, WIDTH, CHANNELS), num_classes=len(labels))

metadata = {
	'outputs' : [{
		'type': 'tensorboard',
		'source': MODEL_DIR,
	}]
}
with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as fl:
	json.dump(metadata, fl)

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
    	# steps=TRAIN_STEPS,
    	# hooks=[logging_hook]
    	)

if EVALUATE_FLAG:
	oct_test_in = lambda: dataset_input_fn(
		testing_filenames,
		batch_size=10)
	res = model_classifier.evaluate(input_fn=oct_test_in, steps=EVAL_STEPS)
	print(res)

# if SAVE_MODEL_FLAG:
serving_input_rcv_fn = data_utils.get_serving_input_receiver_fn(image_size=(HEIGHT, WIDTH, CHANNELS))

model_classifier.export_saved_model(
	SAVE_MODEL_DIR,
	serving_input_receiver_fn=serving_input_rcv_fn
)

# exporting metrics and tensorboard for kubeflow pipelines
metrics = {
	'metrics': [
		{
			'name': 'accuracy-score', # The name of the metric. Visualized as the column name in the runs table.
			'numberValue':  str(res['accuracy']), # The value of the metric. Must be a numeric value.
			'format': "PERCENTAGE",   # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
		},
		{
			'name': 'loss',
			'numberValue':  str(res['loss']), 
		},
		{
			'name': 'global-step',
			'numberValue':  str(res['global_step']),
		},
	]
}

with file_io.FileIO('/mlpipeline-metrics.json', 'w') as fl:
	json.dump(metrics, fl)



with file_io.FileIO('/mlpipeline-metrics.json', 'r') as fl:
	read_metrics = json.load(fl)
	print(read_metrics)

with file_io.FileIO('/mlpipeline-ui-metadata.json', 'r') as fl:
	read_meta = json.load(fl)
	print(read_meta)