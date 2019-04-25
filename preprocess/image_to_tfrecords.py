import re
import os
from tfr_utils import _write_tf_records
from tfr_utils import build_example_list_tf
from tfr_utils import get_example_share
from tfr_utils import split_list

import multiprocessing as mp
pool = mp.Pool(mp.cpu_count())

import tensorflow as tf
import numpy as np
import tensorflow.gfile as tf_reader

import logging

from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("--inp-dir", required=True, dest="INP_DIR", help="Data folder for input images")
parser.add_argument("--ann-dir", dest="ANN_DIR", help="Data folder for annotations", default="")
parser.add_argument("--out-dir", required=True, dest="OUT_DIR", help="Destination folder for converted records")
parser.add_argument("--label-list", required=True, dest="LABEL_LIST", help="Location of labels json file")
parser.add_argument("--num-shards", type=int, dest="NUM_SHARDS", default=2)
parser.add_argument("--split-flag", type=int, dest="SPLIT_FLAG", default=1)
parser.add_argument("--train-split", type=float, dest="TRAIN_SPLIT", default=0.8)
parser.add_argument("--seed", type=int, dest="SEED", default=123)
parser.add_argument("--height", type=int, dest="HEIGHT", default=224)
parser.add_argument("--width", type=int, dest="WIDTH", default=224)
parser.add_argument("--fraction", type=float, dest="FRACTION", default=1)

args = parser.parse_args()

INP_DIR = args.INP_DIR
OUT_DIR = args.OUT_DIR
NUM_SHARDS = int(args.NUM_SHARDS)
SPLIT_FLAG = int(args.SPLIT_FLAG)
TRAIN_SPLIT = float(args.TRAIN_SPLIT)
SEED = int(args.SEED)
HEIGHT = int(args.HEIGHT)
WIDTH = int(args.WIDTH)
FRACTION = float(args.FRACTION)
LABEL_LIST = args.LABEL_LIST

if tf.gfile.IsDirectory(args.ANN_DIR):
	ANN_DIR = args.ANN_DIR
else:
	ANN_DIR = ""

BUCKET_NAME = ""
BUCKET_PATH = ""

def _fast_write_sharded_tfrs(examples, num_shards, output_dir, image_dims, is_training=1):
	sharded_examples = split_list(examples, num_shards)

	for count, shard in enumerate(sharded_examples, start=1):
		print("Starting %s shard %d" % ('train' if is_training else 'test', count))
		if is_training == 1:
			shard_prefix = "train"
		elif is_training == 0:
			shard_prefix = "test"
		else:
			shard_prefix = "data"
		output_filename = '{0}_{1:02d}_of_{2:02d}.tfrecord'.format(shard_prefix, count, num_shards)
		out_filepath = os.path.join(output_dir, output_filename)
		pool.apply_async(_write_tf_records, args=(shard, out_filepath, image_dims, ANN_DIR))

# MAIN

if (INP_DIR).startswith("gs://"):
	GCS_FLAG = 1

	try:
		BUCKET_NAME = re.search("gs://(.+?)/", (INP_DIR)).group(1)
	except AttributeError:
		logging.exception("Invalid GS bucket name")

	BUCKET_PATH = "gs://" + BUCKET_NAME
	print("Accessing Bucket: ", BUCKET_PATH)

else:
	GCS_FLAG = 0

if SPLIT_FLAG == 1 or SPLIT_FLAG == 0:
	if GCS_FLAG:
		print("\nReading from GCS Bucket:")
		examples, labels = build_example_list_tf(INP_DIR, SEED)
	else:
		print("\nReading from local directory:")
		examples, labels = build_example_list_tf(INP_DIR, SEED)

	print("Total Images found: ", len(examples))

	tar_size = int((FRACTION)*len(examples))
	examples = examples[:tar_size]
	print("Using:", len(examples), "images.")

	for k in examples:
		tmp = np.zeros(len(labels))
		tmp[k['label']] = 1
		k['oh_label'] = tmp

elif SPLIT_FLAG == 2:
	train_inp_dir = os.path.join(INP_DIR, "train")
	test_inp_dir = os.path.join(INP_DIR, "test")

	print("\nReading files........")
	train_examples, labels = build_example_list_tf(train_inp_dir, SEED)
	test_examples, test_labels = build_example_list_tf(test_inp_dir, SEED)

	print("Total Training Images found: ", len(train_examples))
	print("Total Testing Images found: ", len(test_examples))

	for example_list in [train_examples, test_examples]:
		tar_size = int((FRACTION)*len(example_list))
		example_list = example_list[:tar_size]
		print("Using:", len(example_list), "images.")

	for example_list in [train_examples, test_examples]:
		for k in example_list:
			tmp = np.zeros(len(labels))
			tmp[k['label']] = 1
			k['oh_label'] = tmp

	train_dir = os.path.join((OUT_DIR), "train")
	test_dir = os.path.join((OUT_DIR), "test")

	if not (OUT_DIR).startswith("gs://"):
		try:
			if not os.path.exists(train_dir):
				os.makedirs(train_dir)

			if not os.path.exists(test_dir):
				os.makedirs(test_dir)

		except Exception as e:
			print(e)
			exit(0)

	print("Creating training shards", flush=True)
	_fast_write_sharded_tfrs(train_examples, (NUM_SHARDS), train_dir, (HEIGHT, WIDTH), is_training=1)
	print("\n", flush=True)
	print("Creating testing shards", flush=True)
	_fast_write_sharded_tfrs(test_examples, (NUM_SHARDS), test_dir, (HEIGHT, WIDTH), is_training=0)
	print("\n", flush=True)

if SPLIT_FLAG == 1:

	train_examples, test_examples = get_example_share(examples, TRAIN_SPLIT)

	train_dir = os.path.join((OUT_DIR), "train")
	test_dir = os.path.join((OUT_DIR), "test")

	if not (OUT_DIR).startswith("gs://"):
		try:
			if not os.path.exists(train_dir):
				os.makedirs(train_dir)

			if not os.path.exists(test_dir):
				os.makedirs(test_dir)

		except Exception as e:
			print(e)
			exit(0)

	print("Creating training shards", flush=True)
	_fast_write_sharded_tfrs(train_examples, (NUM_SHARDS), train_dir, (HEIGHT, WIDTH), is_training=1)
	print("\n", flush=True)
	print("Creating testing shards", flush=True)
	_fast_write_sharded_tfrs(test_examples, (NUM_SHARDS), test_dir, (HEIGHT, WIDTH), is_training=0)
	print("\n", flush=True)
	
elif SPLIT_FLAG == 0:

	if not OUT_DIR.startswith("gs://"):
		try:
			if not os.path.exists(OUT_DIR):
				os.makedirs(OUT_DIR)

		except Exception as e:
			print(e)
			exit(0)

	print("Creating dataset shards", flush=True)
	_fast_write_sharded_tfrs(examples, NUM_SHARDS, OUT_DIR, (HEIGHT, WIDTH), is_training=2)
	print("\n", flush=True)

pool.close()
pool.join()

print("\nWriting Labels....\n")
import json
try:
	with tf_reader.GFile(LABEL_LIST, 'w') as fl:
		json.dump(labels, fl)
except Exception as e:
	print(str(e))

print("DONE")