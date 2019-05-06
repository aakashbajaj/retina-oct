import tensorflow as tf
import data_utils
import argparse
import json
import os

import logging

import tensorflow.gfile as tf_reader

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("--input-dir", required=True)
	parser.add_argument("--csv-list", default="/tmp/")
	parser.add_argument("--label-list", required=True, help="Location of labels json file")
	parser.add_argument("--split-flag", type=int, default=2)
	parser.add_argument("--train-split", type=float, default=0.8)
	parser.add_argument("--seed", type=int, default=123)

	args = parser.parse_args()
	arguments = args.__dict__

	INPUT_DIR = arguments['input_dir']
	SPLIT_FLAG = arguments['split_flag']
	SEED = arguments['seed']
	TRAIN_SPLIT = arguments['train_split']
	LABEL_LIST = arguments['label_list']

	data_files = dict()

	CSV_FILE_LOCATION = arguments['csv_list']
	print("\n Saving CSV list to {}".format(CSV_FILE_LOCATION))

	# no split, direct data write
	if SPLIT_FLAG == 0:
		targets = ['data']

		data_files['data_csv'] = os.path.join(CSV_FILE_LOCATION, "data_list.csv")
		print(data_files['data_csv'])
		data_files['data_examples'], data_files['labels'] = data_utils.filenames_to_csv(INPUT_DIR, SEED)
 
 	# common file list in input_dir, split list into train and test
	elif SPLIT_FLAG == 1:
		common_examples , data_files['labels'] = data_utils.filenames_to_csv(INPUT_DIR, SEED)

		# split the example list
		data_files['train_examples'], data_files['test_examples'] = data_utils.get_example_share(common_examples, TRAIN_SPLIT)
		
		targets = ['train', 'test']
		for step in targets:
			data_files['{}_csv'.format(step)] = os.path.join(CSV_FILE_LOCATION, "{}_list.csv".format(step))
			print(data_files['{}_csv'.format(step)])

	# already train and test dir in input_dir
	elif SPLIT_FLAG == 2:
		targets = ['train', 'test']

		for step in targets:
			data_files['{}_csv'.format(step)] = os.path.join(CSV_FILE_LOCATION, "{}_list.csv".format(step))
			print(data_files['{}_csv'.format(step)])
			data_dir = os.path.join(INPUT_DIR, step)
			print(data_dir)
			data_files['{}_examples'.format(step)], data_files['labels'] = data_utils.filenames_to_csv(data_dir, SEED)

	else:
		print("\n Invalid Split Flag value! Exiting.....")
		exit(1)

	data_utils.write_file_csv(targets, data_files)

	print(data_files['labels'])

	print("\nWriting Labels to {}\n".format(LABEL_LIST))

	try:
		with tf_reader.GFile(LABEL_LIST, 'w') as fl:
			json.dump(data_files['labels'], fl)
	except Exception as e:
		print(str(e))