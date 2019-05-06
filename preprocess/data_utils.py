import tensorflow as tf
import os
import random
import csv

def filenames_to_csv(input_dir, seed):
	examples = []
	labels = {}
	class_cnt = 0

	for classname in tf.gfile.ListDirectory(input_dir):
		if classname.endswith("/"):
			classname = classname[:-1]
		class_dir = os.path.join(input_dir, classname)
		# print(class_dir)
		
		if tf.gfile.IsDirectory(class_dir):	
			for filename in tf.gfile.ListDirectory(class_dir):
				filepath = os.path.join(class_dir, filename)

				line_to_write = [filepath, classname]
				examples.append(line_to_write)

			labels[class_cnt] = classname
			class_cnt = class_cnt + 1

	random.seed(seed)
	random.shuffle(examples)
	
	return examples, labels


def get_example_share(examples, train_split):
	example_len = len(examples)
	training_len = int(example_len*train_split)

	return [list(k) for k in np.split(examples, [training_len])]

def write_file_csv(targets, data_files):
	for step in targets:		
		try:
			with tf.gfile.GFile(data_files['{}_csv'.format(step)], 'w') as fl:
				writer = csv.writer(fl)
				writer.writerows(data_files['{}_examples'.format(step)])
		except Exception as e:
			print("ERROR: ", str(e))
			exit(1)