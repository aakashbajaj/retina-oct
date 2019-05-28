import subprocess
import argparse
import os

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("--input-dir", required=True)
	# parser.add_argument("--csv-list", default="/tmp/")
	# parser.add_argument("--label-location", required=True, help="Directory to save labels json file")
	parser.add_argument("--dataprep-flag", type=int, default=2)
	parser.add_argument("--split-flag", type=int, default=2)
	parser.add_argument("--train-split", type=float, default=0.8)
	parser.add_argument("--seed", type=int, default=123)


	# parser.add_argument("--train-csv", required=True)
	# parser.add_argument("--test-csv")
	parser.add_argument("--output-dir", required=True, help="Converted files will be saved to tfrecords/ dir inside output-dir")
	parser.add_argument('--project-id',	required=True)
	parser.add_argument('--runner',	default=None)
	parser.add_argument("--num-shards", type=int, default=5)
	parser.add_argument("--height", type=int, default=256)
	parser.add_argument("--width", type=int, default=256)

	args = parser.parse_args()
	arguments = args.__dict__

	if int(arguments['dataprep_flag']) == 1:
		gen_csv_cmd = ('python3 generate_csv.py --input-dir {0} --csv-list {1} --label-list {2} --split-flag {3} --train-split {4} --seed {5}').format(
			arguments['input_dir'],
			os.path.join(arguments['output_dir'], "file_list_csv"),
			os.path.join(arguments['output_dir'], "labels.json"),
			arguments['split_flag'],
			arguments['train_split'],
			arguments['seed']
		)
		
		try:
			print("\nCalling: {}\n".format(gen_csv_cmd))
			subprocess.check_call(gen_csv_cmd.split())
		except Exception as e:
			print("\n Error in generating csv:\n", str(e))
			exit(1)

		if arguments['split_flag'] == 2:
			df_prep_cmd = 'python3 prep_df.py --train-csv {0} --test-csv {1} --output-dir {2} --label-list {3} --project-id {4} --runner {5} --num-shards {6} --split-flag {7} --height {8} --width {9} --input-dir {10}'.format(
				# os.path.join(arguments['csv_list'], "train_list.csv"),
				# os.path.join(arguments['csv_list'], "test_list.csv"),
				os.path.join(arguments['output_dir'], "file_list_csv/train_list.csv"),
				os.path.join(arguments['output_dir'], "file_list_csv/test_list.csv"),
				os.path.join(arguments['output_dir'], "tfrecords"),
				os.path.join(arguments['output_dir'], "labels.json"),
				arguments['project_id'],
				arguments['runner'],
				arguments['num_shards'],
				arguments['split_flag'],
				arguments['height'],
				arguments['width'],
				arguments['input_dir']
			)

		elif arguments['split_flag'] == 0:
			df_prep_cmd = 'python3 prep_df.py --train-csv {0} --output-dir {1} --label-list {2} --project-id {3} --runner {4} --num-shards {5} --split-flag {6} --height {7} --width {8} --input-dir {9}'.format(
				# os.path.join(arguments['csv_list'], "data_list.csv"),
				# os.path.join(arguments['output_dir'], "tfrecords"),
				# os.path.join(arguments['label_location'], "labels.json"),
				os.path.join(arguments['output_dir'], "file_list_csv/data_list.csv"),
				os.path.join(arguments['output_dir'], "tfrecords"),
				os.path.join(arguments['output_dir'], "labels.json"),
				arguments['project_id'],
				arguments['runner'],
				arguments['num_shards'],
				arguments['split_flag'],
				arguments['height'],
				arguments['width'],
				arguments['input_dir']
			)

		else:
			print("\nInvalid split flag value. Exiting......")
			exit(1)

		try:
			print("\nCalling: {}\n".format(df_prep_cmd))
			subprocess.check_call(df_prep_cmd.split())
		except Exception as e:
			print("\n Error in running pipeline : {}\n".format(str(e)))
			exit(1)

	else:
		print("\nSkipping dataprep")
		exit(0)