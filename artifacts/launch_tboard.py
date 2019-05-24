import json

from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("--model-dir", required=True, dest="MODEL_DIR", help="Folder for model checkpointing")
# parser.add_argument("--save-model-dir", dest="SAVE_MODEL_DIR", help="Folder for exporting saved model", default="")

args = parser.parse_args()
MODEL_DIR = args.MODEL_DIR
# SAVE_MODEL_DIR = args.SAVE_MODEL_DIR

metadata = {
	'outputs' : [{
		'type': 'tensorboard',
		'source': MODEL_DIR,
	}]
}
with open('/mlpipeline-ui-metadata.json', 'w') as fl:
	json.dump(metadata, fl)

with open('/mlpipeline-ui-metadata.json', 'r') as fl:
	read_meta = json.load(fl)
	print(read_meta)