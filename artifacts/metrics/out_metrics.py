import json

metrics = {
	'metrics': [
		{
			'name': 'accuracy-score', # The name of the metric. Visualized as the column name in the runs table.
			'numberValue':  str(0.859667), # The value of the metric. Must be a numeric value.
			'format': "PERCENTAGE",   # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
		},
		{
			'name': 'train-loss',
			'numberValue':  str(0.321564),
			'format': "PERCENTAGE", 
		},
	]
}

with open('/mlpipeline-metrics.json', 'w') as fl:
	json.dump(metrics, fl)

with open('/mlpipeline-metrics.json', 'r') as fl:
	read_metrics = json.load(fl)
	print(read_metrics)