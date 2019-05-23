import argparse
import datetime
import json
import os
import time
import logging
import requests
import subprocess
import six
from tensorflow.python.lib.io import file_io
import time
import yaml


def main(argv=None):
  parser = argparse.ArgumentParser(description='ML Trainer')
  parser.add_argument(
      '--model_name',
      help='...',
      required=True)

  parser.add_argument(
      '--num_gpus',
      help='...',
      default=0)

  parser.add_argument(
      '--model_path',
      help='...',
      required=True)

  parser.add_argument('--cluster', type=str,
                      help='GKE cluster set up for kubeflow. If set, zone must be provided. ' +
                           'If not set, assuming this runs in a GKE container and current ' +
                           'cluster is used.')
  parser.add_argument('--zone', type=str, help='zone of the kubeflow cluster.')
  args = parser.parse_args()

  KUBEFLOW_NAMESPACE = 'kubeflow'

  print("contents:")
  subprocess.call(['ls', '/secret/'])
  print("GOOGLE_APPLICATION_CREDENTIALS:")
  subprocess.call(['echo', '$GOOGLE_APPLICATION_CREDENTIALS'])
  print("user-gcp:")
  subprocess.call(['cat', '/secret/user-gcp-sa.json'])
  print("admin-gcp:")
  subprocess.call(['cat', '/secret/admin-gcp-sa.json'])

  # Make sure model dir exists before proceeding
  retries = 0
  sleeptime = 5
  while retries < 20:
    try:
      model_dir = os.path.join(args.model_path, file_io.list_directory(args.model_path)[-1])
      print("model subdir: %s" % model_dir)
      break
    except Exception as e:
      print(e)
      print("Sleeping %s seconds to sync with GCS..." % sleeptime)
      time.sleep(sleeptime)
      retries += 1
      sleeptime *= 2
  if retries >=20:
    print("could not get model subdir from %s, exiting" % args.model_path)
    exit(1)

  logging.getLogger().setLevel(logging.INFO)
  args_dict = vars(args)
  if args.cluster and args.zone:
    cluster = args_dict.pop('cluster')
    zone = args_dict.pop('zone')
  else:
    # Get cluster name and zone from metadata
    metadata_server = "http://metadata/computeMetadata/v1/instance/"
    metadata_flavor = {'Metadata-Flavor' : 'Google'}
    cluster = requests.get(metadata_server + "attributes/cluster-name",
                           headers = metadata_flavor).text
    zone = requests.get(metadata_server + "zone",
                        headers = metadata_flavor).text.split('/')[-1]


  logging.info('Cluster: {0}\nZone: {1}\n'.format(cluster, zone))


  logging.info('Getting credentials for GKE cluster %s.' % cluster)
  subprocess.call(['gcloud', 'container', 'clusters', 'get-credentials', cluster, '--zone', zone])

  args_list = ['--%s=%s' % (k.replace('_', '-'),v)
               for k,v in six.iteritems(args_dict) if v is not None]
  logging.info('Generating tfserving template.')


  template_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tfserve-template.yaml')
  target_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tf-serve.yaml')

  with open(template_file, 'r') as f:
    with open( target_file, "w" ) as target:
      data = f.read()
      changed = data.replace('MODEL_NAME',args.model_name)
      changed1 = changed.replace('KUBEFLOW_NAMESPACE',KUBEFLOW_NAMESPACE)
      changed2 = changed1.replace('MODEL_PATH', args.model_path)
      
      if int(args.num_gpus) > 0:
        changed3 = changed2.replace('TFSERVE_IMAGE', "tensorflow/serving:1.11.1-gpu")
        changed4 = changed3.replace('GPU_PLACEHOLDER', "nvidia.com/gpu: "+ str(args.num_gpus))      
      else:
        changed3 = changed2.replace('TFSERVE_IMAGE', "tensorflow/serving:1.11.1")
        changed4 = changed3.replace('GPU_PLACEHOLDER', "")

      target.write(changed4)

  logging.info(changed4)

  logging.info('deploying model serving.')
  subprocess.call(['kubectl', 'create', '-f', target_file])


if __name__== "__main__":
  main()

# tensorflow/serving:1.11.1-gpu