import os
import numpy as np
import tensorflow as tf
import tensorflow.gfile as tf_reader
from tensorflow.python.platform import tf_logging as logging
from functools import partial


import data_utils
import tf_model

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    dev_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return len(dev_list)

logging.set_verbosity(logging.INFO)
logging.log(logging.INFO, "Tensorflow version " + tf.__version__)

LABEL_LIST = "gs://kfp-testing/retin_oct/convkfp/labels.json"

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

num_classes = len(labels)

def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_decoded = image_decoded
    image_decoded.set_shape([224, 224, 3])
    return {"input_1": image_decoded}

def predict_input_fn(image_path):
    img_filenames = tf.constant(image_path)

    dataset = tf.data.Dataset.from_tensor_slices(img_filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(32)
    iterator = dataset.make_one_shot_iterator()
    image = iterator.get_next()

    return image

model = tf_model.cnn_model_fn(labels)

model.compile(loss='categorical_crossentropy',
		optimizer=tf.train.AdamOptimizer(),
		metrics=['accuracy'])

train_path = "gs://kfp-testing/retin_oct/convkfp/train"
test_path = "gs://kfp-testing/retin_oct/convkfp/test"

training_filenames = []
testing_filenames = []

# NUM_GPUS = 2
NUM_GPUS = get_available_gpus()
print("\n{0} GPUs available".format(NUM_GPUS))

strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
config = tf.estimator.RunConfig(train_distribute=strategy, model_dir="/tmp/mkkt/")

vgg_est = tf.keras.estimator.model_to_estimator(keras_model=model, config=config)

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


train_input = lambda: data_utils.dataset_input_fn(training_filenames, labels, num_epochs=1)
vgg_est.train(input_fn=train_input, steps=7000)

test_input = lambda: data_utils.dataset_input_fn(testing_filenames, labels)
res = vgg_est.evaluate(input_fn=test_input, steps=1)

print(res)

model_input_name = model.input_names[0]

def serving_input_receiver_fn():
    input_ph = tf.placeholder(tf.string, shape=[None], name='image_binary')
    images = tf.map_fn(partial(tf.image.decode_image, channels=1), input_ph, dtype=tf.uint8)
    images = tf.cast(images, tf.float32) / 255.
    images.set_shape([None, 224, 224, 1])
 
    return tf.estimator.export.ServingInputReceiver({model_input_name: images}, {'bytes': input_ph})
 
vgg_est.export_savedmodel("gs://kfp-testing/retin_oct/model_kkt", serving_input_receiver_fn=serving_input_receiver_fn)
