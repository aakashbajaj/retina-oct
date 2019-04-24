import tensorflow as tf
import os
import time

tf.logging.set_verbosity(tf.logging.INFO)

class TimeHistory(tf.train.SessionRunHook):
	def begin(self):
		self.times = []
	def before_run(self, run_context):
		self.iter_time_start = time.time()
	def after_run(self, run_context, run_values):
		self.times.append(time.time() - self.iter_time_start)

def input_fn(file_pattern, labels, image_size=(224,224), shuffle=False,	batch_size=64,	num_epochs=None, buffer_size=4096,
		prefetch_buffer_size=None):


	table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(labels))
	num_classes = len(labels)

	def _map_func(filename):
		label = tf.string_split([filename], delimiter=os.sep).values[-2]
		image = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		image = tf.image.resize_images(image, size=image_size)
		return (image, tf.one_hot(table.lookup(label), num_classes))

	dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)

	if num_epochs is not None and shuffle:
		dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size, num_epochs))
	elif shuffle:
		dataset = dataset.shuffle(buffer_size)
	elif num_epochs is not None:
		dataset = dataset.repeat(num_epochs)

	dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=_map_func,
								batch_size=batch_size,
								num_parallel_calls=os.cpu_count()))
	dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

	return dataset




train_folder = os.path.join("oct_data", "train", "**", "*.jpeg")
test_folder = os.path.join("oct_data", "test", "**", "*.jpeg")

labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

keras_vgg = tf.keras.applications.VGG16(input_shape=(224,224,3), include_top=False)

output = keras_vgg.output
output = tf.keras.layers.Flatten()(output)
prediction = tf.keras.layers.Dense(len(labels), activation=tf.nn.softmax)(output)


model = tf.keras.Model(inputs=keras_vgg.input, outputs=prediction)

for layers in keras_vgg.layers[:-4]:
	layers.trainable = False

model.compile(loss='categorical_crossentropy',
		optimizer=tf.train.AdamOptimizer(),
		metrics=['accuracy'])

NUM_GPUS = 2


strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
config = tf.estimator.RunConfig(train_distribute=strategy)
estimator = tf.keras.estimator.model_to_estimator(model, config=config)

BATCH_SIZE = 32
EPOCHS = 1

time_hist = TimeHistory()
#logging_hook = tf.train.LoggingTensorHook({"loss" : loss, "accuracy" : accuracy}, every_n_iter=10)

oct_train_in = lambda:input_fn(train_folder, labels, shuffle=True, batch_size=BATCH_SIZE, buffer_size=2048, num_epochs=EPOCHS, prefetch_buffer_size=4)
train_spec = tf.estimator.TrainSpec(input_fn=oct_train_in, max_steps=10000)

# estimator.train(input_fn=oct_train_in, hooks=[time_hist])

oct_test_in = lambda:input_fn(test_folder, labels, shuffle=False, batch_size=BATCH_SIZE, buffer_size=1024, num_epochs=1)
eval_spec = tf.estimator.EvalSpec(input_fn=oct_test_in)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
#estimator.evaluate(input_fn=oct_test_in)
