import tensorflow as tf

def cnn_model_fn(labels):
	keras_vgg = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False)

	for layers in keras_vgg.layers[:-4]:
		layers.trainable = False

	output = keras_vgg.output
	output = tf.keras.layers.Flatten()(output)

	prediction = tf.keras.layers.Dense(len(labels), activation=tf.nn.softmax)(output)

	model = tf.keras.Model(inputs=keras_vgg.input, outputs=prediction)

	return model