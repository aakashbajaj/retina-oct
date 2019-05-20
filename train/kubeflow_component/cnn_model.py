import tensorflow as tf

def gen_cnn_model_fn(image_size, num_classes):

	def cnn_model_fn(features, labels, mode):
		image_tensor = features["image"]

		input_layer = tf.reshape(image_tensor, [-1,image_size[0],image_size[1],image_size[2]])

		conv1 = tf.layers.conv2d(
			inputs=input_layer,
			filters=256,
			kernel_size=[3,3],
			padding="valid",
			activation=tf.nn.relu)

		# bn1 = tf.layers.batch_normalization(inputs=conv1, training=(mode == tf.estimator.ModeKeys.TRAIN))
		mp1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
		drp1 = tf.layers.dropout(inputs=mp1, rate=0.25, training=(mode == tf.estimator.ModeKeys.TRAIN))

		conv2 = tf.layers.conv2d(
			inputs=drp1,
			filters=128,
			kernel_size=[3,3],
			padding="valid",
			activation=tf.nn.relu)

		# bn2 = tf.layers.batch_normalization(inputs=conv2, training=(mode == tf.estimator.ModeKeys.TRAIN))
		mp2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
		drp2 = tf.layers.dropout(inputs=mp2, rate=0.25, training=(mode == tf.estimator.ModeKeys.TRAIN))

		conv3 = tf.layers.conv2d(
			inputs=drp2,
			filters=64,
			kernel_size=[3,3],
			padding="valid",
			activation=tf.nn.relu)

		# bn3 = tf.layers.batch_normalization(inputs=conv3, training=(mode == tf.estimator.ModeKeys.TRAIN))
		drp3 = tf.layers.dropout(inputs=conv3, rate=0.25, training=(mode == tf.estimator.ModeKeys.TRAIN))

		flt = tf.layers.flatten(inputs=drp3)

		dns1 = tf.layers.dense(inputs=flt, units=32, activation=tf.nn.relu)
		# dense_bn = tf.layers.batch_normalization(inputs=dns1, training=(mode == tf.estimator.ModeKeys.TRAIN))
		drp4 = tf.layers.dropout(inputs=dns1, rate=0.5, training=(mode == tf.estimator.ModeKeys.TRAIN))

		logits = tf.layers.dense(inputs=drp4, units=num_classes)

		predictions = {
			"classes": tf.argmax(input=logits, axis=1),
			"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
		}

		if mode == tf.estimator.ModeKeys.PREDICT:
			return tf.estimator.EstimatorSpec(
				mode=mode,
				predictions=predictions,
				export_outputs={
					'predict': tf.estimator.export.PredictOutput(predictions)}
				)

		# fix_labels = tf.stop_gradient(labels)

		# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=fix_labels, logits=logits))
		# loss = (tf.nn.softmax_cross_entropy_with_logits_v2(labels=fix_labels, logits=logits))
		# loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
		loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

		# accuracy = tf.metrics.accuracy(
		# 	labels=labels, predictions=predictions["classes"])

		# logging_hook = tf.train.LoggingTensorHook({"loss" : loss, "accuracy" : accuracy[1]}, every_n_iter=50)

		if mode == tf.estimator.ModeKeys.TRAIN:
			accuracy =  tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
			tf.summary.scalar('accuracy', accuracy[1])
			eval_train_metrics = {
				"accuracy": accuracy
			}
			optimizer = tf.train.AdamOptimizer()
			train_op = optimizer.minimize(
				loss=loss,
				global_step=tf.train.get_global_step())

			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			train_op = tf.group([train_op, update_ops])

			return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_train_metrics,
				# training_hooks = [logging_hook]
				)

		eval_metric_ops = {
			"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
		}
		
		return tf.estimator.EstimatorSpec(
			mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

	return cnn_model_fn