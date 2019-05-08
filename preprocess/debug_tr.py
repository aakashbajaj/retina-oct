import tensorflow as tf


class ImageCoder(object):
	"""Helper class that provides TensorFlow image coding utilities."""

	def __init__(self, height=224, width=224, channels=1):
		# Create a single Session to run all image coding calls.
		self._sess = tf.Session()

		# Initializes function that decodes RGB JPEG data.
		self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
		self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=channels)
		# self._batch_img = tf.expand_dims(self._decode_jpeg, 0)
		self._resize_area = tf.image.resize_images(self._decode_jpeg, size=[height, width], method=tf.image.ResizeMethod.AREA)

	def decode_jpeg(self, image_data):
		image = self._sess.run(self._resize_area, feed_dict={self._decode_jpeg_data: image_data})
		tf.print(image.shape)
		
		assert len(image.shape) == 3
		assert image.shape[2] == 1
		return image

	# def __del__(self):
	# 	self._sess.close()


def _get_image_data(filename, coder):
	"""Process a single image file.

	Args:
	filename: string, path to an image file e.g., '/path/to/example.JPG'.
	coder: instance of ImageCoder to provide TensorFlow image coding utils.
	Returns:
	image_buffer: string, JPEG encoding of RGB image.
	height: integer, image height in pixels.
	width: integer, image width in pixels.
	"""
	# Read the image file.
	with tf.gfile.GFile(filename, 'rb') as ifp:
		image_data = ifp.read()

	#decode(image_data)
	#resize(image_data)

	# Decode the RGB JPEG.
	image = coder.decode_jpeg(image_data)

	# Check that image converted to RGB
	assert len(image.shape) == 3
	height = image.shape[0]
	width = image.shape[1]
	assert image.shape[2] == 1

	return image, height, width

if __name__ == '__main__':
	coder = ImageCoder()

	filename = "/home/techno/docker_mnt/dog_data/Image_large/n02085620-Chihuahua/n02085620_3488.jpg"

	image, ht, wt = _get_image_data(filename, coder)

	print(image.shape)
	print(ht, wt)