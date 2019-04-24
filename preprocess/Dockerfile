FROM tensorflow/tensorflow:latest-py3

WORKDIR /app

COPY requirements.txt image_to_tfrecords.py tfr_utils.py ./

RUN	apt-get update && \
	pip install --no-cache-dir --upgrade -r requirements.txt && \
	apt-get install -y libxrender1 && \
	apt install -y libglib2.0-0 libsm6 libxext6

ENTRYPOINT ["python3", "image_to_tfrecords.py"]