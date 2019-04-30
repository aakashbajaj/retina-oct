FROM tensorflow/tensorflow:latest-py3

WORKDIR /app

COPY train.py ./

RUN	apt-get update && \
	apt-get install -y libxrender1 && \
	apt install -y libglib2.0-0 libsm6 libxext6

ENTRYPOINT ["python3", "train.py"]