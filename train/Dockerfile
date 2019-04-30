FROM nvidia/cuda:10.0-cudnn7-devel

WORKDIR /app

COPY train.py ./

RUN apt-get update && \
	apt-get install python3 python3-pip -y && \
	pip3 install --upgrade pip && \
	pip install tensorflow-gpu

ENTRYPOINT ["python3", "train.py"]