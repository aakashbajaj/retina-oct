FROM google/cloud-sdk:latest

WORKDIR /app

COPY deploy_tfserve.py tfserve-template_GPU.yaml tfserve-template_GPU.yaml ./

RUN apt-get update && \
	apt-get install python3 python3-pip -y && \
	pip3 install --upgrade pip && \
	pip install tensorflow && \
	pip install requests pyyaml

ENTRYPOINT ["python3", "deploy_tfserve.py"]