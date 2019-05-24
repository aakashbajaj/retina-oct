FROM google/cloud-sdk:latest

WORKDIR /app

COPY requirements.txt data_utils.py setup.py generate_csv.py prep_df.py ./

RUN	apt-get update && \
	apt-get install python3 python3-pip && \
	pip3 install --upgrade pip && \
	pip install --no-cache-dir --upgrade -r requirements.txt && \
	apt-get install -y libxrender1 && \
	apt install -y libglib2.0-0 libsm6 libxext6

ENTRYPOINT ["python3", "setup.py"]