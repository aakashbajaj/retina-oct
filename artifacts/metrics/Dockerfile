FROM python:3.5

WORKDIR /app

COPY out_metrics.py ./

RUN apt-get update

ENTRYPOINT ["python", "out_metrics.py"]