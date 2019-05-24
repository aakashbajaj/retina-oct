FROM python:3.5

WORKDIR /app

COPY launch_tboard.py ./

RUN apt-get update

ENTRYPOINT ["python", "launch_tboard.py"]