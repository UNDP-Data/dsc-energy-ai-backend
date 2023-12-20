# syntax=docker/dockerfile:1

FROM python:3.9.17-slim-bullseye

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN python3.9 -m venv venv
RUN python3.9 -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install bs4
RUN pip install openpyxl

COPY . /python-docker
ENV FLASK_APP=/python-docker/main.py
EXPOSE 5000

CMD [ "python3.9", "-m" , "flask", "run", "--host=0.0.0.0"]
