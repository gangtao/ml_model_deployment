FROM python:2.7-alpine3.8

MAINTAINER gang.tao@outlook.com

RUN apk add build-base
RUN mkdir /mlflow/
RUN pip install mlflow

EXPOSE 5000

CMD mlflow server \
    --file-store /mlflow \
    --host 0.0.0.0