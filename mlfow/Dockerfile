FROM continuumio/miniconda

MAINTAINER gang.tao@outlook.com

RUN mkdir /mlflow/
RUN pip install mlflow

EXPOSE 5000

CMD mlflow server \
    --file-store /mlflow \
    --host 0.0.0.0