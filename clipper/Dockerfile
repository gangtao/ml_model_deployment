FROM continuumio/anaconda
MAINTAINER gtao@splunk.com

RUN pip install boto3 clipper_admin cloudpickle==0.5.3 
RUN /opt/conda/bin/conda install jupyter -y --quiet
RUN mkdir /opt/notebooks
COPY ./*.ipynb /opt/notebooks/

COPY ./start.sh /
RUN chmod +x /start.sh

COPY ./current.config /root/.kube/config

WORKDIR /
EXPOSE 8888

CMD ["/bin/sh","./start.sh"]