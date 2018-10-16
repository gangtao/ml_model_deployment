FROM naughtytao/mlflow:0.1

MAINTAINER gang.tao@outlook.com

COPY ./mlruns/0/44ae85c084904b4ea5bad5aa42c9ce05/artifacts/model /model

EXPOSE 5000

CMD ["mlflow","sklearn","serve","-m","/model", "--host", "0.0.0.0"]