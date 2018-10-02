FROM pyspark-model:0.19
MAINTAINER gang.tao@outlook.com

RUN apt-get install -y --no-install-recommends software-properties-common
RUN add-apt-repository -y ppa:openjdk-r/ppa
RUN apt-get update
RUN apt-get install -y openjdk-8-jdk
RUN apt-get install -y openjdk-8-jre
RUN update-alternatives --config java
RUN update-alternatives --config javac
RUN update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java