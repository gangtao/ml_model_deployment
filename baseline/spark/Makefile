BIN_NAME ?= spark
VERSION ?= 2.3.1-glibc
IMAGE_NAME ?= $(BIN_NAME):$(VERSION)
DOCKER_ID_USER ?= naughtytao

FULLNAME=$(DOCKER_ID_USER)/${BIN_NAME}:${VERSION}

docker: Dockerfile
	docker build --no-cache -t $(IMAGE_NAME) .

push:
	docker tag $(IMAGE_NAME) ${FULLNAME}
	docker push ${FULLNAME}

deploy:
	helm install --namespace mdt --name spark -f values.yaml stable/spark 

undeploy:
	helm del --purge spark