BIN_NAME ?= spark
VERSION ?= 2.3
IMAGE_NAME ?= $(BIN_NAME):$(VERSION)
DOCKER_ID_USER ?= naughtytao

FULLNAME=$(DOCKER_ID_USER)/${BIN_NAME}:${VERSION}

docker: Dockerfile
	docker build --no-cache -t $(IMAGE_NAME) .

push:
	docker tag $(IMAGE_NAME) ${FULLNAME}
	docker push ${FULLNAME}