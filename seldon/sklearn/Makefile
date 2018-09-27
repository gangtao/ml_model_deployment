BIN_NAME ?= sklearn-model
VERSION ?= 0.1
IMAGE_NAME ?= $(BIN_NAME):$(VERSION)
DOCKER_ID_USER ?= naughtytao

FULLNAME=$(DOCKER_ID_USER)/${BIN_NAME}:${VERSION}

BUILD_IMAGE=seldonio/seldon-core-s2i-python3:0.2


build:
	s2i build . $(BUILD_IMAGE) $(IMAGE_NAME)

push:
	docker tag $(IMAGE_NAME) ${FULLNAME}
	docker push ${FULLNAME}

usage:
	s2i usage $(BUILD_IMAGE)

run:
	docker run --name "predictor" -d --rm -p 5000:5000 $(IMAGE_NAME)

stop:
	docker kill predictor

test: 
	curl  -d 'json={"data":{"ndarray":[[5.9, 3.0, 5.1, 1.8]]}}' http://0.0.0.0:5000/predict

