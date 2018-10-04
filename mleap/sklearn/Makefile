BIN_NAME ?= mleap-sklearn-classification
VERSION ?= 0.1
IMAGE_NAME ?= $(BIN_NAME):$(VERSION)
DOCKER_ID_USER ?= naughtytao

train:
	python train.py

docker:
	docker build --no-cache -t $(IMAGE_NAME) .

push:
	docker tag $(IMAGE_NAME) ${FULLNAME}
	docker push ${FULLNAME}

server:
	docker run -p 65327:65327 ${IMAGE_NAME} 

load:
	curl -XPUT -H "content-type: application/json" \
		-d '{"path":"/models/mleap-scikit-rf-pipeline.zip"}' \
		http://localhost:65327/model

scheme:
	curl -XGET -H "content-type: application/json" \
		http://localhost:65327/model

transform:
	curl -XPOST -H "accept: application/json" \
		-H "content-type: application/json" \
		-d @./payload.json \
		http://localhost:65327/transform
