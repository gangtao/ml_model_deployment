BIN_NAME ?= mlflow-sklearn-classification
VERSION ?= 0.1
IMAGE_NAME ?= $(BIN_NAME):$(VERSION)
DOCKER_ID_USER ?= naughtytao

docker:
	docker build --no-cache -t $(IMAGE_NAME) .

push:
	docker tag $(IMAGE_NAME) ${FULLNAME}
	docker push ${FULLNAME}

run:
	docker run -p 5000:5000 $(IMAGE_NAME)

serve:
	mlflow sklearn serve -m ./mlruns/0/44ae85c084904b4ea5bad5aa42c9ce05/artifacts/model/

test:
	curl -d '[[5.9, 3.0, 5.1, 1.8]]' -H 'Content-Type: application/json' -X POST localhost:5000/invocations