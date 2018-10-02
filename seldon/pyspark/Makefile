BIN_NAME ?= pyspark-model
VERSION ?= 0.19
IMAGE_NAME ?= $(BIN_NAME):$(VERSION)
DOCKER_ID_USER ?= naughtytao

FULLNAME=$(DOCKER_ID_USER)/${BIN_NAME}:${VERSION}

BUILD_IMAGE=seldonio/seldon-core-s2i-python3:0.2


build:
	s2i build . $(BUILD_IMAGE) $(IMAGE_NAME)
	docker build --no-cache -t $(BIN_NAME)-wrapper:$(VERSION) .

push:
	docker tag $(BIN_NAME)-wrapper:$(VERSION) $(DOCKER_ID_USER)/${BIN_NAME}-wrapper:${VERSION}
	docker push $(DOCKER_ID_USER)/${BIN_NAME}-wrapper:${VERSION}

usage:
	s2i usage $(BUILD_IMAGE)

deploy:
	kubectl apply -f deployment.yaml

undeploy:
	kubectl delete SeldonDeployment seldon-deployment-pyspark  -n mdt

predict_internal:
	curl -d 'json={"data":{"ndarray":[[5.9, 3.0, 5.1, 1.8]]}}' -X POST http://localhost:8001/api/v1/namespaces/mdt/services/pyspark-deployment-pyspark-predictor-pyspark-classifier:9000/proxy/predict

predict_external:
	curl -u oauth-key:oauth-secret --data '{"data":{"ndarray":[[5.9, 3.0, 5.1, 1.8]]}}' -X POST -H "Content-Type: application/json" -H " " http://localhost:8001/api/v1/namespaces/mdt/services/pyspark-deployment:8000/proxy/api/v0.1/predictions 
