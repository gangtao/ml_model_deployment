BIN_NAME ?= tf-model
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

deploy:
	kubectl apply -f deployment.yaml

undeploy:
	kubectl delete SeldonDeployment seldon-deployment-tf  -n mdt

predict_internal:
	curl -d 'json={"data":{"ndarray":[[5.9, 3.0, 5.1, 1.8]]}}' -X POST http://localhost:8001/api/v1/namespaces/mdt/services/tf-deployment-tf-predictor-tf-classifier:9000/proxy/predict

predict_external:
	curl -u oauth-key:oauth-secret --data '{"data":{"ndarray":[[5.9, 3.0, 5.1, 1.8]]}}' -X POST -H "Content-Type: application/json" -H " " http://localhost:8001/api/v1/namespaces/mdt/services/tf-deployment:8000/proxy/api/v0.1/predictions 
