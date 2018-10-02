BIN_NAME ?= conda-clipper
VERSION ?= 0.3
IMAGE_NAME ?= $(BIN_NAME):$(VERSION)
DOCKER_ID_USER ?= naughtytao

FULLNAME=$(DOCKER_ID_USER)/${BIN_NAME}:${VERSION}

gen_config:
	kubectl config view current > current.config

docker:
	docker build --no-cache -t $(IMAGE_NAME) .

push:
	docker tag $(IMAGE_NAME) ${FULLNAME}
	docker push ${FULLNAME}

deploy_conda:
	kubectl apply -f conda_deploy.yaml

install:
	pip install clipper_admin
	pip install cloudpickle==0.5.3 --user

deploy:
	python deploy.py --op d

undeploy:
	python deploy.py --op u
