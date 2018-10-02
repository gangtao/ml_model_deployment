
build:
	$(aws ecr get-login --no-include-email --region us-west-2)
	python main.py

train:
	python train.py

predict:
	curl -d 'json={"input":[5.9, 3.0, 5.1, 1.8]}' -X POST http://localhost:8001/api/v1/namespaces/mdt/services/query-frontend-at-default-cluster:1337/proxy/tf-classification/predict


