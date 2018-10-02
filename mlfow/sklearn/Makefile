serve:
	mlflow sklearn serve -m ./mlruns/0/44ae85c084904b4ea5bad5aa42c9ce05/artifacts/model/

test:
	curl -d '[[5.9, 3.0, 5.1, 1.8]]' -H 'Content-Type: application/json' -X POST localhost:5000/invocations