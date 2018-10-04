
server:
	docker run -p 65327:65327 -v /tmp/models:/models combustml/mleap-serving:0.9.0-SNAPSHOT 

load:
	curl -XPUT -H "content-type: application/json" \
		-d '{"path":"/models/mleap-scikit-test-pipeline.zip"}' \
		http://localhost:65327/model

scheme:
	curl -XGET -H "content-type: application/json" \
		http://localhost:65327/model

transform:
	curl -XPOST -H "accept: application/json" \
		-H "content-type: application/json" \
		-d @./payload.json \
		http://localhost:65327/transform