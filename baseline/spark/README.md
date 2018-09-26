
install
helm install --namespace {ns} --name {name} -f values.yaml stable/spark 

delete
helm del --purge {name}