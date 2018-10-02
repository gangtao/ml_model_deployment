from sklearn.externals import joblib
from clipper_admin import ClipperConnection, KubernetesContainerManager
from clipper_admin.deployers import python as python_deployer

K8S_ADDR = '127.0.0.1:8001'
K8S_NS = 'mdt'

APP_NAME = 'sklearn-classification'
MODEL_NAME = 'classification.pkl'
PREDICT_NAME = 'clipper-sklearn-predict'
VERSION = 16

clf = joblib.load(MODEL_NAME)


def predict_wrapper(X):
    print("inputs {}".format(X))
    try:
        result = clf.predict(X)
        print("result is {}".format(result))
        ret = [str(i) for i in result]
        print("return is {}".format(ret))
        return ret
    except Exception as e:
        print(e)
        return [str(e)]


manager = KubernetesContainerManager(
    kubernetes_proxy_addr=K8S_ADDR, namespace=K8S_NS)
clipper_conn = ClipperConnection(manager)
clipper_conn.connect()


# clipper_conn.delete_application(APP_NAME)
# clipper_conn.register_application(
#    name = APP_NAME, input_type = 'doubles', default_output = '0', slo_micros = 100000000)


python_deployer.deploy_python_closure(clipper_conn,
                                      name = PREDICT_NAME,
                                      version = VERSION,
                                      input_type = "doubles",
                                      func = predict_wrapper,
                                      registry = '658391232643.dkr.ecr.us-west-2.amazonaws.com',
                                      pkgs_to_install = ['sklearn'])

# clipper_conn.link_model_to_app(app_name=APP_NAME, model_name=PREDICT_NAME)
