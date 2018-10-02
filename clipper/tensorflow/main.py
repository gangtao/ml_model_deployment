import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest

from clipper_admin import ClipperConnection, KubernetesContainerManager
from clipper_admin.deployers.tensorflow import deploy_tensorflow_model

K8S_ADDR = '127.0.0.1:8001'
K8S_NS = 'mdt'

APP_NAME = 'tf-classification'
PREDICT_NAME = 'clipper-tf-predict'
VERSION = 3

REGISTRY = '658391232643.dkr.ecr.us-west-2.amazonaws.com'


sess = tf.Session()
graph = tf.get_default_graph()
saver = tf.train.import_meta_graph(
    './model.ckpt.meta')
saver.restore(sess, './model.ckpt')
load_infer_op = graph.get_tensor_by_name('probabilities:0')
accuracy_op = graph.get_tensor_by_name('Mean_1:0')
oX = graph.get_tensor_by_name('Placeholder:0')
oY = graph.get_tensor_by_name('Placeholder_1:0')

def predict(X):
    print("inputs {}".format(X))
    result = sess.run(load_infer_op, feed_dict={oX: X})
    ret = [str(i) for i in result]
    print("return is {}".format(ret))
    return ret


manager = KubernetesContainerManager(
    kubernetes_proxy_addr=K8S_ADDR, namespace=K8S_NS)
clipper_conn = ClipperConnection(manager)
clipper_conn.connect()


# clipper_conn.delete_application(APP_NAME)
# clipper_conn.register_application(
#   name = APP_NAME, input_type = 'doubles', default_output = '0', slo_micros = 100000000)


deploy_tensorflow_model(clipper_conn,
                        name=PREDICT_NAME,
                        version=VERSION,
                        input_type="doubles",
                        func=predict,
                        tf_sess_or_saved_model_path=sess,
                        registry=REGISTRY,
                        pkgs_to_install=['tensorflow'])

clipper_conn.link_model_to_app(app_name=APP_NAME, model_name=PREDICT_NAME)
