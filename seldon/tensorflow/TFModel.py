from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.tensor_forest.python import tensor_forest

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TFModel(object):
    """
    Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """

    def __init__(self):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """
        pass

    def predict(self, X, features_names):
        """
        Return a prediction.

        Parameters
        ----------
        X : array-like
        feature_names : array of feature names (optional)
        """
        graph = tf.get_default_graph()
        with tf.Session() as sess:
            self.saver = tf.train.import_meta_graph(
                './classification_model.ckpt.meta')
            self.saver.restore(sess, './classification_model.ckpt')
            #input = graph.get_operation_by_name("train")
            # print(graph.as_graph_def())
            load_infer_op = graph.get_tensor_by_name('probabilities:0')
            accuracy_op = graph.get_tensor_by_name('Mean_1:0')
            oX = graph.get_tensor_by_name('Placeholder:0')
            oY = graph.get_tensor_by_name('Placeholder_1:0')

            return sess.run(load_infer_op, feed_dict={oX: X})
