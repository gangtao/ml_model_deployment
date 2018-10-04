from __future__ import print_function

import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# note: this has to be imported in case to support forest graph
from tensorflow.contrib.tensor_forest.python import tensor_forest

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

saver = tf.train.import_meta_graph('/tmp/model_01.ckpt.meta')

data = load_iris()

dX, dy = data["data"], data["target"]

graph = tf.get_default_graph()
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('/tmp/model.ckpt.meta')
    new_saver.restore(sess, '/tmp/model.ckpt')
    #input = graph.get_operation_by_name("train")
    # print(graph.as_graph_def())
    load_infer_op = graph.get_tensor_by_name('probabilities:0')
    accuracy_op = graph.get_tensor_by_name('Mean_1:0')
    X = graph.get_tensor_by_name('Placeholder:0')
    Y = graph.get_tensor_by_name('Placeholder_1:0')
    print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: dX, Y: dy}))
    result = sess.run(load_infer_op, feed_dict={X: dX})
    prediction_result = [i.argmax() for i in result]
    print(classification_report(dy, prediction_result,
                                target_names=data["target_names"]))
