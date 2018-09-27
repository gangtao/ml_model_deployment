from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import numpy as np

# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

data = load_iris()
dX, dy = data["data"], data["target"]
X_train, X_test, y_train, y_test = train_test_split(
    dX, dy, test_size=0.33, random_state=42)

# Parameters
num_steps = 500  # Total steps to train
batch_size = 10  # The number of samples per batch
num_classes = 3  # The 10 digits
num_features = 4  # Each image is 28x28 pixels
num_trees = 10
max_nodes = 100

# Input and Target data
X = tf.placeholder(tf.float32, shape=[None, num_features])
# For random forest, labels must be integers (the class id)
Y = tf.placeholder(tf.int32, shape=[None])

# Random Forest Parameters
hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)
# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

# Measure the accuracy
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value) and forest resources
init_vars = tf.group(tf.global_variables_initializer(),
                     resources.initialize_resources(resources.shared_resources()))


def next_batch(size):
    index = range(len(X_train))
    index_batch = np.random.choice(index, size)
    return X_train[index_batch], y_train[index_batch]


# Start TensorFlow session
sess = tf.Session()

# Run the initializer
sess.run(init_vars)

saver = tf.train.Saver()

# Training
for i in range(1, num_steps + 1):
    # Prepare Data
    # Get the next batch of MNIST data (only images are needed, not labels)
    batch_x, batch_y = next_batch(batch_size)
    _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
    if i % 50 == 0 or i == 1:
        acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))
# Test Model
print("Test Accuracy:", sess.run(
    accuracy_op, feed_dict={X: X_test, Y: y_test}))

# Print the tensors related to this model
print(accuracy_op)
print(infer_op)
print(X)
print(Y)

# save the model to a check point file
save_path = saver.save(sess, "/tmp/model.ckpt")
