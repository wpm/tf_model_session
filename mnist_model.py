import argparse
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from model_session import ModelSession


class MNISTModelSession(ModelSession):
    """
    An example ModelSession for MNIST digit classification.
    """
    IMAGE_SIZE = 28
    DIGITS = 10

    @staticmethod
    def create_graph(layer_1=32, layer_2=64):
        def weight_variable(shape):
            return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

        def bias_variable(shape, name=None):
            return tf.Variable(tf.constant(0.1, shape=shape), name=name)

        def conv2d(x, w):
            return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        iteration = tf.Variable(initial_value=0, trainable=False, name="iteration")
        with tf.variable_scope("parameters"):
            x = tf.placeholder(tf.float32, shape=[None, MNISTModelSession.IMAGE_SIZE * MNISTModelSession.IMAGE_SIZE],
                               name="x")
            y = tf.placeholder(tf.float32, shape=[None, MNISTModelSession.DIGITS], name="y")
            keep_probability = tf.placeholder(tf.float32, name="keep_probability")
            learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        with tf.variable_scope("convolution_1"):
            x_image = tf.reshape(x, [-1, MNISTModelSession.IMAGE_SIZE, MNISTModelSession.IMAGE_SIZE, 1])
            w_conv1 = weight_variable([5, 5, 1, layer_1])
            b_conv1 = bias_variable([layer_1], "bias")
            h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)
        with tf.variable_scope("convolution_2"):
            w_conv2 = weight_variable([5, 5, layer_1, layer_2])
            b_conv2 = bias_variable([layer_2], "bias")
            h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)
        with tf.variable_scope("fully_connected"):
            w_fc1 = weight_variable([7 * 7 * layer_2, 1024])
            b_fc1 = bias_variable([1024])
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * layer_2])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_probability)
        with tf.variable_scope("readout"):
            w_fc2 = weight_variable([1024, MNISTModelSession.DIGITS])
            b_fc2 = bias_variable([MNISTModelSession.DIGITS])
            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
        with tf.variable_scope("train"):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv), reduction_indices=[1]))
            tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=iteration, name="train_step")
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
            tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

    def __str__(self):
        return "MNIST Model(%d, %d, iteration %d)" % (
            self.b_conv1.get_shape()[0], self.b_conv2.get_shape()[0], self.session.run(self.iteration))

    def train(self, x, y, keep_probability, learning_rate):
        return self.session.run([self.train_step, self.iteration],
                                feed_dict={self.x: x,
                                           self.y: y,
                                           self.keep_probability: keep_probability,
                                           self.learning_rate: learning_rate})[1]

    def test(self, x, y):
        return self.session.run(self.accuracy, feed_dict={self.x: x, self.y: y, self.keep_probability: 1})

    @property
    def b_conv1(self):
        return self._tensor("convolution_1/bias:0")

    @property
    def b_conv2(self):
        return self._tensor("convolution_2/bias:0")

    @property
    def train_step(self):
        return self._tensor("train/train_step:0")

    @property
    def accuracy(self):
        return self._tensor("train/accuracy:0")

    @property
    def iteration(self):
        return self._tensor("iteration:0")

    @property
    def x(self):
        return self._tensor("parameters/x:0")

    @property
    def y(self):
        return self._tensor("parameters/y:0")

    @property
    def keep_probability(self):
        return self._tensor("parameters/keep_probability:0")

    @property
    def learning_rate(self):
        return self._tensor("parameters/learning_rate:0")

    def _tensor(self, name):
        return self.session.graph.get_tensor_by_name(name)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="MNIST model")
    parser.set_defaults(func=lambda _: parser.print_usage())

    shared_arguments = argparse.ArgumentParser(add_help=False)
    shared_arguments.add_argument("--mnist", default="mnist.data", help="MNIST files directory, default mnist.data")
    shared_arguments.add_argument("--model_directory", metavar="model-directory", default="checkpoint",
                                  help="Model directory, default checkpoint")

    train_parser = subparsers.add_parser("train", parents=[shared_arguments],
                                         description="Train a digit classifier model", help="train model")
    train_parser.add_argument("--layer-1", type=int, default=32, help="convolution layer 1 size, default 32")
    train_parser.add_argument("--layer-2", type=int, default=64, help="convolution layer 2 size, default 64")
    train_parser.add_argument("--learning-rate", type=float, default=1e-4, help="learning rate, default 1e-4")
    train_parser.add_argument("--keep-probability", type=float, default=0.5,
                              help="fully-connected layer keep probability, default 0.5")
    train_parser.add_argument("--training-examples", type=int, help="number of training examples, default one epoch")
    train_parser.add_argument("--batch-size", type=int, default=50, help="batch size, default 50")
    train_parser.add_argument("--report-interval", type=int, default=50,
                              help="how often to report training batch accuracy, default every 50 iterations")
    train_parser.add_argument("--validation-interval", type=int, default=100,
                              help="how often to run the validation set, default every 100 iterations")
    train_parser.add_argument("--checkpoint-interval", type=int, default=500,
                              help="how often to save a checkpoint model, default every 500 iterations")
    train_parser.set_defaults(func=train)

    test_parser = subparsers.add_parser("test", parents=[shared_arguments], description="Test a digit classifier model",
                                        help="test model")
    test_parser.set_defaults(func=test)

    args = parser.parse_args()
    args.func(args)


def train(args):
    training_data = input_data.read_data_sets(args.mnist, one_hot=True).train
    validation_data = input_data.read_data_sets(args.mnist, one_hot=True).validation
    if os.path.exists(args.model_directory):
        model = MNISTModelSession.restore(args.model_directory)
    else:
        os.makedirs(args.model_directory)
        model = MNISTModelSession.create(layer_1=args.layer_1, layer_2=args.layer_2)
    print(model)
    if args.training_examples is None:
        args.training_examples = training_data.num_examples
    for _ in range(args.training_examples // args.batch_size):
        x, y = training_data.next_batch(args.batch_size)
        iteration = model.train(x, y, args.keep_probability, args.learning_rate)
        if iteration % args.report_interval == 0:
            training_batch_accuracy = model.test(x, y)
            print("%s: training batch accuracy %0.4f" % (model, training_batch_accuracy))
        if iteration % args.validation_interval == 0:
            validation_accuracy = model.test(validation_data.images, validation_data.labels)
            print("%s: validation accuracy %0.4f" % (model, validation_accuracy))
        if iteration % args.checkpoint_interval == 0:
            model.save(args.model_directory)
    model.save(args.model_directory)
    print("Final model %s" % model)


def test(args):
    test_data = input_data.read_data_sets(args.mnist, one_hot=True).test
    model = MNISTModelSession.restore(args.model_directory)
    print(model)
    accuracy = model.test(test_data.images, test_data.labels)
    print("Test accuracy %0.4f" % accuracy)


if __name__ == "__main__":
    main()
