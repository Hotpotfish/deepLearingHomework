from tensorflow.python.keras.engine.network import Network
import tensorflow as tf
import tensorflow.contrib.slim as slim


class Lenet(Network):

    def __init__(self):
        self._build_graph()

    def _build_graph(self):
        self._setup_placeholders_graph()
        self._build_network_graph()
        self._compute_loss_graph()
        self._compute_acc_graph()
        self._create_train_op_graph()

    def _setup_placeholders_graph(self):
        self.x = tf.placeholder("float", shape=[None, 28, 28, 1], name='x')
        self.y_ = tf.placeholder("float", shape=[None, 10], name='y_')

    def _build_network_graph(self):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.2),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            padding='SAME'
                            ):
            net = slim.repeat(self.x, 2, slim.conv2d, 6, [3, 3], scope='conv1')
            # now 28 * 28 * 6
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            # now 14 * 14 * 6
            net = slim.repeat(net, 2, slim.conv2d, 12, [3, 3], scope='conv2')
            # now 14 * 14 * 12
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            # now 7 * 7 * 12

        net = tf.contrib.layers.flatten(net)

        net = slim.fully_connected(net, 1024, scope='fc1')

        # now 7 * 7 * 16
        net = slim.fully_connected(net, 10, scope='fc2')
        # now 10

        # flatten

        net = tf.reshape(net, [-1, 10])

        self.y_predicted = tf.nn.softmax(net)

    def _compute_loss_graph(self):
        with tf.name_scope("loss_function"):
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_predicted), 1))

    def _compute_acc_graph(self):
        with tf.name_scope("acc_function"):
            self.accuracy = \
                tf.metrics.accuracy(labels=tf.argmax(self.y_, 1), predictions=tf.argmax(self.y_predicted, 1))[1]

    def _create_train_op_graph(self):
        self.train_op = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
