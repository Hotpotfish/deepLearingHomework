from tensorflow.python.keras.engine.network import Network
import tensorflow as tf


class Lenet(Network):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self._build_graph()

    def _build_graph(self, network_name='Lenet'):
        self._setup_placeholders_graph()
        self._build_network_graph(network_name)
        self._compute_loss_graph()
        self._compute_acc_graph()
        self._create_train_op_graph()
        self.merged_summary = tf.summary.merge_all()

    def _setup_placeholders_graph(self):
        self.x = tf.placeholder("float", shape=[None, 32, 32, 1], name='x')
        self.y_ = tf.placeholder("float", shape=[None, 10], name='y_')
        self.keep_prob = tf.placeholder("float", name='keep_prob')

    def _cnn_layer(self, scope_name, W_name, b_name, x, filter_shape, conv_strides, padding_tag='VALID'):
        with tf.variable_scope(scope_name):
            conv_W = tf.get_variable(W_name,
                                     dtype=tf.float32,
                                     initializer=tf.truncated_normal(shape=filter_shape, mean=self.mu,
                                                                     stddev=self.sigma))
            conv_b = tf.get_variable(b_name,
                                     dtype=tf.float32,
                                     initializer=tf.zeros(filter_shape[3]))
            conv = tf.nn.conv2d(x, conv_W,
                                strides=conv_strides,
                                padding=padding_tag) + conv_b
            tf.summary.histogram('weights', conv_W)
            tf.summary.histogram('biases', conv_b)
            tf.summary.histogram('activations', conv)
            return conv

    def _pooling_layer(self, scope_name, x, pool_ksize, pool_strides, padding_tag='VALID'):
        with tf.variable_scope(scope_name):
            pool = tf.nn.avg_pool(x, pool_ksize, pool_strides, padding=padding_tag)
            tf.summary.histogram('Pooling', pool)
            return pool

    def _fully_connected_layer(self, scope_name, W_name, b_name, x, W_shape):
        with tf.variable_scope(scope_name):
            x = tf.reshape(x, [-1, 14 * 14 * 6])
            w = tf.get_variable(W_name,
                                dtype=tf.float32,
                                initializer=tf.truncated_normal(shape=W_shape, mean=self.mu,
                                                                stddev=self.sigma))
            b = tf.get_variable(b_name,
                                dtype=tf.float32,
                                initializer=tf.zeros(W_shape[1]))

            r = tf.add(tf.matmul(x, w), b)
            tf.summary.histogram('full_connected', r)
        return r

    def _build_network_graph(self, scope_name):
        with tf.variable_scope(scope_name):
            conv1 = self._cnn_layer('layer_1_conv', 'conv1_w', 'conv1_b', self.x, (5, 5, 1, 6), [1, 1, 1, 1])
            self.conv1 = tf.nn.relu(conv1)
            self.pool1 = self._pooling_layer('layer_1_pooling', self.conv1, [1, 2, 2, 1], [1, 2, 2, 1])
            self.logits = self._fully_connected_layer('full_connected', 'full_connected_w', 'full_connected_b',
                                                      self.pool1, (14 * 14 * 6, 10))

            self.y_predicted = tf.nn.softmax(self.logits)
            tf.summary.histogram("y_predicted", self.y_predicted)

    def _compute_loss_graph(self):
        with tf.name_scope("loss_function"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_predicted)
            self.loss = tf.reduce_mean(loss)
            tf.summary.scalar("cross_entropy", self.loss)

    def _compute_acc_graph(self):
        with tf.name_scope("acc_function"):
            self.accuracy = tf.metrics.accuracy(labels=self.y_, predictions=self.y_predicted)[1]
            tf.summary.scalar("cross_entropy", self.accuracy)

    def _create_train_op_graph(self):
        self.train_op = tf.train.AdamOptimizer(0.1).minimize(self.loss)
        # tf.summary.histogram("train_op", self.train_op)
