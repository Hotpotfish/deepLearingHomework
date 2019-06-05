import tensorflow as tf


class simpleNet():

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self._build_graph()

    def _build_graph(self):
        self._setup_placeholders_graph()
        self._build_generator1_graph()
        # self._get_real_and_fake_prob1_graph()
        self._compute_loss_graph1()

        self._create_train_op_graph()

    def _setup_placeholders_graph(self):
        self.x1 = tf.placeholder("float", shape=[None, 4160, 2768, 3], name='x1')
        self.x2 = tf.placeholder("float", shape=[None, 4160, 2768, 3], name='x2')
        # self.x3 = tf.placeholder("float", shape=[None, 4160, 2768, 3], name='x3')
        # self.x4 = tf.placeholder("float", shape=[None, 4160, 2768, 3], name='x4')
        # self.y = tf.placeholder("float", shape=[None, 4160, 2768, 3], name='y')

    def _cnn_layer(self, scope_name, W_name, b_name, x, filter_shape, conv_strides, padding_tag='SAME'):
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

            return conv

    def _transpose_layer(self, scope_name, W_name, b_name, x, filter_shape, output_shape, conv_strides,
                         padding_tag='SAME'):
        with tf.variable_scope(scope_name):
            tran_W = tf.get_variable(W_name,
                                     dtype=tf.float32,
                                     initializer=tf.truncated_normal(shape=filter_shape, mean=self.mu,
                                                                     stddev=self.sigma))
            tran_b = tf.get_variable(b_name,
                                     dtype=tf.float32,
                                     initializer=tf.zeros(filter_shape[3]))
            tran = tf.nn.conv2d_transpose(x, tran_W, output_shape,
                                          strides=conv_strides,
                                          padding=padding_tag) + tran_b

            return tran

    def _pooling_layer(self, scope_name, x, pool_ksize, pool_strides, padding_tag='SAME'):
        with tf.variable_scope(scope_name):
            pool = tf.nn.max_pool(x, pool_ksize, pool_strides, padding=padding_tag)
            tf.summary.histogram('Pooling', pool)
            return pool

    def _fully_connected_layer(self, scope_name, W_name, b_name, x, W_shape):
        with tf.variable_scope(scope_name):
            w = tf.get_variable(W_name,
                                dtype=tf.float32,
                                initializer=tf.truncated_normal(shape=W_shape, mean=self.mu,
                                                                stddev=self.sigma))
            b = tf.get_variable(b_name,
                                dtype=tf.float32,
                                initializer=tf.zeros(W_shape[1]))

            r = tf.add(tf.matmul(x, w), b)
            r = tf.layers.batch_normalization(r, training=True)

        return r

    # def _build_generator1_graph(self):
    #     with tf.variable_scope("generator1"):
    #         # encoder
    #         # 2080 * 1384 * 6
    #         en1_conv = self._cnn_layer('en1_conv', 'en1_conv_w', 'en1_conv_b', self.x3, (5, 5, 3, 6), [1, 2, 2, 1])
    #
    #         # 1040 * 692 * 6
    #         en1_pool = self._pooling_layer('en1_pooling', en1_conv, [1, 2, 2, 1], [1, 2, 2, 1])
    #
    #         # 520 * 346 * 10
    #         en2_conv = self._cnn_layer('en2_conv', 'en2_conv_w', 'en2_conv_b', en1_pool, (5, 5, 6, 10), [1, 2, 2, 1])
    #
    #         # 260 * 173 * 10
    #         en2_pool = self._pooling_layer('en2_pooling', en2_conv, [1, 2, 2, 1], [1, 2, 2, 1])
    #
    #         # decoder
    #         # 520 * 346 * 10
    #         de1_tran = self._transpose_layer('de1_tran', 'de1_tran_w', 'de1_tran_b', en2_pool, [3, 3, 10, 10],
    #                                          [1, 520, 346, 10], [1, 2, 2, 1])
    #
    #         # 520 * 346 * 6
    #         de1_conv = self._cnn_layer('de1_conv', 'de1_conv_w', 'de1_conv_b', de1_tran, (5, 5, 10, 6), [1, 1, 1, 1])
    #
    #         # 1040 * 692 * 6
    #         de2_tran = self._transpose_layer('de2_tran', 'de2_tran_w', 'de2_tran_b', de1_conv, [3, 3, 6, 6],
    #                                          [1, 1040, 692, 6], [1, 2, 2, 1])
    #
    #         # 1040 * 692 * 3
    #         de2_conv = self._cnn_layer('de2_conv', 'de2_conv_w', 'de2_conv_b', de2_tran, (5, 5, 6, 3), [1, 1, 1, 1])
    #
    #         # 2080 * 1384 * 3
    #         de3_tran = self._transpose_layer('de3_tran', 'de3_tran_w', 'de3_tran_b', de2_conv, [3, 3, 3, 3],
    #                                          [1, 2080, 1384, 3], [1, 2, 2, 1])
    #
    #         # 4160 * 2768 * 3
    #         de4_tran = self._transpose_layer('de4_tran', 'de4_tran_w', 'de4_tran_b', de3_tran, [3, 3, 3, 3],
    #                                          [1, 4160, 2768, 3], [1, 2, 2, 1])
    #
    #         self.generator1 = de4_tran
    #
    #         return de4_tran
    def _build_generator1_graph(self):
        ### Encoder
        conv1 = tf.layers.conv2d(inputs=self.x2, filters=10, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        # Now 28x28x32
        maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')
        # Now 14x14x32
        conv2 = tf.layers.conv2d(inputs=maxpool1, filters=10, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        # Now 14x14x32
        maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')
        # Now 7x7x32
        conv3 = tf.layers.conv2d(inputs=maxpool2, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        # Now 7x7x16
        encoded = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same')
        # Now 4x4x16
        ### Decoder
        upsample1 = tf.image.resize_images(encoded, size=(520, 346), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Now 7x7x16
        conv4 = tf.layers.conv2d(inputs=upsample1, filters=8, kernel_size=(3, 3), padding='same',
                                 activation=tf.nn.relu)
        # Now 7x7x16
        upsample2 = tf.image.resize_images(conv4, size=(1040, 692), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Now 14x14x16
        conv5 = tf.layers.conv2d(inputs=upsample2, filters=10, kernel_size=(3, 3), padding='same',
                                 activation=tf.nn.relu)
        # Now 14x14x32
        upsample3 = tf.image.resize_images(conv5, size=(2080, 1384), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Now 28x28x32
        conv6 = tf.layers.conv2d(inputs=upsample3, filters=10, kernel_size=(3, 3), padding='same',
                                 activation=tf.nn.relu)

        upsample4 = tf.image.resize_images(conv6, size=(4160, 2768), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv7 = tf.layers.conv2d(inputs=upsample4, filters=10, kernel_size=(3, 3), padding='same',
                                 activation=tf.nn.relu)
        # Now 28x28x32
        logits = tf.layers.conv2d(inputs=conv7, filters=3, kernel_size=(3, 3), padding='same', activation=None)

        self.generator1 = logits

        # with tf.variable_scope('generator1'):
        #     # encoder
        #     # 4160 * 2768 * 6
        #     en1_conv = self._cnn_layer('en1_conv', 'en1_conv_w', 'en1_conv_b', self.x3, (3, 3, 3, 6),
        #                                [1, 1, 1, 1])
        #
        #     # 2080 * 1384 * 6
        #     en1_pool = self._pooling_layer('en1_pooling', en1_conv, [1, 3, 3, 1], [1, 2, 2, 1])
        #
        #     # 2080 * 1384 * 10
        #     en2_conv = self._cnn_layer('en2_conv', 'en2_conv_w', 'en2_conv_b', en1_pool, (3, 3, 6, 10), [1, 1, 1, 1])
        #
        #     # 1040 * 692 * 10
        #     en2_pool = self._pooling_layer('en2_pooling', en2_conv, [1, 3, 3, 1], [1, 2, 2, 1])
        #
        #     # decoder
        #     # 2080 * 1384 * 10
        #     de1_tran = self._transpose_layer('de1_tran', 'de1_tran_w', 'de1_tran_b', en2_pool, [3, 3, 10, 10],
        #                                      [1, 2080, 1384, 10], [1, 2, 2, 1])
        #     # 2080 * 1384 * 6
        #     de1_conv = self._cnn_layer('de1_conv', 'de1_conv_w', 'de1_conv_b', de1_tran, (3, 3, 10, 6), [1, 1, 1, 1])
        #
        #     # 4160 * 2768 * 6
        #     de2_tran = self._transpose_layer('de2_tran', 'de2_tran_w', 'de2_tran_b', de1_conv, [3, 3, 6, 6],
        #                                      [1, 4160, 2768, 6], [1, 2, 2, 1])
        #     # 4160 * 2768 * 3
        #     de2_conv = self._cnn_layer('de2_conv', 'de2_conv_w', 'de2_conv_b', de2_tran, (3, 3, 6, 3), [1, 1, 1, 1])
        #
        #     self.generator1 = de2_conv
        #
        #     return de2_conv

    def _build_discriminator1_graph(self, x):
        with tf.variable_scope("discriminator1", reuse=tf.AUTO_REUSE):
            # now  4160 * 2768 * 1
            d1_conv1 = self._cnn_layer('d1_conv1', 'd1_conv_w', 'd1_conv_b', x, (5, 5, 3, 1), [1, 1, 1, 1])

            # now 2080 * 1384 * 1
            d1_conv2 = self._cnn_layer('d1_conv2', 'd1_conv_w', 'd1_conv_b', d1_conv1, (5, 5, 1, 1), [1, 16, 16, 1])

            d1_flatten = tf.reshape(tf.layers.flatten(d1_conv2), [-1, 260 * 173 * 1])

            d1_fc_1 = self._fully_connected_layer('d1_fc_1', 'd1_fc_1_w', 'd1_fc_1_b', d1_flatten, [260 * 173, 1024])

            d1_fc_2 = self._fully_connected_layer('d1_fc_2', 'd1_fc_2_w', 'd1_fc_2_b', d1_fc_1, [1024, 1])

            d1_fc_2_prob = tf.nn.sigmoid(d1_fc_2)

        return d1_fc_2_prob

    def _get_real_and_fake_prob1_graph(self):
        self.d1_fake_prob = self._build_discriminator1_graph(self.generator1)
        self.d1_real_prob = self._build_discriminator1_graph(self.x1)

    # def _compute_loss_graph1(self):
    #     self.D1_loss = -tf.reduce_mean(tf.log(self.d1_real_prob) + tf.log(1 - self.d1_fake_prob))
    #     self.G1_loss = -tf.reduce_mean(tf.log(self.d1_fake_prob))
    #
    #
    #
    # def _create_train_op_graph(self):
    #     self.trainOp1_D = tf.train.AdamOptimizer(1e-4).minimize(self.D1_loss)
    #     self.trainOp1_G = tf.train.AdamOptimizer(1e-4).minimize(self.G1_loss)

    def _compute_loss_graph1(self):
        self.loss1 = tf.reduce_mean(tf.square(self.x1 - self.generator1))

    def _create_train_op_graph(self):
        self.trainOp1 = tf.train.AdamOptimizer(0.001).minimize(self.loss1)
