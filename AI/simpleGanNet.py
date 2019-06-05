import tensorflow as tf


class simpleGanNet():

    def __init__(self):
        self._build_graph()

    def buildGraph(self):
        self.setupPlaceholdersGraph()
        self.generator1()
        self.discriminator1_real()
        self.discriminator1_fake()
        self.getLoss1()
        self.trainOp1()
        pass

    def setupPlaceholdersGraph(self):
        self.x1 = tf.placeholder("float", shape=[None, 4160, 2768, 3], name='x')
        self.x2 = tf.placeholder("float", shape=[None, 4160, 2768, 3], name='x')
        self.x3 = tf.placeholder("float", shape=[None, 4160, 2768, 3], name='x')  # 随机噪声
        self.x4 = tf.placeholder("float", shape=[None, 4160, 2768, 3], name='x')
        self.y = tf.placeholder("float", shape=[None, 4160, 2768, 3], name='y')

    def generator1(self):
        # encoder
        # now  4160 * 2768 * 6
        ge = tf.layers.conv2d(self.x3, 6, 5, 1, 'same', activation=tf.nn.relu)

        # 2080 * 1384 * 6
        ge = tf.layers.average_pooling2d(ge, pool_size=2, strides=2)

        # now  2080 * 1384 * 10
        ge = tf.layers.conv2d(ge, 10, 5, 1, 'same', activation=tf.nn.relu)

        # 1040 * 692 * 10
        ge = tf.layers.average_pooling2d(ge, pool_size=2, strides=2)

        # decoder
        # now  2080 * 1384 * 10
        ge = tf.layers.conv2d_transpose(ge, 10, 2, 2, 'same', activation=tf.nn.relu)

        # now 2080 * 1384 * 6
        ge = tf.layers.conv2d(ge, 6, 5, 1, 'same', activation=tf.nn.relu)

        # now 4160 * 2768 * 6
        ge = tf.layers.conv2d_transpose(ge, 6, 2, 2, 'same', activation=tf.nn.relu)

        # now 4160 * 2768 * 3
        ge = tf.layers.conv2d(ge, 3, 5, 1, 'same', activation=tf.nn.relu)

        self.generator1 = ge

    def discriminator1_real(self):
        # now  4160 * 2768 * 1
        di = tf.layers.conv2d(self.x1, 1, 5, 1, 'same', activation=tf.nn.relu)

        # now 260 * 173 * 1
        di = tf.layers.average_pooling2d(di, pool_size=16, strides=2)
        # flatten
        di = tf.reshape(tf.layers.flatten(di), [-1, 260 * 173 * 1])

        di = tf.layers.dense(di, 128, tf.nn.relu)

        di = tf.layers.dense(di, 1, tf.nn.sigmoid)

        self.d1_real_prob = di

    def discriminator1_fake(self):
        # now  4160 * 2768 * 1
        di = tf.layers.conv2d(self.generator1, 1, 5, 1, 'same', activation=tf.nn.relu)

        # now 260 * 173 * 1
        di = tf.layers.average_pooling2d(di, pool_size=16, strides=2)
        # flatten
        di = tf.reshape(tf.layers.flatten(di), [-1, 260 * 173 * 1])

        di = tf.layers.dense(di, 128, tf.nn.relu)

        di = tf.layers.dense(di, 1, tf.nn.sigmoid)

        self.d1_fake_prob = di

    def getLoss1(self):
        self.D_loss = -tf.reduce_mean(tf.log(self.d1_real_prob) + tf.log(1 - self.d1_fake_prob))
        self.G_loss = -tf.reduce_mean(tf.log(self.d1_fake_prob))

    def trainOp1(self):
        self.trainOp1_D = tf.train.AdamOptimizer(1e-4).minimize(self.D_loss)
        self.trainOp1_G = tf.train.AdamOptimizer(1e-4).minimize(self.G_loss)
