import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np

lr = 0.01
iter = 20000
batch_size = 64 * 2
Q = 5
threshold = 2


# 第一层隐藏层
def hiddenLayer1(X):
    with tf.variable_scope("hiddenLayer1", reuse=tf.AUTO_REUSE):
        w1 = tf.get_variable('w1',
                             shape=[28 * 28, 500],
                             initializer=tf.random_normal_initializer)
    return tf.nn.relu(tf.matmul(X, w1))


# 第二层隐藏层
def hiddenLayer2(X):
    with tf.variable_scope("hiddenLayer2", reuse=tf.AUTO_REUSE):
        w2 = tf.get_variable('w2',
                             shape=[500, 10],
                             initializer=tf.random_normal_initializer)
    return tf.nn.relu(tf.matmul(X, w2))


def getFeature(X):
    t1 = hiddenLayer1(X)
    t2 = hiddenLayer2(t1)
    return t2


def EW(X1, X2):
    ew = tf.sqrt(tf.reduce_sum(tf.square(X1 - X2), 1) + 1e-6)[:, np.newaxis]
    return ew


# Y的预测值
def Y_Pred(X1, X2):
    ew = EW(X1, X2)
    threshold_t = tf.constant(threshold, shape=[batch_size / 2, 1], dtype=tf.float32)
    y_pred = tf.cast(tf.less(threshold_t, ew), dtype=tf.float32)

    return y_pred


def Loss(X1, X2, Y):
    ew = EW(X1, X2)
    t1 = (1 - Y) * (2 / Q) * ew * ew
    t2 = Y * 2 * Q * tf.exp((-2.77) / Q * ew)
    r = tf.add(t1, t2)

    return r


def main():
    # 占位符
    tf_x1 = tf.placeholder(tf.float32, [None, 28 * 28])  # input x1
    tf_y1 = tf.placeholder(tf.float32, [None, 1])  # input y1
    tf_x2 = tf.placeholder(tf.float32, [None, 28 * 28])  # input x2
    tf_y2 = tf.placeholder(tf.float32, [None, 1])  # input y2

    # 计算出标签值
    y = tf.cast(tf.not_equal(tf_y1, tf_y2), dtype=tf.float32)

    # 网络抽取特征
    x1 = getFeature(tf_x1)
    x2 = getFeature(tf_x2)

    ew = EW(x1, x2)
    t1 = (1 - y) * (2 / Q) * ew * ew
    t2 = y * 2 * Q * tf.exp((-2.77) / Q * ew)
    loss = tf.add(t1, t2)
    loss = tf.reduce_mean(loss)

    # y_pred = Y_Pred(x1, x2)

    # accuracy = tf.metrics.accuracy(labels=y, predictions=y_pred)[1]
    # loss = tf.reduce_mean(Loss(x1, x2, y), 0)
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer(),
                 sess.run(tf.local_variables_initializer())
                 )
        mninst = input_data.read_data_sets('../Resource/hw_3/MNIST_data/', one_hot=False)

        for itera in range(iter):
            with tf.variable_scope("hiddenLayer2", reuse=tf.AUTO_REUSE):
                w2 = tf.get_variable('w2',
                                     shape=[500, 10],
                                     initializer=tf.random_normal_initializer)

            train_x, train_y = mninst.train.next_batch(batch_size)
            train_x1 = train_x[0:-1:2]
            train_x2 = train_x[1::2]
            train_y1 = train_y[0:-1:2][:, np.newaxis]
            train_y2 = train_y[1::2][:, np.newaxis]
            _, L, Y, E_W, W2 = sess.run([train_op, loss, y, ew, w2], feed_dict={tf_x1: train_x1,
                                                                                tf_x2: train_x2,
                                                                                tf_y1: train_y1,
                                                                                tf_y2: train_y2})
            fpr, tpr, thresholds = roc_curve(y_true=Y, y_score=E_W, pos_label=1)
            plt.plot(fpr, tpr)
            plt.show()
            print(fpr,tpr)

            print(L)

            # if itera % 1000 == 0:
            #     print("step:", itera, "  ", "Loss:", L, "  ", "accuracy:", ACC)


if __name__ == '__main__':
    main()
    pass
