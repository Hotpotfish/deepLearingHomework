import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import matplotlib.pyplot as plt


def main():
    # 读取mninst数据集
    mninst = input_data.read_data_sets('../Resource/hw_3/MNIST_data/', one_hot=True)

    test_x = mninst.test.images[:3000]
    test_y = mninst.test.labels[:3000]

    # 占位符
    tf_x = tf.placeholder(tf.float32, test_x.shape)  # input x
    tf_y = tf.placeholder(tf.float32, test_y.shape)  # input y

    y_pred = Y_Pred(tf_x)
    # 经典交叉熵公式（第一版弃用，改用第二版）
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_y, logits=y_pred)
    # 所有交叉熵求平均
    loss = tf.reduce_mean(cross_entropy)
    # 理解准确度定义
    accuracy = tf.metrics.accuracy(
        labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(y_pred, axis=1)
        , )[1]
    train_op = tf.train.AdamOptimizer(0.05).minimize(loss)

    with tf.Session() as sess:
        # 共享变量调用
        with tf.variable_scope("X_W", reuse=tf.AUTO_REUSE):
            w = tf.get_variable(name='w', shape=[14 * 28, 10])
        with tf.variable_scope("Y_Pred", reuse=tf.AUTO_REUSE):
            b = tf.get_variable(name='b', shape=[10])

        # 初始化全局变量和局部变量（局部不初始化的话accuracy会报错）
        sess.run(tf.global_variables_initializer(),
                 sess.run(tf.local_variables_initializer())
                 )
        for i in range(2000):
            W, B, C_E, LOSS, ACC, _ = sess.run([w, b, cross_entropy, loss, accuracy, train_op],
                                               {tf_x: test_x, tf_y: test_y})
            # print('w:', W)
            # print('b:', B)

            if i % 200 == 0:
                print('Step %d,' % i, 'Training Accuracy %f' % ACC)

        print('[accuracy,loss]:', [ACC, LOSS])

        W = W.reshape([-1, 14, 28])
        # 显示图片
        plt.figure()
        for i in range(10):
            plt.subplot(1, 10, i + 1)
            plt.imshow(W[i])

        plt.show()


def Y_Pred(X):
    with tf.variable_scope("Y_Pred", reuse=tf.AUTO_REUSE):
        t1 = X_W(X[:, 0:14 * 28])
        t2 = X_W(X[:, 14 * 28: 28 * 28])
        b = tf.get_variable('b', shape=[10], initializer=tf.zeros_initializer)
        r = tf.add(t1, t2) + b

    return tf.nn.softmax(r)


def X_W(X):
    with tf.variable_scope("X_W", reuse=tf.AUTO_REUSE):
        # 按照题目要求设定共享变量
        w = tf.get_variable('w',
                            shape=[14 * 28, 10],
                            initializer=tf.random_normal_initializer)

        return tf.matmul(X, w)


if __name__ == '__main__':
    main()
    pass
