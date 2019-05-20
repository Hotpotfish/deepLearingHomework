import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt



# 重新对点进行采样
x = np.linspace(-7 / 18, (2 * math.pi - 7) / 18, 2000)[:, np.newaxis]
y = np.cos(18 * x + 7)

# 初始化四个变量对应题目中要求解的。
w1 = tf.Variable(tf.random_normal([1, 1]), name='w1')
w2 = tf.Variable(tf.random_normal([1, 1]), name='w2')
w3 = tf.Variable(tf.random_normal([1, 1]), name='w3')
b = tf.Variable(tf.zeros([1, 1]), name='b')

# 数据占位符
tf_x = tf.placeholder(tf.float32, x.shape)  # input x
tf_y = tf.placeholder(tf.float32, y.shape)  # input y

# 表示因式
t1 = tf.matmul(tf.pow(tf_x, 0), b)  # b
t2 = tf.matmul(tf.pow(tf_x, 1), w1)  # w1 * x
t3 = tf.matmul(tf.pow(tf_x, 2), w2)  # w2 * x^2
t4 = tf.matmul(tf.pow(tf_x, 3), w3)  # w3 * x^3

y_pre = tf.add_n([t1, t2, t3, t4], name='y_pre')
saver = tf.train.Saver()

with tf.Session() as sess:

    saver.restore(sess, 'Resource/hw_2/ckpt/hw_2.ckpt')
    print("w1:", sess.run(w1))
    print("w2:", sess.run(w2))
    print("w3:", sess.run(w3))
    print("b:", sess.run(b))

    # 训练输出每个位置对应输出结果
    Y_pred = sess.run([y_pre], {tf_x: x, tf_y: y})

    l1, = plt.plot(x, y, 'g-')
    l2, = plt.plot(x, Y_pred[0], 'r--')
    plt.legend(handles=[l1, l2], labels=['Original', 'Fitting'], loc='best')
    plt.show()
