# 进阶作业：
# 使用TensorFlow定义函数y = 1 - sin(x)/x, 并求解y达到最小值时对应的x
# 步骤提示：
# (1) 使用TensorFlow给出函数定义
# (2) 利用tf.train.GradientDescentOptimizer定义优化器
# (3) 启动会话，用梯度下降算法训练一定的迭代次数，在每次迭代之后输出当前的x和y值
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def getMinYAndCurrentX():
    # 初始化x变量
    x = tf.Variable(1.0, name='x')
    # 列出y的表达式
    y = tf.subtract(1.0, tf.divide(tf.math.sin(x), x))
    # 此时y即是优化器要优化的目标，之所以这样写是为了强调这一点
    loss = y
    # 调用优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 迭代求最小值
        for step in range(1000):
            X, Y, Loss, _ = sess.run([x, y, loss, train_op])
            print('x:', X, 'y:', Y, 'loss:', Loss)


def main():
    getMinYAndCurrentX()
    pass


if __name__ == "__main__":
    main()
