# 基础作业：
# 假设有函数y = cos(ax + b), 其中a为学号前两位，b为学号最后两位。首先从此函数中以相同步长（点与点之间在x轴上距离相同），在0<(ax+b)<2pi范围内，采样出2000个点，然后利用采样的2000个点作为特征点进行三次函数拟合(三次函数形式为 y = w1 * x + w2 * x^2 + w3 * x^3 + b, 其中wi为可训练的权值，b为可训练的偏置值，x和y为输入的训练数据 ) 。要求使用TensorFlow实现三次函数拟合的全部流程。拟合完成后，分别使用ckpt模式和PB模式保存拟合的模型。然后，针对两种模型存储方式分别编写恢复模型的程序。两个模型恢复程序分别使用ckpt模式和PB模式恢复模型，并将恢复的模型参数（wi和b）打印在屏幕上，同时绘制图像（图像包括所有的2000个采样点及拟合的函数曲线）。
#
# 请提交文档，内容包括模型参数配置、程序运行截图、拟合的三次函数以及绘制的图像，同时提交三个python脚本文件：1. 函数拟合及两种模式的模型保存程序2. ckpt模型恢复程序，3. PB模型恢复程序。
#
#
# 提交内容：
# 源码、运行截图和报告，报告中应包括有关的参数，如x的初始化方式、学习率、迭代次数等，以及每次迭代后的x和y的值。
#
# 注意：基本作业为所有人必做。进阶作业选作，不计入总分。
#
#
# 作业格式规范：
#
# 提交源码、文档。文档包括：运行截图和保存的结果图片，文档中也标明学号和姓名。将所有文件打包后命名为："姓名拼音_学号_i",i是第i次作业。例如：张三，学号66666，提交的第二次作业命名为：zhangsan_66666_2。
import math

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import graph_util


def cubicFunctionFitting():
    # 构建数据集
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

    # 计算损失
    loss = tf.losses.mean_squared_error(tf_y, y_pre)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss)

    # 创建保存器
    saver = tf.train.Saver()

    # 会话处理相关
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 开启训练过程的连续显示
    plt.ion()
    for step in range(1000):
        # 训练输出每个位置对应输出结果
        _, l, Y_pred, W1, W2, W3, B = sess.run(
            [train_op, loss, y_pre, w1, w2, w3, b],
            {tf_x: x, tf_y: y})
        if step % 5 == 0:
            # 展示学习进程
            plt.cla()
            l1, = plt.plot(x, y, 'g-')
            l2, = plt.plot(x, Y_pred, 'r--')
            plt.legend(handles=[l1, l2], labels=['Original', 'Fitting'], loc='best')
            plt.pause(0.1)

            # 打印
            print(W1, "* x + ",
                  W2, "* x ^ 2 + ",
                  W3, "* x ^ 3 + ",
                  B)
            print('loss：', l)
    # 将session保存为ckpt格式
    saver.save(sess, "Resource/hw_2/ckpt/hw_2.ckpt")

    # 将session的变量存储为pb文件
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['y_pre'])
    with tf.gfile.FastGFile('Resource/hw_2/pb/hw_2.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())

    # 关闭训练过程的连续显示
    plt.ioff()
    plt.show()
    sess.close()


def main():
    cubicFunctionFitting()
    pass


if __name__ == "__main__":
    main()
