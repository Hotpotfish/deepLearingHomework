# 题目：
#
# 设计python程序，首先安装并导入opencv库：
# 例如：conda install opencv
# import cv2
#
# 然后使用cv2.imread()读取任意彩色图片为numpy矩阵，然后进行以下操作：
# (1) 将图片的三个通道顺序进行改变，由RGB变为BRG，并用imshow()或者matplotlib中的有关函数显示图片
# (2) 利用Numpy给改变通道顺序的图片中指定位置打上红框，其中红框左上角和右下角坐标定义方式为：假设学号为12069028，则左上角坐标为(12, 06), 右下角坐标为(12+90, 06+28).  (不可使用opencv中自带的画框工具）
# (3) 利用cv2.imwrite()函数保存加上红框的图片。
#
#
# 进阶题目：
# 假设有函数y = cos(ax + b), 其中a为学号前两位，b为学号最后两位。首先从此函数中以相同步长（点与点之间在x轴上距离相同），在0<(ax+b)<2pi范围内，采样出2000个点，然后利用采样的2000个点作为特征点进行三次函数拟合。请提交拟合的三次函数以及对应的图样（包括采样点及函数曲线）。
#
#
# 注意：基本作业为所有人必做。进阶作业选作，不计入总分。
#
#
#
# 提交内容：
#
# 源码、文档。文档包括：运行截图和保存的结果图片，文档中也标明学号和姓名。打包后命名为："姓名拼音_学号_i",i是第i次作业。例如：张三，学号66666，提交的第一次作业命名为：zhangsan_66666_1。
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def changeBgrIntoBrg(imagePath):
    image = cv2.imread(imagePath)[:, :, (0, 2, 1)]

    cv2.imshow("BRG_image", image)

    cv2.waitKey(0)

    return image


def giveRedboxToImage(image, tl, br):
    image[tl[0]:br[0] + 1, tl[1]] = [0, 0, 255]  # 北边红线

    image[tl[0], tl[1]:br[1] + 1] = [0, 0, 255]  # 西边红线

    image[br[0] + 1, tl[1]:br[1] + 1] = [0, 0, 255]  # 东边红线

    image[tl[0]:br[0] + 1, br[1] + 1] = [0, 0, 255]  # 南边红线

    cv2.imshow("BRG_image", image)

    cv2.waitKey(0)

    cv2.imwrite(r"images\hw_1\image1_modifiy.jpg", image)

    return image


def cosinFunctionFitting():
    x = np.linspace(-7 / 18, (2 * math.pi - 7) / 18, 2000)[:, np.newaxis]

    y = np.cos(18 * x + 7)

    plt.scatter(x, y)

    plt.show()

    tf_x = tf.placeholder(tf.float32, x.shape)  # input x
    tf_y = tf.placeholder(tf.float32, y.shape)  # input y

    # neural network layers
    l1 = tf.layers.dense(tf_x, 10, tf.nn.sigmoid)  # hidden layer
    output = tf.layers.dense(l1, 1)  # output layer

    loss = tf.losses.mean_squared_error(tf_y, output)  # compute cost
    optimizer = tf.train.AdamOptimizer(learning_rate=0.3)
    train_op = optimizer.minimize(loss)

    sess = tf.Session()  # control training and others
    sess.run(tf.global_variables_initializer())  # initialize var in graph

    plt.ion()  # something about plotting

    for step in range(1000):
        # train and net output
        _, l, pred = sess.run([train_op, loss, output], {tf_x: x, tf_y: y})
        if step % 5 == 0:
            # plot and show learning process
            plt.cla()
            plt.scatter(x, y)
            plt.plot(x, pred, 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)

    plt.ioff()
    plt.show()




def main():
    # image = changeBgrIntoBrg(r"images\hw_1\image1.jpg")
    #
    # image = giveRedboxToImage(image, (12, 6), (12 + 90, 6 + 28))

    cosinFunctionFitting()

    pass


if __name__ == "__main__":
    main()
