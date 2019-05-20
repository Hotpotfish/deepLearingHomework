import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import math
import matplotlib.pyplot as plt

# 重新对点进行采样
x = np.linspace(-7 / 18, (2 * math.pi - 7) / 18, 2000)[:, np.newaxis]
y = np.cos(18 * x + 7)

# 数据占位符
tf_x = tf.placeholder(tf.float32, x.shape)  # input x
tf_y = tf.placeholder(tf.float32, y.shape)  # input y

with tf.Session() as sess:
    # 遵照ppt摘的
    with gfile.FastGFile('Resource/hw_2/pb/hw_2.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    sess.run(tf.global_variables_initializer())
    # 获取变量
    w1 = sess.graph.get_tensor_by_name('w1:0')
    w2 = sess.graph.get_tensor_by_name('w2:0')
    w3 = sess.graph.get_tensor_by_name('w3:0')
    b = sess.graph.get_tensor_by_name('b:0')
    # 打印变量
    print('w1:', sess.run(w1))
    print('w2:', sess.run(w2))
    print('w3:', sess.run(w3))
    print('b:', sess.run(b))

    # 表示因式
    t1 = tf.matmul(tf.pow(tf_x, 0), b)  # b
    t2 = tf.matmul(tf.pow(tf_x, 1), w1)  # w1 * x
    t3 = tf.matmul(tf.pow(tf_x, 2), w2)  # w2 * x^2
    t4 = tf.matmul(tf.pow(tf_x, 3), w3)  # w3 * x^3

    # 计算出y_pre
    y_pre = tf.add_n([t1, t2, t3, t4], name='y_pre')

    Y_pred = sess.run([y_pre], {tf_x: x, tf_y: y})

    l1, = plt.plot(x, y, 'g-')
    l2, = plt.plot(x, Y_pred[0], 'r--')
    plt.legend(handles=[l1, l2], labels=['Original', 'Fitting'], loc='best')
    plt.show()
