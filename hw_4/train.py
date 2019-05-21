import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from hw_4.lenet import Lenet

batch_size = 128

mninst = input_data.read_data_sets('../Resource/hw_3/MNIST_data/', one_hot=True)
train_x, train_y = mninst.train.next_batch(batch_size)
test_x, test_y = mninst.train.next_batch(batch_size)

train_x = train_x.reshape(batch_size, 28, 28, 1)
test_x = train_x.reshape(batch_size, 28, 28, 1)

# Pad images with 0s
train_x = np.pad(train_x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
test_x = np.pad(test_x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
print("Updated Image Shape: {}".format(train_x[0].shape))

lenet_part = Lenet(0, 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer(),
             sess.run(tf.local_variables_initializer())
             )

    for i in range(2000):
        _, loss, acc = sess.run([lenet_part.train_op,
                                 lenet_part.loss, lenet_part.accuracy],
                                feed_dict={lenet_part.x: train_x,
                                           lenet_part.y_: train_y})
        if i % 200 == 0:
            print('train_step:%d' % i, 'loss:', loss, ' ', "acc:", acc)

    loss, acc = sess.run([lenet_part.loss, lenet_part.accuracy],
                         feed_dict={lenet_part.x: test_x,
                                    lenet_part.y_: test_y})

    print("test:", 'loss:', loss, ' ', "acc:", acc)
