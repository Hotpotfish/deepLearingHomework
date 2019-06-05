import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from hw_5.Lenet import Lenet

batch_size = 64

mninst = input_data.read_data_sets('../Resource/hw_3/MNIST_data/', one_hot=True)

lenet_part = Lenet()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer(),
             sess.run(tf.local_variables_initializer())
             )

    for i in range(2000):
        train_x, train_y = mninst.train.next_batch(batch_size)
        train_x = train_x.reshape(batch_size, 28, 28, 1)
        _, loss, acc, pre = sess.run([lenet_part.train_op,
                                      lenet_part.loss,
                                      lenet_part.accuracy,
                                      lenet_part.y_predicted],
                                     feed_dict={lenet_part.x: train_x,
                                                lenet_part.y_: train_y})



        if i % 200 == 0:
            print('train_step:%d' % i, 'loss:', loss, ' ', "acc:", acc)
    test_x, test_y = mninst.train.next_batch(batch_size)
    test_x = test_x.reshape(batch_size, 28, 28, 1)
    loss, acc = sess.run([lenet_part.loss, lenet_part.accuracy],
                         feed_dict={lenet_part.x: test_x,
                                    lenet_part.y_: test_y})

    print("test:", 'loss:', loss, ' ', "acc:", acc)
