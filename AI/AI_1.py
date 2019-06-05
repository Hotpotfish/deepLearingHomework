import tensorflow as tf
import numpy as np
import cv2
from AI.simpleGanNet import simpleGanNet as sg
from AI.simpleNet import simpleNet
tain_eps= 10000

dictPath = '../Resource/AI/'


def readImage(filePath):
    image = cv2.imread(filePath)
    return image


def getData(n):
    input1 = readImage(dictPath + str(n) + '/input1.JPG')[np.newaxis, :, :, :]
    input2 = readImage(dictPath + str(n) + '/input2.JPG')[np.newaxis, :, :, :]
    input3 = readImage(dictPath + str(n) + '/input3.JPG')[np.newaxis, :, :, :]
    input4 = readImage(dictPath + str(n) + '/input4.JPG')[np.newaxis, :, :, :]
    output = readImage(dictPath + str(n) + '/output.JPG')[np.newaxis, :, :, :]
    return input1, input2, input3, input4, output


def main():
    for j in range(4, 5):
        input1, input2, input3, input4, output = getData(j)
        simplenet = simpleNet(0, 1)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer(),
                     sess.run(tf.local_variables_initializer())
                     )
            for i in range(tain_eps):
                loss1, _1, g1 = sess.run([simplenet.loss1,
                                          simplenet.trainOp1,
                                          simplenet.generator1],
                                         feed_dict={simplenet.x1: input1,
                                                    simplenet.x2: input3,
                                                    })
                mark = i

                g1 = g1[0].reshape(4160, 2768, 3)
                # print(g1)
                print(str(i), ':', 'loss:', loss1)

                # if i % 10 == 0:
                #     cv2.imwrite('result/' + str(mark) + '_step_' + 'result1.jpg', g1,
                #                 [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            r1 = g1

            cv2.imwrite('result/' + 'document' + str(j) + '_' + str(mark + 1) + '_step_' + 'fin_result1.jpg', g1,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer(),
                     sess.run(tf.local_variables_initializer())
                     )
            for i in range(tain_eps):
                loss1, _1, g1 = sess.run([simplenet.loss1,
                                          simplenet.trainOp1,
                                          simplenet.generator1],
                                         feed_dict={simplenet.x1: input2,
                                                    simplenet.x2: input4,
                                                    })
                mark = i

                g1 = g1[0].reshape(4160, 2768, 3)
                # print(g1)
                print(str(i), ':', 'loss:', loss1)

                # if i % 10 == 0:
                #     cv2.imwrite('result/' + str(mark) + '_step_' + 'result2.jpg', g1,
                #                 [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            r2 = g1

            cv2.imwrite('result/' + 'document' + str(j) + '_' + str(mark + 1) + '_step_' + 'fin_result2.jpg', g1,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            r3 = cv2.add(r1, r2).reshape(1, 4160, 2768, 3)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer(),
                         sess.run(tf.local_variables_initializer())
                         )
                for i in range(tain_eps):
                    loss1, _1, g1 = sess.run([simplenet.loss1,
                                              simplenet.trainOp1,
                                              simplenet.generator1],
                                             feed_dict={simplenet.x1: output,
                                                        simplenet.x2: r3,
                                                        })
                    mark = i

                    g1 = g1[0].reshape(4160, 2768, 3)
                    # print(g1)
                    print(str(i), ':', 'loss:', loss1)

                    # if i % 10 == 0:
                    #     cv2.imwrite('result/' + str(mark) + '_step_' + 'result2.jpg', g1,
                    #                 [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                fin = g1

                cv2.imwrite('result/' + 'document' + str(j) + '_' + str(mark + 1) + '_step_' + 'fin_result.jpg', fin,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            # r3 = cv2.add(r1, r2)
            #
            # cv2.imwrite('result/' + 'document' + str(j) + '_' + str(mark + 1) + '_step_' + 'fin_result.jpg', fin,
            #             [int(cv2.IMWRITE_JPEG_QUALITY), 100])


# def main():
#     input1, input2, input3, input4, output = getData(4)
#     simplenet = simpleNet(0, 1)
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer(),
#                  sess.run(tf.local_variables_initializer())
#                  )
#         for i in range(400):
#             t1d, t1g, D1_loss, G1_loss, g1 = sess.run([simplenet.trainOp1_D,
#                                                        simplenet.trainOp1_G,
#                                                        simplenet.d1_real_prob,
#                                                        simplenet.d1_fake_prob,
#                                                        simplenet.generator1],
#                                                       feed_dict={simplenet.x1: input1,
#                                                                  simplenet.x2: input2,
#                                                                  simplenet.x3: input3,
#                                                                  simplenet.x4: input4,
#                                                                  simplenet.y: output,
#                                                                  })
#             mark = i
#             # g1 = g1[0].reshape(260, 173, 1)
#             g1 = g1[0].reshape(4160, 2768, 3)
#             print(g1)
#
#             cv2.imwrite('result/' + str(mark) + 'step_' + 'result.jpg', g1, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
#
#             print('D1_loss:', D1_loss, ' ', ' G1_loss', G1_loss)
#
#         g1 = sess.run([simplenet.generator1,
#                        ],
#                       feed_dict={simplenet.x1: input1,
#                                  simplenet.x2: input2,
#                                  simplenet.x3: input3,
#                                  simplenet.x4: input4,
#                                  simplenet.y: output,
#                                  })
#         g1 = g1[0].reshape(4160, 2768, 3)
#
#         cv2.imwrite('result/' + str(mark) + 'step_' + 'result.jpg', g1, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


if __name__ == '__main__':
    main()
