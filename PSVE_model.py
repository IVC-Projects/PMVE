import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn
import numpy as np
from PIL import Image

#  第一层就是三个，第二层、第三层都是俩（为了过渡），然后开始6个残差单元（3 * 6 = 18 layers）
def model_single(frame2, reuse = False, scope='netflow'):

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], activation_fn=tflearn.activations.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)), \
             slim.arg_scope([slim.conv2d_transpose], activation_fn=tflearn.activations.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)):

            # feature extration

            c12 = slim.conv2d(frame2, 64, [3, 3], scope='conv1_2')
            conv = c12
            for i in range(10):

                c1 = slim.conv2d(conv, 64, [1, 1], scope='convB_%02d' % (i))
                convC = slim.conv2d(c1, 64, [3, 3], scope='convC_%02d' % (i))
                c2 = slim.conv2d(convC, 64, [1, 1], scope='convA_%02d' % (i))
                conv = tf.add(conv, c2)

            c5 = slim.conv2d(conv, 1, [5, 5], activation_fn=None, scope='conv5')

            # enhanced frame reconstruction
            output = tf.add(c5, frame2)

        return output
