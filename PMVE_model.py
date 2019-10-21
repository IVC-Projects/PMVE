import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn

#  第一层就是三个，第二层、第三层都是俩（为了过渡），然后开始6个残差单元（3 * 6 = 18 layers）
def network(frame1, frame2, frame3, reuse = False, scope='netflow'):

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], activation_fn=tflearn.activations.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)), \
             slim.arg_scope([slim.conv2d_transpose], activation_fn=tflearn.activations.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)):

            # feature extration
            c11 = slim.conv2d(frame1, 64, [3, 3], scope='conv1_1')
            c12 = slim.conv2d(frame2, 64, [3, 3], scope='conv1_2')
            c13 = slim.conv2d(frame3, 64, [3, 3], scope='conv1_3')


            concat1_12 = tf.concat([c11, c12], 3, name='concat1_12')
            concat1_23 = tf.concat([c12, c13], 3, name='concat1_23')

            #feature merging
            concat12_1x1 = slim.conv2d(concat1_12, 64, [1, 1], scope='concat12_1x1')
            c21 = slim.conv2d(concat12_1x1, 64, [3, 3], scope='conv2_1')
            concat23_1x1 = slim.conv2d(concat1_23, 64, [1, 1], scope='concat23_1x1')
            c22 = slim.conv2d(concat23_1x1, 64, [3, 3], scope='conv2_2')

            # complex feature extration
            c31 = slim.conv2d(c21, 64, [3, 3], scope='conv3_1')
            c32 = slim.conv2d(c22, 64, [3, 3], scope='conv3_2')

            concat3_12 = tf.concat([c31, c32], 3, name='concat3_12')

            # rename!
            conv = concat3_12

            # non-linear mapping
            # residual reconstruction
            # residual cell 1
            for i in range(5):

                c1 = slim.conv2d(conv, 64, [1, 1], scope='convB_%02d' % (i))
                convC = slim.conv2d(c1, 64, [3, 3], scope='convC_%02d' % (i))
                c2 = slim.conv2d(convC, 128, [1, 1], scope='convA_%02d' % (i))
                conv = tf.add(conv, c2)

            c5 = slim.conv2d(conv, 1, [5, 5], activation_fn=None, scope='conv5')

            # enhanced frame reconstruction
            output = tf.add(c5, frame2)

        return output
