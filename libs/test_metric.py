import tensorflow as tf
import numpy as np
tf.enable_eager_execution()


def sparse_ml(nebula1d,
              nebula2d,
              nebula3d=tf.constant(np.random.rand(5, 32)),
              feat=tf.constant(np.random.rand(8, 32)),
              label=tf.constant([0, 1, 2, 4, 3, 1, 1, 2]),
              n_clusters=10,
              n_code=32,
              info_type='scalar'):

    # feat should be a batch of vectors, so the rank should be 2
    # tf.assert_rank(feat, 2)

    with tf.variable_scope('Anchors'):
        """
        # Nebula 2D with shape [1, n_codes]
        nebula2d = tf.transpose(
                tf.matmul(
                    tf.transpose(nebula3d),
                    tf.Variable(
                        tf.truncated_normal([n_clusters, 1]))))

        # Nebula 1D is a scalar [1]
        nebula1d = tf.sigmoid(tf.reduce_mean(
                tf.matmul(
                    nebula2d,
                    tf.Variable(
                        tf.truncated_normal([n_code, 1])))))
        """
    # Metric learning for latent variables
    with tf.variable_scope('Metric'):
        # dist_0d -> [N]
        dist_0d = tf.reduce_sum(
            tf.square(
                tf.subtract(
                    tf.expand_dims(feat, 2),
                    tf.expand_dims(tf.transpose(nebula3d), 0))), 1)
        loss_0d = tf.reduce_mean(tf.reduce_min(dist_0d, 1))

        # 1D
        # Note that we should convert the labels starting from 0 for
        # one-hot coding in Tensorflow
        # If `indices` is a vector of length `features`,
        # the output shape will be:
        # features x depth if axis == -1
        # depth x features if axis == 0
        onehots = tf.one_hot(label, n_clusters, axis=-1, dtype=tf.int32)
        np.save('onehots.npy', onehots)
        np.save('label.npy', label)

        # dist_1d -> [N]
        dist_1d = tf.reduce_sum(
            tf.square(
                tf.subtract(feat,
                            tf.matmul(tf.cast(onehots, tf.float32),
                                      nebula3d))), 1)
        np.save('dist_1d.npy', dist_1d)
        loss_1d = tf.reduce_mean(dist_1d)

        # 2D
        # dist_2d -> [N, N]
        # positive -> [N, N]
        dist_2d = tf.reduce_sum(
            tf.square(
                tf.subtract(tf.expand_dims(feat, 0), tf.expand_dims(feat, 1))),
            2)
        np.save('dist_2d.npy', dist_2d)

        # Here I remove the self similarity by add a diagnal martrix of 1
        '''
        positive = tf.subtract(
                tf.matmul(
                    onehots,
                    tf.transpose(onehots)),
                tf.diag(
                    tf.ones([batch_size])))
        '''
        positive = tf.matmul(onehots, tf.transpose(onehots))

        negative = tf.subtract(tf.ones_like(positive), positive)
        np.save('transpose_onehots.npy', tf.transpose(onehots))
        np.save('positive.npy', positive)
        np.save('negative.npy', negative)

        # Loss function in 2D space based on positive relationships
        loss_positive = tf.reduce_mean(
            tf.multiply(dist_2d, tf.cast(positive, tf.float32)))

        loss_negative = tf.reduce_mean(
            tf.multiply(
                tf.abs(tf.subtract(1.0, dist_2d)), tf.cast(
                    negative, tf.float32)))
        loss_2d = loss_positive  # + loss_negative
        # 3D
        # dist_3d -> [N, N, N]
        # triplet -> [N, N, N]
        # Both positive and negative matrix with shape [N, N] are like the
        # one of dist_2d, we should expand it now by adding additional
        # dimension. It will be dist_2d(x,y) - dist_2d(x,z), so we can just
        # expand the dimension on z and y respectively which means that
        # we should expand the dimension on 2 and 1
        x_equal_y = -tf.sign(
            tf.cast(
                tf.tile(tf.expand_dims(tf.diag(tf.ones([16])), 2), [1, 1, 16])
                - 1, tf.int32))
        x_equal_z = -tf.sign(
            tf.cast(
                tf.tile(tf.expand_dims(tf.diag(tf.ones([16])), 1), [1, 16, 1])
                - 1, tf.int32))
        dist_3d = tf.divide(
            tf.expand_dims(dist_2d, 2),
            tf.add(tf.expand_dims(dist_2d, 1), nebula1d))
        dist_3d *= tf.cast(x_equal_y, tf.float32)
        dist_3d *= tf.cast(x_equal_z, tf.float32)
        np.save('dist_3d.npy', dist_3d)
        # 3D relationship if dependent on 2D relationships
        triplet = tf.multiply(
            tf.expand_dims(positive, 2), tf.expand_dims(negative, 1))
        triplet *= x_equal_y
        triplet *= x_equal_z
        np.save('triplet.npy', triplet)

        # Loss function in 3D space based on triplet relationships
        loss_3d = tf.divide(
            tf.reduce_sum(tf.multiply(dist_3d, tf.cast(triplet, tf.float32))),
            tf.cast(tf.count_nonzero(triplet), tf.float32))

    return loss_0d, loss_1d, loss_2d, loss_3d, nebula3d


label = np.load('label.npy')
feat = np.load('feat.npy')
nebula1d = np.load('nebula1d.npy')
nebula2d = np.load('nebula2d.npy')
nebula3d = np.load('nebula3d.npy')
sparse_ml(
    nebula1d=nebula1d,
    nebula2d=nebula2d,
    label=label,
    feat=feat,
    nebula3d=nebula3d)
