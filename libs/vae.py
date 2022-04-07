"""
Clustered/Convolutional/Variational autoencoder, including demonstration of
training such a network on MNIST, CelebNet and the film, "Sita Sings The Blues"
using an image pipeline.

Copyright Yida Wang, May 2017
"""
import tensorflow as tf
from libs.batch_norm import batch_norm
from libs import utils
from libs.metric import sparse_ml
from libs.vb import variational_bayes


def VAE(input_shape=[None, 784],
        output_shape=[None, 784],
        n_filters=[64, 64, 64],
        filter_sizes=[4, 4, 4],
        n_hidden=32,
        n_code=2,
        n_clusters=10,
        activation=tf.nn.tanh,
        dropout=False,
        denoising=False,
        convolutional=False,
        fire=False,
        variational=False,
        metric=False,
        order=-1):
    """(Variational) (Convolutional) (Denoising) (Clustered) Autoencoder.

    Uses tied weights.

    Parameters
    ----------
    input_shape : list, optional
        Shape of the input to the network. e.g. for MNIST: [None, 784].
    n_filters : list, optional
        Number of filters for each layer.
        If convolutional=True, this refers to the total number of output
        filters to create for each layer, with each layer's number of output
        filters as a list.
        If convolutional=False, then this refers to the total number of neurons
        for each layer in a fully connected network.
    filter_sizes : list, optional
        Only applied when convolutional=True.  This refers to the ksize (height
        and width) of each convolutional layer.
    n_hidden : int, optional
        Only applied when variational=True.  This refers to the first fully
        connected layer prior to the variational embedding, directly after
        the encoding.  After the variational embedding, another fully connected
        layer is created with the same size prior to decoding.  Set to 0 to
        not use an additional hidden layer.
    n_code : int, optional
        Only applied when variational=True.  This refers to the number of
        latent Gaussians to sample for creating the inner most encoding.
    activation : function, optional
        Activation function to apply to each layer, e.g. tf.relu
    dropout : bool, optional
        Whether or not to apply dropout.  If using dropout, you must feed a
        value for 'keep_prob', as returned in the dictionary.  1.0 means no
        dropout is used.  0.0 means every connection is dropped.  Sensible
        values are between 0.5-0.8.
    denoising : bool, optional
        Whether or not to apply denoising.  If using denoising, you must feed a
        value for 'corrupt_prob', as returned in the dictionary.  1.0 means no
        corruption is used.  0.0 means every feature is corrupted.  Sensible
        values are between 0.5-0.8.
    convolutional : bool, optional
        Whether or not to use a convolutional network or else a fully connected
        network will be created.  This effects the n_filters parameter's
        meaning.
    fire: bool, optional
        Whether or not to use fire modules described like those in SqueezeNet
        or not. Otherwise it will be traditional convalution layer.
    variational : bool, optional
        Whether or not to create a variational embedding layer.  This will
        create a fully connected layer after the encoding, if `n_hidden` is
        greater than 0, then will create a multivariate gaussian sampling
        layer, then another fully connected layer.  The size of the fully
        connected layers are determined by `n_hidden`, and the size of the
        sampling layer is determined by `n_code`.

    Returns
    -------
    model : dict
        {
            'cost': Tensor to optimize.
            'Ws': All weights of the encoder.
            'x': Input Placeholder
            't': Target Placeholder
            'z': Inner most encoding Tensor (latent features)
            'y': Reconstruction of the Decoder
            'centroids': Centers of the latent spaces
            'keep_prob': Amount to keep when using Dropout
            'corrupt_prob': Amount to corrupt when using Denoising
            'train': Set to True when training/Applies to Batch Normalization.
        }
    """
    # network input / placeholders for train (bn) and dropout
    x = tf.placeholder(tf.float32, input_shape, 'x')
    t = tf.placeholder(tf.float32, output_shape, 't')
    label = tf.placeholder(tf.int32, [None], 'label')
    # Nebula 3D with shape [n_cluster, n_codes]
    nebula3d = tf.Variable(
        tf.truncated_normal([n_clusters, n_code]), 'nebula3d')
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    corrupt_prob = tf.placeholder(tf.float32, [1])

    # apply noise if denoising
    x_ = (utils.corrupt(x) * corrupt_prob +
          x * (1 - corrupt_prob)) if denoising else x

    # 2d -> 4d if convolution
    x_tensor = utils.to_tensor(x_) if convolutional else x_
    current_input = x_tensor

    Ws = []
    shapes = []

    # Build the encoder
    for layer_i, n_output in enumerate(n_filters):
        with tf.variable_scope('encoder/{}'.format(layer_i)):
            shapes.append(current_input.get_shape().as_list())
            if convolutional:
                if fire:
                    h, W = utils.conv2d_squeeze(
                        x=current_input, n_output=n_output)
                else:
                    h, W = utils.conv2d(
                        x=current_input,
                        n_output=n_output,
                        k_h=filter_sizes[layer_i],
                        k_w=filter_sizes[layer_i])
            else:
                h, W = utils.linear(x=current_input, n_output=n_output)
            h = activation(batch_norm(h, phase_train, 'bn' + str(layer_i)))
            if dropout:
                h = tf.nn.dropout(h, keep_prob)
            Ws.append(W)
            current_input = h

    shapes.append(current_input.get_shape().as_list())

    dims = current_input.get_shape().as_list()
    flattened = tf.contrib.layers.flatten(current_input)

    if n_hidden:
        h = utils.linear(flattened, n_hidden, name='W_fc')[0]
        h = activation(batch_norm(h, phase_train, 'fc/bn'))
        if dropout:
            h = tf.nn.dropout(h, keep_prob)
    else:
        h = flattened

    # Sample from posterior
    with tf.variable_scope('variational'):
        if variational:
            z, z_mu, z_log_sigma, loss_z = variational_bayes(
                h=h, n_code=n_code)
            tf.summary.histogram('LatentMean', z_mu)
            tf.summary.histogram('LatentCov', z_log_sigma)
        else:
            z = utils.linear(h, n_code, name='mu')[0]

    loss_0d, loss_1d, loss_2d, loss_3d, nebula1d, nebula2d, nebula3d, nebula_index, dist_0d  = sparse_ml(
        n_clusters, n_code, nebula3d, z, label, info_type='unsupervised')
    current_input = tf.concat([z, tf.nn.softmax(dist_0d, axis=1)], -1)

    if n_hidden:
        h = utils.linear(current_input, n_hidden, name='fc_t')[0]
        h = activation(batch_norm(h, phase_train, 'fc_t/bn'))
        if dropout:
            h = tf.nn.dropout(h, keep_prob)
    else:
        h = z

    size = dims[1] * dims[2] * dims[3] if convolutional else dims[1]
    h = utils.linear(h, size, name='fc_t2')[0]
    current_input = activation(batch_norm(h, phase_train, 'fc_t2/bn'))
    if dropout:
        current_input = tf.nn.dropout(current_input, keep_prob)

    if convolutional:
        current_input = tf.reshape(
            current_input,
            tf.stack([tf.shape(current_input)[0], dims[1], dims[2], dims[3]]))

    shapes[0] = utils.to_tensor(t).get_shape().as_list()
    shapes.reverse()
    n_filters.reverse()
    Ws.reverse()

    n_filters += [output_shape[-1]]

    # %%
    # Decoding layers
    for layer_i, n_output in enumerate(n_filters[1:]):
        with tf.variable_scope('decoder/{}'.format(layer_i)):
            shape = shapes[layer_i + 1]
            if convolutional:
                if fire:
                    h, W = utils.deconv2d_squeeze(
                        x=current_input,
                        n_output_h=shape[1],
                        n_output_w=shape[2],
                        n_output_ch=shape[3],
                        n_input_ch=shapes[layer_i][3])
                else:
                    h, W = utils.deconv2d(
                        x=current_input,
                        n_output_h=shape[1],
                        n_output_w=shape[2],
                        n_output_ch=shape[3],
                        n_input_ch=shapes[layer_i][3],
                        k_h=filter_sizes[layer_i],
                        k_w=filter_sizes[layer_i])
            else:
                h, W = utils.linear(x=current_input, n_output=n_output)
            h = activation(batch_norm(h, phase_train, 'dec/bn' + str(layer_i)))
            if dropout:
                h = tf.nn.dropout(h, keep_prob)
            current_input = h

    y = current_input

    # l2 loss
    # this is for regression
    mask = tf.tile(tf.expand_dims(tf.cast(tf.math.greater(tf.reduce_sum(t, 3), 0), tf.float32), -1), [1,1,1,3])
    # t_flat = tf.contrib.layers.flatten(t)
    # y_flat = tf.contrib.layers.flatten(y)
    # mask is for Yanyan
    loss_t = tf.reduce_sum(tf.squared_difference(t * mask, y * mask), [1, 2, 3])
    loss_t += tf.reduce_sum(tf.contrib.layers.flatten((1-tf.reduce_sum(t*y, 3))*tf.reduce_mean(tf.cast(tf.math.greater(t, 0), tf.float32), 3)), 1)
    # this for binary mask generation
    """
    mask = tf.contrib.layers.flatten(tf.tanh(t * 100))
    t_flat = tf.contrib.layers.flatten(t)
    y_flat = tf.contrib.layers.flatten(tf.clip_by_value(tf.nn.sigmoid(y), 1e-6, 1 - 1e-6))
    loss_t = -tf.reduce_sum(mask * tf.log(y_flat) + (1-mask) * tf.log(1-y_flat), 1)
    """

    tf.summary.scalar('reconstruction cost', tf.reduce_mean(loss_t))
    tf.summary.scalar(
        'entropy1',
        tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.nn.softmax(z), logits=z)))
    tf.summary.scalar(
        'entropy2',
        tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.nn.softmax(z), logits=z)))

    if variational:
        cost = tf.reduce_mean(loss_t + loss_z)
        tf.summary.scalar('encoding cost', tf.reduce_mean(loss_z))
        tf.summary.scalar('inference cost', tf.reduce_mean(loss_t + loss_z))
    else:
        cost = tf.reduce_mean(loss_t)

    if metric:
        # unsupervised learning
        cost += loss_0d
        # supervised learning
        if order is 1:
            cost += loss_1d
        elif order is 2:
            cost += loss_2d
        elif order is 3:
            cost += loss_3d
        elif order is -1:
            cost += loss_1d
            cost += loss_3d

    tf.summary.scalar('unsupervised clustering cost', loss_0d)
    tf.summary.scalar('order-1 metric learning cost', loss_1d)
    tf.summary.scalar('order-2 metric learning cost', loss_2d)
    tf.summary.scalar('order-3 metric learning cost', loss_3d)

    merged = tf.summary.merge_all()

    return {
        'cost': cost,
        'Ws': Ws,
        'x': x,
        't': t,
        'mask': mask,
        'z': z,
        'y': y,
        'label': label,
        'nebula1d': nebula1d,
        'nebula2d': nebula2d,
        'nebula3d': nebula3d,
        'nebula_index': nebula_index,
        'keep_prob': keep_prob,
        'corrupt_prob': corrupt_prob,
        'train': phase_train,
        'merged': merged
    }
