"""
Clustered/Convolutional/Variational autoencoder, including demonstration of
training such a network on MNIST, CelebNet and the film, "Sita Sings The Blues"
using an image pipeline.

Copyright Yida Wang, May 2017
"""

import matplotlib
import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import os
import csv
from libs.dataset_utils import create_input_pipeline
from libs.vae import VAE
# Those lines must be setted when we ssh other servers without displaying
matplotlib.use('Agg')


def generate_vae(files_train,
                 input_shape=[None, 784],
                 output_shape=[None, 784],
                 batch_size=128,
                 n_examples=6,
                 crop_shape=[64, 64],
                 crop_factor=1,
                 n_filters=[100, 100, 100, 100],
                 n_hidden=256,
                 n_code=50,
                 n_clusters=12,
                 convolutional=True,
                 fire=True,
                 variational=True,
                 denoising=False,
                 metric=False,
                 filter_sizes=[3, 3, 3, 3],
                 dropout=True,
                 keep_prob=1.0,
                 activation=tf.nn.relu,
                 output_path="result",
                 image_path="images",
                 ckpt_name="vae.ckpt"):
    """General purpose training of a (Variational) (Convolutional) (Clustered)
        Autoencoder.

    Supply a list of file paths to images, and this will do everything else.

    Parameters
    ----------
    files_train : list of strings
        List of paths to images for training.
    input_shape : list
        Must define what the input image's shape is.
    batch_size : int, optional
        Batch size.
    n_examples : int, optional
        Number of example to use while demonstrating the current training
        iteration's reconstruction.  Creates a square montage, so:
        n_examples**2 = 16, 25, 36, ... 100.
    crop_shape : list, optional
        Size to centrally crop the image to.
    crop_factor : float, optional
        Resize factor to apply before cropping.
    n_filters : list, optional
        Same as VAE's n_filters.
    n_hidden : int, optional
        Same as VAE's n_hidden.
    n_code : int, optional
        Same as VAE's n_code.
    convolutional : bool, optional
        Use convolution or not.
    fire: bool, optional
        Use fire module or not.
    variational : bool, optional
        Use variational layer or not.
    filter_sizes : list, optional
        Same as VAE's filter_sizes.
    dropout : bool, optional
        Use dropout or not
    keep_prob : float, optional
        Percent of keep for dropout.
    activation : function, optional
        Which activation function to use.
    img_step : int, optional
        How often to save training images showing the manifold and
        reconstruction.
    save_step : int, optional
        How often to save checkpoints.
    output_path : str, optional
        Defien a path for saving result and sample images
    ckpt_name : str, optional
        Checkpoints will be named as this, e.g. 'model.ckpt'
    """

    # Those should be defined before we finalize the graph
    batch_train = create_input_pipeline(
        files=files_train,
        batch_size=batch_size,
        n_epochs=8,
        crop_shape=crop_shape,
        crop_factor=crop_factor,
        input_shape=input_shape,
        output_shape=output_shape,
        shuffle=False)

    ae = VAE(
        input_shape=[None] + crop_shape + [input_shape[-1]],
        output_shape=[None] + crop_shape + [output_shape[-1]],
        convolutional=convolutional,
        variational=variational,
        fire=fire,
        metric=metric,
        n_filters=n_filters,
        n_hidden=n_hidden,
        n_code=n_code,
        n_clusters=n_clusters,
        dropout=dropout,
        filter_sizes=filter_sizes,
        activation=activation)

    # We create a session to use the config = tf.ConfigProto()
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # This will handle our threaded image pipeline
    coord = tf.train.Coordinator()

    # Ensure no more changes to graph
    tf.get_default_graph().finalize()

    # Start up the queues for handling the image pipeline
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if (os.path.exists(output_path + '/' + ckpt_name + '.index')
            or os.path.exists(ckpt_name)):
        saver.restore(sess, output_path + '/' + ckpt_name)
        print("Model restored.")
    else:
        print("No model, train at first!")

    # Get the number of training samples
    with open(files_train, "r") as f:
        reader = csv.reader(f, delimiter=",")
        data = list(reader)
        n_files_train = len(data)

    batch_i = 0

    try:
        while not coord.should_stop() and batch_i < n_files_train / batch_size:
            import time
            tic = time.clock()
            batch_xs, batch_ts, batch_ys = sess.run(batch_train)
            toc = time.clock()
            print(toc - tic)
            batch_xs = batch_xs.astype(np.float32) / 255.0
            batch_ts = batch_ts.astype(np.float32) / 255.0

            # Plot example reconstructions
            input_x, recon, target = sess.run(
                [ae['x'], ae['y'], ae['t']],
                feed_dict={
                    ae['x']: batch_xs,
                    ae['t']: batch_ts,
                    ae['train']: False,
                    ae['keep_prob']: 1.0
                })[:]

            # This is for avoiding 0 in the denomiters
            depth_recon = recon[:, :, :, -1] + 0.001
            depth_target = target[:, :, :, -1]

            for sam_id in range(target.shape[0]):
                name_img = data[batch_i * batch_size + sam_id][0]
                label_start = name_img.rfind('_') + 1
                label_end = name_img.rfind('.')
                imsave(
                    arr=input_x[sam_id, :],
                    name=image_path + '/input_' +
                    name_img[label_start:label_end] + '.png')
                imsave(
                    arr=np.squeeze(target[sam_id, :]),
                    name=image_path + '/target_' +
                    name_img[label_start:label_end] + '.png')
                imsave(
                    arr=np.squeeze(recon[sam_id, :]),
                    name=image_path + '/recon_' +
                    name_img[label_start:label_end] + '.png')
                if np.shape(recon[sam_id, :])[-1] == 1:
                    imsave(
                        arr=np.append(
                            input_x[sam_id, :], recon[sam_id, :], axis=-1),
                        name=image_path + '/masked_' +
                        name_img[label_start:label_end] + '.png')
                elif np.shape(recon[sam_id, :])[-1] == 3:
                    imsave(
                        arr=np.append(
                            input_x[sam_id, :],
                            np.expand_dims(
                                recon[sam_id][:, :, 0] + recon[sam_id][:, :, 1]
                                + recon[sam_id][:, :, 2], -1) / 3,
                            axis=-1),
                        name=image_path + '/masked_' +
                        name_img[label_start:label_end] + '.png')

            # Evaluation for depth images
            valid_pos = np.nonzero(depth_target)
            delta1 = (np.count_nonzero(
                np.maximum(
                    1.25 - np.maximum(
                        depth_recon[valid_pos] / depth_target[valid_pos],
                        depth_target[valid_pos] / depth_recon[valid_pos]), 0))
                      / depth_target[valid_pos].size)
            delta2 = (np.count_nonzero(
                np.maximum(
                    1.25**2 - np.maximum(
                        depth_recon[valid_pos] / depth_target[valid_pos],
                        depth_target[valid_pos] / depth_recon[valid_pos]), 0))
                      / depth_target[valid_pos].size)
            delta3 = (np.count_nonzero(
                np.maximum(
                    1.25**3 - np.maximum(
                        depth_recon[valid_pos] / depth_target[valid_pos],
                        depth_target[valid_pos] / depth_recon[valid_pos]), 0))
                      / depth_target[valid_pos].size)
            rel = (np.mean(
                np.abs(depth_recon[valid_pos] - depth_target[valid_pos]) /
                depth_target[valid_pos]))

            print('rel:', rel, ', delta 1:', delta1, ', delta 2:', delta2,
                  ', delta 3:', delta3)
            batch_i += 1

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        # One of the threads has issued an exception.  So let's tell all the
        # threads to shutdown.
        coord.request_stop()

    # Wait until all threads have finished.
    coord.join(threads)

    # Clean up the session.
    sess.close()
