"""
Clustered/Convolutional/Variational autoencoder, including demonstration of
training such a network on MNIST, CelebNet and the film, "Sita Sings The Blues"
using an image pipeline.

Copyright Yida Wang, May 2017
"""

import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import os
import csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from libs.dataset_utils import create_input_pipeline
from libs.vae import VAE
from pca import pca
from libs import utils


def train_vae(files_train,
              files_valid='',
              input_shape=[None, 784],
              output_shape=[None, 784],
              learning_rate=0.0001,
              batch_size=128,
              n_epochs=50,
              n_examples=6,
              crop_shape=[64, 64],
              crop_factor=0.8,
              n_filters=[100, 100, 100, 100],
              n_hidden=256,
              n_code=50,
              n_clusters=12,
              convolutional=True,
              fire=True,
              variational=True,
              metric=False,
              order=-1,
              filter_sizes=[3, 3, 3, 3],
              dropout=True,
              keep_prob=0.8,
              activation=tf.nn.relu,
              img_step=1000,
              save_step=2000,
              output_path="result",
              ckpt_name="vae.ckpt"):
    """
    General purpose training of a (Variational) (Convolutional) (Clustered)
        Autoencoder.

    Supply a list of file paths to images, and this will do everything else.

    Parameters
    ----------
    files_train : list of strings
        List of paths to images for training.
    files_valid : list of strings
        List of paths to images for validation.
    input_shape : list
        Must define what the input image's shape is.
    learning_rate : float, optional
        Learning rate.
    batch_size : int, optional
        Batch size.
    n_epochs : int, optional
        Number of epochs.
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
    metric : bool, optional,
        Use metric learning based on label or not.
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
        n_epochs=n_epochs,
        crop_shape=crop_shape,
        crop_factor=crop_factor,
        input_shape=input_shape,
        output_shape=output_shape)

    if files_valid != '':
        batch_valid = create_input_pipeline(
            files=files_valid,
            batch_size=batch_size,
            n_epochs=n_epochs,
            crop_shape=crop_shape,
            crop_factor=crop_factor,
            input_shape=input_shape,
            output_shape=output_shape)

    ae = VAE(
        input_shape=[None] + crop_shape + [input_shape[-1]],
        output_shape=[None] + crop_shape + [output_shape[-1]],
        convolutional=convolutional,
        fire=fire,
        variational=variational,
        metric=metric,
        denoising=False,
        order=order,
        n_filters=n_filters,
        n_hidden=n_hidden,
        n_code=n_code,
        n_clusters=n_clusters,
        dropout=dropout,
        filter_sizes=filter_sizes,
        activation=activation)

    if metric is True:
        output_path += str(order)
    # Create a manifold of our inner most layer to show
    # example reconstructions.  This is one way to see
    # what the "embedding" or "latent space" of the encoder
    # is capable of encoding, though note that this is just
    # a random hyperplane within the latent space, and does not
    # encompass all possible embeddings.
    np.random.seed(1)
    # print(np.random.get_state())
    zs = np.random.uniform(-1.0, 1.0, [4, n_code]).astype(np.float32)
    zs = utils.make_latent_manifold(zs, 6)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        ae['cost'])

    # We create a session to use the config = tf.ConfigProto()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(output_path + '/logs', sess.graph)

    # This will handle our threaded image pipeline
    coord = tf.train.Coordinator()

    # Ensure no more changes to graph
    tf.get_default_graph().finalize()

    # Start up the queues for handling the image pipeline
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if (os.path.exists(output_path + '/' + ckpt_name + '.index')
            or os.path.exists(ckpt_name)):
        saver.restore(sess, output_path + '/' + ckpt_name)
        print("Model restored")

    # Get the number of training samples
    with open(files_train, "r") as f:
        reader = csv.reader(f, delimiter=",")
        data = list(reader)
        n_files_train = len(data)

    # Arrange some training data for summary
    train_xs, train_ts, train_ys = sess.run(batch_train)
    train_xs = train_xs.astype(np.float32) / 255.0
    train_ts = train_ts.astype(np.float32) / 255.0

    for idx in range(0, 4):
        temp_xs, temp_ts, temp_ys = sess.run(batch_train)
        train_xs = np.append(
            train_xs, temp_xs.astype(np.float32) / 255.0, axis=0)
        train_ts = np.append(
            train_ts, temp_ts.astype(np.float32) / 255.0, axis=0)
        train_ys = np.append(train_ys, temp_ys, axis=0)

    utils.montage(train_xs[:n_examples**2], output_path + '/input_train.png')
    utils.montage(train_ts[:n_examples**2], output_path + '/target_train.png')

    if files_valid != '':
        # Get the number of validation samples
        with open(files_valid, "r") as f:
            reader = csv.reader(f, delimiter=",")
            data = list(reader)
            n_files_valid = len(data)
        valid_xs, valid_ts, valid_ys = sess.run(batch_valid)
        valid_xs = valid_xs.astype(np.float32) / 255.0
        valid_ts = valid_ts.astype(np.float32) / 255.0
        utils.montage(valid_xs[:n_examples**2],
                      output_path + '/input_valid.png')
        utils.montage(valid_ts[:n_examples**2],
                      output_path + '/target_valid.png')

    # Fit all training data
    t_i = 0
    batch_i = 0
    epoch_i = 0
    cost = 0

    try:
        while not coord.should_stop() and epoch_i < n_epochs:
            batch_i += 1
            batch_xs, batch_ts, batch_ys = sess.run(batch_train)
            batch_xs = batch_xs.astype(np.float32) / 255.0
            batch_ts = batch_ts.astype(np.float32) / 255.0
            train_cost, _ = sess.run(
                [ae['cost'], optimizer],
                feed_dict={
                    ae['x']: batch_xs,
                    ae['t']: batch_ts,
                    ae['label']: batch_ys[:, 0],
                    ae['train']: True,
                    ae['keep_prob']: keep_prob
                })

            # write summary
            summary = sess.run(
                ae['merged'],
                feed_dict={
                    ae['x']: train_xs,
                    ae['t']: train_ts,
                    ae['label']: train_ys[:, 0],
                    ae['train']: False,
                    ae['keep_prob']: 1.0
                })

            # Get new centroids
            nebula3d = sess.run(
                ae['nebula3d'],
                feed_dict={
                    ae['train']: False,
                    ae['keep_prob']: 1.0
                })

            cost += train_cost
            if batch_i % (n_files_train / batch_size) == 0:
                train_writer.add_summary(
                    summary,
                    epoch_i * (n_files_train / batch_size) + batch_i)
                print('epoch:', epoch_i)
                print('training cost:', cost / batch_i)
                cost = 0
                batch_i = 0
                epoch_i += 1

            if batch_i % img_step == 1:
                # Plot example reconstructions from latent layer
                recon = sess.run(
                    ae['y'],
                    feed_dict={
                        ae['z']: zs,
                        ae['train']: False,
                        ae['keep_prob']: 1.0
                    })
                # utils.montage(
                #    recon.reshape([-1] + crop_shape + [output_shape[-1]]),
                #    output_path + '/manifold_%08d.png' % t_i)
                utils.montage(
                    recon.reshape([-1] + crop_shape + [output_shape[-1]]),
                    output_path + '/manifold.png')

                # we can draw more interesting manifold
                for cat in range(nebula3d.shape[0]):
                    recon = sess.run(
                        ae['y'],
                        feed_dict={
                            ae['z']: zs / 1.2 + nebula3d[cat, :],
                            ae['train']: False,
                            ae['keep_prob']: 1.0
                        })
                    utils.montage(
                        recon.reshape([-1] + crop_shape + [output_shape[-1]]),
                        output_path + '/manifold_%03d.png' % cat)

                # Plot example reconstructions
                recon = sess.run(
                    ae['mask'],
                    feed_dict={
                        ae['t']: train_ts[:n_examples**2],
                        ae['train']: False,
                        ae['keep_prob']: 1.0
                    })
                utils.montage(
                    recon.reshape([-1] + crop_shape + [output_shape[-1]]),
                    output_path + '/mask_train.png')

                recon = sess.run(
                    ae['y'],
                    feed_dict={
                        ae['x']: train_xs[:n_examples**2],
                        ae['train']: False,
                        ae['keep_prob']: 1.0
                    })
                utils.montage(
                    recon.reshape([-1] + crop_shape + [output_shape[-1]]),
                    output_path + '/recon_train.png')
                if np.shape(recon)[-1] == 1:
                    arr = np.append(train_xs[:n_examples**2], recon, axis=-1)
                    utils.montage(
                        arr.reshape([-1] + crop_shape + [4]),
                        output_path + '/masked_train.png')
                t_i += 1

                # test for images in another list
                if files_valid != '':
                    # Start for testing
                    # Plot example reconstructions
                    recon = sess.run(
                        ae['mask'],
                        feed_dict={
                            ae['t']: valid_ts[:n_examples**2],
                            ae['train']: False,
                            ae['keep_prob']: 1.0
                        })
                    utils.montage(
                        recon.reshape([-1] + crop_shape + [output_shape[-1]]),
                        output_path + '/mask_valid.png')
                    recon = sess.run(
                        ae['y'],
                        feed_dict={
                            ae['x']: valid_xs[:n_examples**2],
                            ae['train']: False,
                            ae['keep_prob']: 1.0
                        })
                    # utils.montage(
                    #    recon.reshape([-1] + crop_shape + [output_shape[-1]]),
                    #    output_path+'/reconstruction_%08d.png' % t_i)
                    utils.montage(
                        recon.reshape([-1] + crop_shape + [output_shape[-1]]),
                        output_path + '/recon_valid.png')
                if np.shape(recon)[-1] == 1:
                    arr = np.append(valid_xs[:n_examples**2], recon, axis=-1)
                    utils.montage(
                        arr.reshape([-1] + crop_shape + [4]),
                        output_path + '/masked_valid.png')

                    valid_i = 0
                    valid_cost = 0
                    latent = []
                    label_viz = []
                    round_valid = int(n_files_valid / batch_size)
                    for i in range(0, round_valid):
                        batch_xs, batch_ts, batch_ys = sess.run(batch_valid)
                        batch_xs = batch_xs.astype(np.float32) / 255.0
                        batch_ts = batch_ts.astype(np.float32) / 255.0
                        valid_cost += sess.run(
                            [ae['cost']],
                            feed_dict={
                                ae['x']: batch_xs,
                                ae['t']: batch_ts,
                                ae['label']: batch_ys[:, 0],
                                ae['train']: False,
                                ae['keep_prob']: 1.0
                            })[0]
                        # Plot the latent variables
                        latent = np.append(
                            latent,
                            sess.run(
                                [ae['z']],
                                feed_dict={
                                    ae['x']: batch_xs,
                                    ae['t']: batch_ts,
                                    ae['train']: False,
                                    ae['keep_prob']: 1.0
                                })[0])
                        label_viz = np.append(label_viz,
                                              batch_ys[:, 0]).astype(int)
                        valid_i += 1
                    latent = np.reshape(latent, (-1, n_code))

                    print('validation cost:', valid_cost / valid_i)

                    # Start ploting distributions on latent space
                    # PCA
                    _, V_temp = pca(latent, dim_remain=2)
                    if t_i is 1:
                        V = V_temp
                    V = 0.8 * V + 0.2 * V_temp
                    latent_viz = np.matmul(latent, V)
                    nebula_viz = np.matmul(nebula3d, V)
                    # t-SNE
                    # latent_viz = TSNE(n_components=3).fit_transform(latent)
                    fig = plt.figure()
                    # ax = fig.add_subplot(111, projection='3d')
                    ax = fig.add_subplot(111)
                    ax.set_axis_off()
                    # ax.set_aspect('equal')
                    if n_clusters < 8:
                        cmlist = 'Accent'
                    elif n_cluster < 12:
                        cmlist = 'tab10'
                    else:
                        cmlist = 'rainbow'
                    ax.scatter(
                        latent_viz[:, 0],
                        latent_viz[:, 1],
                        c=label_viz,
                        alpha=0.3,
                        cmap=cmlist)
                    if metric is True:
                        # you can choose whether start from 1 or 0 for index
                        ax.scatter(
                            nebula_viz[1:, 0],
                            nebula_viz[1:, 1],
                            c=np.arange(n_clusters - 1),
                            marker='H',
                            alpha=1,
                            s=250,
                            cmap=cmlist)

                    fig.savefig(output_path + '/scatter.png', transparent=True)
                    plt.close(fig)

            if batch_i % save_step == 0:
                # Save the variables to disk.
                # We should set global_step=batch_i if we want several ckpt
                saver.save(
                    sess,
                    output_path + "/" + ckpt_name,
                    global_step=None,
                    write_meta_graph=False)
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
