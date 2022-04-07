import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import os
import argparse
import imageio
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from libs.datasets import MNIST
from libs import utils
from libs.vae import VAE
from pca import pca


def test_mnist(n_epochs=10000,
               convolutional=False,
               variational=False,
               metric=False,
               fire=False,
               order=-1,
               output_path="result_mnist",
               ckpt_name="vae.ckpt"):
    """Train an autoencoder on MNIST.

    This function will train an autoencoder on MNIST and also
    save many image files during the training process, demonstrating
    the latent space of the inner most dimension of the encoder,
    as well as reconstructions of the decoder.
    """
    # load MNIST
    n_code = 32
    n_clusters = 30
    mnist = MNIST(split=[0.9, 0.05, 0.05])
    ae = VAE(
        input_shape=[None, 784],
        n_filters=[512, 256, 128],
        n_hidden=64,
        n_code=n_code,
        n_clusters=n_clusters,
        activation=tf.nn.sigmoid,
        convolutional=convolutional,
        variational=variational,
        metric=metric,
        fire=fire,
        order=order)

    if metric is True:
        output_path += str(order)

    n_examples = 4
    # rng = np.random.RandomState(1)
    np.random.seed(0)
    # print(np.random.get_state())
    zs = np.random.uniform(-1.0, 1.0, [4, n_code]).astype(np.float32)
    zs = utils.make_latent_manifold(zs, n_examples)

    learning_rate = 0.0002
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        ae['cost'])

    # We create a session to use the graph config = tf.ConfigProto()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.25
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(output_path + '/logs', sess.graph)

    if (os.path.exists(output_path + '/' + ckpt_name + '.index')
            or os.path.exists(ckpt_name)):
        saver.restore(sess, output_path + '/' + ckpt_name)
        print("Model restored")

    # Fit all training data
    t_i = 0
    batch_i = 0
    batch_size = 256
    train_xs = mnist.test.images[:n_examples**2]
    utils.montage(train_xs.reshape((-1, 28, 28)), output_path + '/input.png')

    for epoch_i in range(n_epochs):
        train_i = 0
        train_cost = 0
        for batch_xs, batch_ys in mnist.train.next_batch(batch_size):
            summary, cost_batch, _ = sess.run(
                [ae['merged'], ae['cost'], optimizer],
                feed_dict={
                    ae['x']: batch_xs,
                    ae['t']: batch_xs,
                    ae['label']: np.where(batch_ys == 1)[1],
                    ae['train']: True,
                    ae['keep_prob']: 1.0
                })

            train_cost += cost_batch

            # Get new centroids
            nebula1d, nebula2d, nebula3d = sess.run(
                [ae['nebula1d'], ae['nebula2d'], ae['nebula3d']],
                feed_dict={
                    ae['x']: batch_xs,
                    ae['train']: False,
                    ae['keep_prob']: 1.0
                })
            samp_plot = batch_xs * 255
            samp_plot = samp_plot.astype(np.uint8)
            for idx in range(np.shape(batch_xs)[0]):
                imageio.imwrite(output_path + '/x-' + str(idx) + ".png",
                                  samp_plot[idx, :].reshape(28, 28))
                imageio.imwrite(
                    output_path + '/z-' + str(np.shape(batch_xs)[0] - 1 - idx)
                    + ".png", samp_plot[idx, :].reshape(28, 28))
            np.save(output_path + '/nebula3d.npy', nebula3d)
            np.save(output_path + '/nebula2d.npy', nebula2d)
            np.save(output_path + '/nebula1d.npy', nebula1d)
            feat = sess.run(
                ae['z'],
                feed_dict={
                    ae['x']: batch_xs,
                    ae['train']: False,
                    ae['keep_prob']: 1.0
                })
            np.save(output_path + '/feat.npy', feat)
            np.save(output_path + '/label.npy', np.where(batch_ys == 1)[1])

            train_i += 1
            if batch_i % 50 == 0:
                train_writer.add_summary(
                    summary,
                    epoch_i * (mnist.train.images.shape[0] / batch_size) +
                    train_i)
                # Plot example reconstructions from latent layer
                recon = sess.run(
                    ae['y'],
                    feed_dict={
                        ae['z']: zs,
                        ae['train']: False,
                        ae['keep_prob']: 1.0
                    })
                # utils.montage(recon.reshape((-1, 28, 28)),
                #    output_path + '/manifold_%08d.png' % t_i)
                utils.montage(
                    recon.reshape((-1, 28, 28)), output_path + '/manifold.png')

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
                        recon.reshape((-1, 28, 28)),
                        output_path + '/manifold_%03d.png' % cat)

                # Plot example reconstructions
                recon = sess.run(
                    ae['t'],
                    feed_dict={
                        ae['t']: train_xs[:n_examples**2],
                        ae['train']: False,
                        ae['keep_prob']: 1.0
                    })
                utils.montage(
                    recon.reshape((-1, 28, 28)), output_path + '/target.png')

                recon = sess.run(
                    ae['y'],
                    feed_dict={
                        ae['x']: train_xs[:n_examples**2],
                        ae['train']: False,
                        ae['keep_prob']: 1.0
                    })
                utils.montage(
                    recon.reshape((-1, 28, 28)), output_path + '/recon.png')

                t_i += 1
                valid_i = 0
                valid_cost = 0
                latent = []
                label_viz = []
                nebula_index = []
                for batch_xs, batch_ys in mnist.valid.next_batch(batch_size):
                    valid_cost += sess.run(
                        [ae['cost']],
                        feed_dict={
                            ae['x']: batch_xs,
                            ae['t']: batch_xs,
                            ae['label']: np.where(batch_ys == 1)[1],
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
                                ae['train']: False,
                                ae['keep_prob']: 1.0
                            })[0])
                    label_viz = np.append(label_viz,
                                          batch_ys.argmax(1)).astype(int)
                    nebula_index = np.append(
                        nebula_index,
                        sess.run(
                            [ae['nebula_index']],
                            feed_dict={
                                ae['x']: batch_xs,
                                ae['train']: False,
                                ae['keep_prob']: 1.0
                            })[0])
                    valid_i += 1
                latent = np.reshape(latent, (-1, n_code))

                print('train:', train_cost / train_i, 'valid:',
                      valid_cost / valid_i)

                # Start ploting distributions on latent space
                # PCA
                _, V_temp = pca(latent, dim_remain=2)
                """
                if t_i is 1:
                    V = V_temp
                V = 0.8 * V + 0.2 * V_temp
                """
                V = V_temp
                latent_viz = np.matmul(latent, V)
                nebula_viz = np.matmul(nebula3d, V)
                # t-SNE
                """
                latent_viz = TSNE(n_components=2).fit_transform(latent)
                nebula_viz = TSNE(n_components=2).fit_transform(nebula3d)
                """
                fig = plt.figure()
                ax = fig.add_subplot(111)  # , projection='3d')
                ax.set_axis_off()
                # ax.set_aspect('equal')
                """
                if n_clusters < 8:
                    cmlist = 'Accent'
                elif n_clusters < 12:
                    cmlist = 'tab10'
                else:
                """ 
                cmlist = 'rainbow'
                ax.scatter(
                    latent_viz[:, 0],
                    latent_viz[:, 1],
                    c=nebula_index,
                    alpha=0.5,
                    cmap=cmlist)
                if metric is True:
                    ax.scatter(
                        nebula_viz[:, 0],
                        nebula_viz[:, 1],
                        marker='H',
                        alpha=0.9,
                        s=200,
                        c='black')

                fig.savefig(
                    output_path + '/scatter_unsupervised.png',
                    transparent=True)
                plt.close(fig)
                fig = plt.figure()
                ax = fig.add_subplot(111)  # , projection='3d')
                ax.set_axis_off()
                # ax.set_aspect('equal')
                ax.scatter(
                    latent_viz[:, 0],
                    latent_viz[:, 1],
                    c=label_viz,
                    alpha=0.5,
                    cmap='tab10')
                """
                if metric is True:
                    ax.scatter(
                        nebula_viz[:, 0],
                        nebula_viz[:, 1],
                        marker='H',
                        alpha=0.9,
                        s=150,
                        c='black')
                """

                fig.savefig(
                    output_path + '/scatter_label.png', transparent=True)
                plt.close(fig)

                # Save the variables to disk.
                # We should set global_step=batch_i if we want several ckpt
                saver.save(
                    sess,
                    output_path + "/" + ckpt_name,
                    global_step=None,
                    write_meta_graph=False)
            batch_i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser added')
    parser.add_argument(
        '-c',
        action="store_true",
        dest="convolutional",
        help='Whether use convolution or not')
    parser.add_argument(
        '-f',
        action="store_true",
        dest="fire",
        help='Whether use fire module or not')
    parser.add_argument(
        '-v',
        action="store_true",
        dest="variational",
        help='Wether use latent variance or not')
    parser.add_argument(
        '-m',
        action="store_true",
        dest="metric",
        help='Whether use metric loss or not')
    parser.add_argument(
        '-r',
        action="store",
        type=int,
        dest="rank",
        help='Rank of metric learning')
    parser.add_argument(
        '-o',
        action="store",
        dest="output_path",
        default="result_vae",
        help='Destination for storing results')
    parser.print_help()
    results = parser.parse_args()
    test_mnist(
        convolutional=results.convolutional,
        fire=results.fire,
        variational=results.variational,
        metric=results.metric,
        order=results.rank,
        output_path=results.output_path)
