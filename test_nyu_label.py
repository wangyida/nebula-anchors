import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import argparse
from libs.train_vae import train_vae


def test_depth():
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
        '-o',
        action="store",
        dest="output_path",
        default="result_vae",
        help='Destination for storing results')
    parser.print_help()
    results = parser.parse_args()

    # Train an autoencoder on Synthetic data rendered from ShapeNet.

    train_vae(
        files_train="./list_nyu_label_train.csv",
        files_valid="./list_nyu_label_test.csv",
        input_shape=[240, 320, 3],
        output_shape=[240, 320, 4],
        batch_size=16,
        n_epochs=5000,
        crop_shape=[240, 320],
        crop_factor=1.0,
        convolutional=results.convolutional,
        fire=results.fire,
        variational=results.variational,
        metric=results.metric,
        n_filters=[256, 128, 128, 128, 128, 128],
        n_hidden=None,
        n_code=128,
        n_clusters=28,
        dropout=True,
        filter_sizes=[5, 3, 3, 3, 3, 3],
        activation=tf.nn.relu,
        ckpt_name='depth.ckpt',
        output_path=results.output_path)


if __name__ == '__main__':
    test_depth()
