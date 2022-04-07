import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import argparse
from libs.train_vae import train_vae


def test_shapenet():
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
        default="result_shapenet",
        help='Destination for storing results')
    parser.print_help()
    results = parser.parse_args()

    # Train an autoencoder on Synthetic data rendered from ShapeNet.

    train_vae(
        files_train="./list_annotated_shapenet.csv",
        files_valid="./list_annotated_imagenet.csv",
        input_shape=[116, 116, 3],
        output_shape=[116, 116, 3],
        batch_size=32,
        n_epochs=5000,
        crop_shape=[112, 112],
        crop_factor=1.0,
        convolutional=results.convolutional,
        fire=results.fire,
        variational=results.variational,
        metric=results.metric,
        order=results.rank,
        n_filters=[128, 128, 128, 128, 128],
        n_hidden=128,
        n_code=64,
        n_clusters=13,
        dropout=True,
        filter_sizes=[3, 3, 3, 3, 3],
        activation=tf.nn.sigmoid,
        ckpt_name='./shapenet.ckpt',
        output_path=results.output_path)


if __name__ == '__main__':
    test_shapenet()
