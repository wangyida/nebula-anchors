import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import argparse
from libs.generate_vae import generate_vae


def generate_depth():
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
        '-i',
        action="store",
        dest="input_path",
        default="./list_nyu_depth.csv",
        help='input csv files for images')
    parser.add_argument(
        '-o',
        action="store",
        dest="output_path",
        default="result_vae",
        help='Destination for storing results')
    parser.add_argument(
        '-g',
        action="store",
        dest="image_path",
        default="result_vae/generated",
        help='Destination for storing results')
    parser.print_help()
    results = parser.parse_args()

    # Train an autoencoder on Synthetic data rendered from ShapeNet.

    generate_vae(
        files_train=results.input_path,
        input_shape=[240, 320, 3],
        output_shape=[240, 320, 3],
        batch_size=120,
        crop_shape=[120, 160],
        crop_factor=1.0,
        convolutional=results.convolutional,
        fire=results.fire,
        variational=results.variational,
        metric=results.metric,
        n_filters=[32, 64, 128, 128, 128, 256],
        n_hidden=128,
        n_code=64,
        n_clusters=8,
        dropout=False,
        filter_sizes=[3, 3, 3, 3, 3, 3],
        activation=tf.nn.sigmoid,
        ckpt_name='bleenco.ckpt',
        image_path=results.image_path,
        output_path=results.output_path)


if __name__ == '__main__':
    generate_depth()
