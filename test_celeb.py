import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import argparse
from libs.train_vae import train_vae


def test_celeb(n_epochs=50):
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
        help='whether use fire module or not')
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
        default="result_celeb",
        help='Destination for storing results')
    parser.print_help()
    results = parser.parse_args()
    """Train an autoencoder on Celeb Net.
    """
    train_vae(
        files_train="./list_attr_celeba_origin.csv",
        files_valid="./list_attr_celeba_short.csv",
        input_shape=[218, 178, 3],
        output_shape=[218, 178, 3],
        batch_size=64,
        n_epochs=n_epochs,
        crop_shape=[64, 64],
        crop_factor=1.0,
        convolutional=results.convolutional,
        fire=results.fire,
        variational=results.variational,
        metric=results.metric,
        order=results.rank,
        n_filters=[128, 128, 128],
        n_hidden=128,
        n_code=64,
        n_clusters=2,
        dropout=True,
        filter_sizes=[3, 3, 3],
        activation=tf.nn.sigmoid,
        ckpt_name='./celeb.ckpt',
        output_path=results.output_path)


if __name__ == '__main__':
    test_celeb()
