import argparse
from scipy import misc
import glob
from libs import gif


def gif_stack():
    parser = argparse.ArgumentParser(description='Parser added')
    parser.add_argument(
        '-i',
        action="store",
        dest="input",
        default="./result_shapenet_clvae/reconstruction_",
        help='Traits for finding images')
    parser.add_argument(
        '-o',
        action="store",
        dest="output",
        default="vae.gif",
        help='Name of output GIF file')
    parser.print_help()
    results = parser.parse_args()
    # Start to scan images on by one
    imgs = []

    for image_path in glob.glob(results.input + '*.png'):
        image = misc.imread(image_path)
        imgs.append(image)

    gif.build_gif(imgs, saveto=results.output)


if __name__ == '__main__':
    gif_stack()
