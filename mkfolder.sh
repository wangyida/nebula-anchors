# Shell script for testing
## Making directories for storing results
mkdir result_mnist_clae
mkdir result_mnist_ae
mkdir result_mnist_clvae
mkdir result_mnist_vae
mkdir result_sita_clae
mkdir result_sita_ae
mkdir result_sita_clvae
mkdir result_sita_vae
mkdir result_shapenet_clae
mkdir result_shapenet_ae
mkdir result_shapenet_clvae
mkdir result_shapenet_vae

## Four tests on MNIST databse with variational/clustered without convolutional
screen -r ae_mnist
CUDA_VISIBLE_DEVICES=0 python3 test_mnist.py -o result_mnist_ae

screen -r vae_mnist
CUDA_VISIBLE_DEVICES=1 python3 test_mnist.py -v -o result_mnist_vae

screen -r clae_mnist
CUDA_VISIBLE_DEVICES=2 python3 test_mnist.py -k  -o result_mnist_clae

screen -r clvae_mnist
CUDA_VISIBLE_DEVICES=3 python3 test_mnist.py -v -k -o result_mnist_clvae

## Four tests on ShapeNet databse with variational/clustered with convolutional
screen -r ae_shapenet
CUDA_VISIBLE_DEVICES=0 python3 test_shapenet.py -c -o result_shapenet_ae

screen -r vae_shapenet
CUDA_VISIBLE_DEVICES=1 python3 test_shapenet.py -c -v -o result_shapenet_vae

screen -r clae_shapenet
CUDA_VISIBLE_DEVICES=2 python3 test_shapenet.py -c -k -o result_shapenet_clae

screen -r clvae_shapenet
CUDA_VISIBLE_DEVICES=3 python3 test_shapenet.py -c -v -k -o result_shapenet_clvae

## Four tests on Sita databse with variational/clustered with convolutional
screen -r ae_sita
CUDA_VISIBLE_DEVICES=0 python3 test_sita.py -c -o result_sita_ae

screen -r vae_sita
CUDA_VISIBLE_DEVICES=1 python3 test_sita.py -c -v -o result_sita_vae

screen -r clae_sita
CUDA_VISIBLE_DEVICES=2 python3 test_sita.py -c -k -o result_sita_clae

screen -r clvae_sita
CUDA_VISIBLE_DEVICES=3 python3 test_sita.py -c -v -k -o result_sita_clvae

## Move montage figures into a target folder, this is executed on server jhwl
cd ~/Documents/yida/buildboat/cluster-vae
for f in result_*
do
  echo $f
  rsync -rR $f ../../gitfarm/cluster-vae/readme_images
done
cd ~/Documents/yida/gitfarm/cluster-vae
find readme_images/result_* -name "*ckpt*" | xargs rm -rf
find readme_images/result_* -name "*checkpoint*" | xargs rm -rf
find readme_images/result_* -name "*logs*" | xargs rm -rf

## Resize all images in jpg and png to new size to fit on squared map
## with a small size
cd readme_images
shopt -s nullglob
for image in *.jpg *.png; do
  mogrify -resize 256x256 "${image}"
done
shopt -u nullglob
cd ../

### Moving the logs
for f in result_*/logs
do
  echo $f
  rsync -rR $f ./logs
done

### Delete figures in buildboat folder
for f in result_*
do
  echo $f
  rm -rf $f/*
done
