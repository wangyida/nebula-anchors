# Nebula Variational Coding

![clusters](images/thesis_teaser_pami22.png#center)

BSD 3-Clause License Copyright (c) 2022, Yida Wang All rights reserved.

## Abstrarct

Deep learning approaches process data in a layer-by-layer way with intermediate (or latent) features. We aim at designing a general solution to optimize the latent manifolds to improve the performance on classification, segmentation, completion and/or reconstruction through probabilistic models. This paper proposes a variational inference model which leads to a clustered embedding. We introduce additional variables in the latent space, called *nebula anchors*, that guide the latent variables to form clusters during training. To prevent the anchors from clustering among themselves, we employ the variational constraint that enforces the latent features within an anchor to form a Gaussian distribution, resulting in a generative model we refer as Nebula Variational Coding (NVC). Since each latent feature can be labeled with the closest anchor, we also propose to apply metric learning in a self-supervised way to make the separation between clusters more explicit. As a consequence, the latent variables of our variational coder form clusters which adapt to the generated semantic of the training data, *e.g*. the categorical labels of each sample. We demonstrate experimentally that it can be used within different architectures designed to solve different problems including text sequence, images, 3D point clouds and volumetric data, validating the advantage of our proposed method. 

### Latent distributions
![clusters](images/clusters.png#center)

### Latent covariance
| Covariance matrix of each latent features |  |
| :-: | :-- |
![covariance](images/PAMI_covariance.png#center) | Comparison of covariance matrices computed from the latent features that relies on variational constraint..

## Training

```python3
convolutional : bool, optional
    Use convolution or not.
fire: bool, optional
    Use fire module or not.
variational : bool, optional
    Use variational layer or not.
metric : bool, optional,
    Use metric learning based on label or not.
```

## Testing

### MNIST

Testing with options of variational or/and nebula clusters without convolutions
```sh
CUDA_VISIBLE_DEVICES=0 python3 test_mnist.py -o result_mnist_ae &
CUDA_VISIBLE_DEVICES=1 python3 test_mnist.py -v -o result_mnist_vae &
CUDA_VISIBLE_DEVICES=2 python3 test_mnist.py -m -o result_mnist_tae &
CUDA_VISIBLE_DEVICES=3 python3 test_mnist.py -v -m -o result_mnist_tvae &
wait
echo all processes complete
```

### ShapeNet

Testing with options of variational or/and nebula clusters with convolutions
```sh
CUDA_VISIBLE_DEVICES=0 python3 test_shapenet.py -c -o result_shapenet_ae &
CUDA_VISIBLE_DEVICES=1 python3 test_shapenet.py -c -v -o result_shapenet_vae &
CUDA_VISIBLE_DEVICES=2 python3 test_shapenet.py -c -m -o result_shapenet_tae &
CUDA_VISIBLE_DEVICES=3 python3 test_shapenet.py -c -v -m -o result_shapenet_tvae &
wait
echo all processes complete
```

Squeezed modules *fire module* could be used to replace convolutional modules
```sh
screen -r shapenet
CUDA_VISIBLE_DEVICES=0 python3 test_shapenet.py -c -f -o result_shapenet_ae &
CUDA_VISIBLE_DEVICES=1 python3 test_shapenet.py -c -f -v -o result_shapenet_vae &
CUDA_VISIBLE_DEVICES=2 python3 test_shapenet.py -c -f -m -o result_shapenet_tae &
CUDA_VISIBLE_DEVICES=3 python3 test_shapenet.py -c -f -v -m -o result_shapenet_tvae &
wait
echo all processes complete
```

## Validating results

With the help of `./mkfolder.sh`, images could be resized to fit on squared map with a small scale

```sh
cd readme_images
shopt -s nullglob
for image in *.jpg *.png; do
  mogrify -resize 256x256 "${image}"
done
shopt -u nullglob
cd ../
```

## Cite

If you find this work useful in your research, please cite:

```bash
@article{wang2022self,
  title={Self-supervised Latent Space Optimization with Nebula Variational Coding},
  author={Wang, Yida and Tan, David Joseph and Navab, Nassir and Tombari, Federico},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```
