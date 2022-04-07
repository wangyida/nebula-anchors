# python3 test_mnist.py -v -c -o result_mnist_vae_4 &
# python3 test_mnist.py -c -v -m -r -1 -o result_mnist_nvcsml_4 &
# python3 test_mnist.py -c -v -m -o result_mnist_nvc_4 &
# python3 test_mnist.py -c -o result_mnist_ae_4 &
# python3 test_mnist.py -v -c -o result_mnist_vae_3 &
# python3 test_mnist.py -c -v -m -r -1 -o result_mnist_nvcsml_3 &
# python3 test_mnist.py -c -v -m -o result_mnist_nvc_3 &
# python3 test_mnist.py -c -o result_mnist_ae_3 &
# wait
# echo all processes complete
## 

CUDA_VISIBLE_DEVICES=0 python3 test_mnist.py -c -v -m -r 0 -o result_mnist &
CUDA_VISIBLE_DEVICES=0 python3 test_mnist.py -c -v -m -r 1 -o result_mnist &
CUDA_VISIBLE_DEVICES=0 python3 test_mnist.py -c -v -m -r 2 -o result_mnist &
CUDA_VISIBLE_DEVICES=0 python3 test_mnist.py -c -v -m -r 3 -o result_mnist &
CUDA_VISIBLE_DEVICES=0 python3 test_mnist.py -c -v -m -r -1 -o result_mnist &
wait
echo all processes complete
