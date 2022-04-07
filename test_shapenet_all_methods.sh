python3 test_shapenet.py -v -c -o result_shapenet_vae_2 &
python3 test_shapenet.py -c -v -m -r -1 -o result_shapenet_nvcsml_2 &
python3 test_shapenet.py -c -v -m -o result_shapenet_nvc_2 &
python3 test_shapenet.py -c -o result_shapenet_ae_2 &
wait
echo all processes complete
