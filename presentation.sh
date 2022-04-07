# Generate images based on trained models
mkdir result_{1..4}/generated
# list_1_d2n.csv is the inversly validation list while list_1_n2d.csv is the training list
python3 generate_images.py -c -v -i list_1_n2d.csv -o result_1
python3 generate_images.py -c -v -i list_2_n2d.csv -o result_2
python3 generate_images.py -c -v -i list_3_n2d.csv -o result_3
python3 generate_images.py -c -v -i list_4_n2d.csv -o result_4

# Define a main function
main()
{
	folder=$1 
	for f in ${folder}input_*.png 
	do 
		convert $f -size 10x xc:none ${f/input/target} -size 10x xc:none ${f/input/recon} \
			+append ${f/input/merged} 
	done 
	convert -delay 15 -loop 0 ${folder}merged_*.png \
		${folder}bleenco_scene.gif
}

# Scanning for each folders and make the animation
main result_1/generated/
main result_2/generated/
main result_3/generated/
main result_4/generated/
