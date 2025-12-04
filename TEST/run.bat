mkdir OUT
python ../CODE/fuse.py IN/building.png OUT/ --strength 0.7
python ../CODE/fuse.py IN/portrait.png OUT/ --strength 0.7
mkdir OUT/PRODUCT
python ../CODE/fuse.py IN/PRODUCT/ OUT/PRODUCT/ --strength 0.8
mkdir OUT/TESTIMONIAL
python ../CODE/fuse.py IN/TESTIMONIAL/ OUT/TESTIMONIAL/ --strength 0.6
