import sys, os
import keras
import cv2
import traceback

from src.keras_utils 			import load_model
from glob 						import glob
from os.path 					import splitext, basename
from src.utils 					import im2single
from src.keras_utils 			import load_model, detect_lp
from src.label 					import Shape, writeShapes

def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))


if __name__ == '__main__':

	input_dir = 'ImagesforLD'
	lp_threshold = 0.5

	wpod_net_path = "data/lp-detector/wpod-net_update1.h5"
	wpod_net = load_model(wpod_net_path)


	imgs_paths = glob('%s/*car.png' % input_dir)

	print('Searching for license plates using WPOD-NET')

	print(len(imgs_paths))
	for i, img_path in enumerate(imgs_paths):
		print('\t Processing %s' % img_path)

		bname = splitext(basename(img_path))[0]
		Ivehicle = cv2.imread(img_path)

		ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
		side  = int(ratio*288.)
		bound_dim = min(side + (side%(2**4)),608)
		print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))