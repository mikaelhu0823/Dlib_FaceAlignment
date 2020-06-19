# -*- coding: utf-8 -*-

import sys
import os
import dlib
import glob
from skimage import io

if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")


def facelandmark_detection(imgPath, predictor_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    win = dlib.image_window()

    for f in imgPath:
        print("Processing file: {}".format(f))
        img = io.imread(f)
    
        win.clear_overlay()
        win.set_image(img)
    
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            shape = predictor(img, d)
            print "shape type:", type(shape)
            print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                      shape.part(1)))
            win.add_overlay(shape)
    
        win.add_overlay(dets)
        dlib.hit_enter_to_continue()
    
if __name__ == "__main__" :
    ImagePath = []
    ImagePath.append(r'D:\PreRD\AL\Software\dlib-19.4\examples\faces\2007_007763.jpg')
    predictor_path = r'D:\WorkDir\Spyder\Mnist_test\Dlib_FaceAlignment\shape_predictor_68_face_landmarks.dat'
    facelandmark_detection(ImagePath, predictor_path)