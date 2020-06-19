# -*- coding: utf-8 -*-

import sys
import dlib
from skimage import io
import matplotlib.pyplot as plt
import render_result as rr
import cv2



def faceDectorWithScore(imgPath):
    img = io.imread(imgPath)
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR);
    dets, scores, idx = detector.run(img, 1, -1)
    for i, d in enumerate(dets):
        print("Detection {}, score: {}, face_type:{}".format(
            d, scores[i], idx[i]))

def faceDetector(imgPath):
    detector = dlib.get_frontal_face_detector()
    
    for f in imgPath:
        print("Processing file: {}".format(f))
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        print "img shape:", img.shape
        print "img type:", type(img)

        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        print "dets type:", type(dets)
        for i, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                i, d.left(), d.top(), d.right(), d.bottom()))

            rect=(d.left(), d.top(), d.right(), d.bottom())
            img_rect = rr.draw_rectangle(img, rect, (255,0,0))
            

        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        cv2.imwrite(r"E:\Projects\WorkDir\face_data\face_image\Aaron_detected.jpg", img)

if __name__ == "__main__" :
    ImagePath = []
    ImagePath.append(r'E:\Projects\WorkDir\face_data\face_image\Aaron.jpg')
    faceDetector(ImagePath)
