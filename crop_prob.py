#!/usr/bin/env python
# -*- coding:utf8 -*-

import sys
import os
import facedecor_by_dlib

def crop_img_by_dlib(src_file_path, dest_video_path,img_file):
    facedecor_by_dlib.process(src_file_path, dest_video_path,img_file)

def crop_probs(aligned_db_folder, result_folder):
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    src_people_path=aligned_db_folder+'/'
    dest_people_path=result_folder+'/'
    i = 0
    img_count = 0
    for video_folder in os.listdir(src_people_path):
        src_video_path = src_people_path + video_folder + '/'
        dest_video_path = dest_people_path + video_folder + '/'
        if not os.path.exists(dest_video_path):
           os.mkdir(dest_video_path)
        for img_file in os.listdir(src_video_path):
           src_img_path = src_video_path + img_file
#           dest_img_path = dest_video_path + img_file
           crop_img_by_dlib(src_img_path, dest_video_path,img_file)
        i += 1
        img_count += len(os.listdir(src_video_path))
        sys.stdout.write('\rsub_folder: %d, imgs %d' % (i, img_count) )
        sys.stdout.flush()
    print ''
        
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: python %s aligned_db_folder new_folder' % (sys.argv[0])
        sys.exit()
    aligned_db_folder = sys.argv[1]
    result_folder = sys.argv[2]
    if not aligned_db_folder.endswith('/'):
        aligned_db_folder += '/'
    if not result_folder.endswith('/'):
        result_folder += '/'
    crop_probs(aligned_db_folder, result_folder)

