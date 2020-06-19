#!/usr/bin/env python
# coding=utf-8
import dlib
import os
import sys
import numpy
from PIL import Image, ImageDraw
import cv2
from skimage import io  as imgIO
PREDICTOR_PATH = r"E:\Projects\DL\DataSet\model_img\shape_predictor_68_face_landmarks.dat"

SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

BasePath = r'E:\Projects\DL\DataSet\lfw\lfw_funneled'

ImgAll = BasePath + r'\Img_all.txt'
NoDetectFaceFile =  BasePath + r'\no_detect_face_file.txt'
DetectFaceFile = BasePath + r'\detect_face_file.txt'
DetectMultiFaceFile = BasePath + r'\detect_multi_face_file.txt'

def get_landmarks(im,rect):
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()]);

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    R = (U * Vt).T
    
    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

#def read_im_and_landmarks(img,rect):
##    im = cv2.imread(fname, cv2.IMREAD_COLOR);
##    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
##                         im.shape[0] * SCALE_FACTOR));
#    landmarks = get_landmarks(img,rect);
#
#    return landmarks;

def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype);
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP);
    return output_im;

def Transform(src_img,des_file,model_img,rect_infile, rect_model):
    landmarks_modle = get_landmarks(model_img,rect_model);         
    landmarks_src = get_landmarks(src_img,rect_infile);

    M = transformation_from_points(landmarks_modle[ALIGN_POINTS],
                               landmarks_src[ALIGN_POINTS])
    warped_mask = warp_im(src_img, M, model_img.shape)
    cv2.imwrite(des_file,warped_mask);
    
def detect_object(img):
    rects=detector(img,1);

    return rects;

def read_img(img_file):
    print("Processing file: {}".format(img_file));
    im = cv2.imread(img_file, cv2.IMREAD_COLOR);
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR));
    return  im;

def process(ImagePath, savePathCroped, model_file):
    check_parameter(ImagePath, 'directory', False)
    check_parameter(savePathCroped, 'directory', True)   
    
    list_dirs, list_files = listdirsAndfiles(ImagePath)
    list_files.sort(key=lambda x:len(x))
    
    img_model = read_img(model_file);
    rects_model= detect_object(img_model);   
                              
    fid_img_all = open(ImgAll, 'w')
    fid_no_detect_face = open(NoDetectFaceFile, 'w')
    fid_detect_face = open(DetectFaceFile, 'w')
    fid_detect_multi_face = open(DetectMultiFaceFile, 'w')
    
    count_process = 0
    
    print "Total Imgs:", len(list_files)
    for img_file in list_files:
        count_process += 1

        if 0 == count_process%20000:
            print "Process Imgs:{0}".format(count_process)
            
        fid_img_all.write(img_file+'\n')
        
        img_src = read_img(img_file)                          
        rects_src= detect_object(img_src)
        if len(rects_src) <= 0:
            fid_no_detect_face.write(img_file+'\n') 
            continue

        face_rects=[];
        for count in range(len(rects_src)):
            left=rects_src[count].left();
            top=rects_src[count].top();
            right=rects_src[count].left()+rects_src[count].width();
            bottom=rects_src[count].top()+rects_src[count].height();
            face_rects.append((left,top,right,bottom));             
    
        save_file_name = make_sub_dir(ImagePath, savePathCroped, img_file)                     
        try:
            if len(face_rects) == 1:
                face = face_rects[0]
                line = img_file +','+str(face[0])+','+str(face[1])+','+str(face[2])+','+str(face[3])+'\n'
                fid_detect_face.write(line)
                Transform(img_src,save_file_name,img_model,rects_src[0], rects_model[0])
            elif len(face_rects) > 1:
                max_face,index = get_center_face(img_src, face_rects)
                line = img_file +','+str(max_face[0])+','+str(max_face[1])+','+str(max_face[2])+','+str(max_face[3])+'\n'
                fid_detect_face.write(line)
                fid_detect_multi_face.write(img_file +'\n')
                Transform(img_src,save_file_name,img_model,rects_src[index], rects_model[0])
            else:
                print "Error: cannot detect faces on %s" % img_file  
        except:
            print "Exception rasied!!!"
            assert(0)
        
    fid_img_all.close()
    fid_no_detect_face.close()
    fid_detect_face.close()
    fid_detect_multi_face.close()
    
def get_noexist_filelist(ImagePath, savePathCroped):
    list_dirs, list_files = listdirsAndfiles(ImagePath)
    list_files_ret = []
    for img_file in list_files:
        img_file_1 = img_file.replace(ImagePath, savePathCroped)
        if not os.path.exists(img_file_1):
            list_files_ret.append(img_file)
            
    return list_files_ret

def max_area_face(face_rects):
    index = 0;
    max_area = 0;
    count = 0;
    for face in face_rects:
        area = (face[2]-face[0])*(face[3]-face[1])
        if area > max_area:
            max_area= area
            index = count
            
        count += 1
        
    return face_rects[index], index

def get_center_face(img, face_rects):
    index = 0
    center = (img.shape[1] / 2, img.shape[0] / 2)
    distance_to_center = 99999.0
    count = 0
    for face in face_rects:
        distance = abs((face[0] + face[2]) / 2 - center[0]) + abs((face[1] + face[3]) / 2 - center[1])
        if (distance < distance_to_center):
            distance_to_center = distance
            index = count
            
        count += 1
            
    return face_rects[index], index 

def make_sub_dir(ImagePath, savePathCroped, img_file):
    check_parameter(ImagePath, 'directory', False)
    check_parameter(savePathCroped, 'directory', True)  
    check_parameter(img_file, 'file')
    
    img_file = img_file.replace(ImagePath, savePathCroped)
    pos = img_file.rfind('\\')
    sub_dirs = img_file[:pos]
    check_parameter(sub_dirs, 'directory', True)  
        
    return img_file
        
def check_parameter(file_path, file_type, create_new_if_missing=False):
    assert file_type == 'file' or file_type == 'directory'
    if file_type == 'file':
        assert os.path.exists(file_path)
        assert os.path.isfile(file_path)
    else:
        if create_new_if_missing is True:
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            else:
                assert os.path.isdir(file_path)
        else:
            assert os.path.exists(file_path)
            assert os.path.isdir(file_path)
            
def listdirsAndfiles(top_dir):
    check_parameter(top_dir, 'directory', False)
    
    tmp_file_lists = os.listdir(top_dir)
    file_lists = []
    dir_lists = []
    
    for e in tmp_file_lists:
        full_path = os.path.join(top_dir,e)
        if os.path.isdir(full_path):            
            sub_dir_lists, sub_file_lists = listdirsAndfiles(full_path)
            if len(sub_file_lists) > 0:
                file_lists.extend(sub_file_lists)
        else:
            if e.endswith('.jpg') or e.endswith('.png') or e.endswith('.bmp'):
                file_lists.append(full_path)

    return dir_lists, file_lists

def insert_text_linenum(filename,index=1):
    fid=open(filename)
    lines=fid.readlines()
    fid.close()
    
    text=str(len(lines))+'\n'
    lines.insert(index-1,text)
    s=''.join(lines)
    fid=open(filename,'w')
    fid.write(s)
    fid.close()
    
if __name__=="__main__":
    ImagePath =  r"E:\Projects\DL\DataSet\lfw\lfw_funneled\origin"
    savePathCroped = r"E:\Projects\DL\DataSet\lfw\lfw_funneled\croped_112_96_dlib_MTCNN01"
    model_file= r'E:\Projects\DL\DataSet\model_img\Aaron_Eckhart_MTCNN01.jpg'
    process(ImagePath,savePathCroped,model_file)

