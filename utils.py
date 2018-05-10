import numpy as np
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
import dlib
import cv2
import os

emo_list = {'angry': 0,'contemptuous': 1,'disgusted': 2,'fearful': 3,'happy': 4,'neutral': 5,'sad': 6,'surprised': 7}

def load_data(train_list,landmark):

    data = []
    label = []
    shape = []
    for file in train_list:

        tag = os.path.basename(file)[:-4].split('_')
        im = cv2.imread(file)
        if im is None:
            continue
        data.append(im)
        label.append(emo_list[tag[2]])
        shape.append(landmark[file[:-4]])
    label = one_hot(label, emo_cls=8)

    return data, label, shape

def one_hot(label, emo_cls):

    label_oh = np.zeros(len(label),dtype = np.int)
    for i in range(len(label)):
        label_oh[i] = int(label[i])
    label = np.eye(emo_cls)[label_oh]

    return label

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
 
    return coords

def draw_pts(img,points):

    im_clone = deepcopy(img)
    l = len(points)
    for i in range(l):
        x = int(points[i][0])
        y = int(points[i][1])
        a = cv2.circle(im_clone,(x,y),1,(0,0,255),2)
    return im_clone

def dlib_detect(img, down_scale, face_det, lm_det, w, h):
    '''
    dlib face detection
    face_det: dlib face detector
    lm_det: facial landmark detetor
    w & h: width and height of the desired shape of output
    '''
    im = cv2.resize(img,(int(img.shape[1]/down_scale),int(img.shape[0]/down_scale)),interpolation=cv2.INTER_CUBIC)
    shift = int(im.shape[0]*0.02)
    dets = face_det(im, 1)
    shape = 0
    res = 0
    shape_origin = 0
    for i, d in enumerate(dets):

        shape = lm_det(im, d)
        shape = shape_to_np(shape)
        shape_origin = shape*down_scale

        top = np.max((np.min(shape[:,1]) - shift, 0))
        bottom = np.min((np.max(shape[:,1]) + shift, w))
        left = np.max((np.min(shape[:,0]) - shift, 0))
        right = np.min((np.max(shape[:,0]) + shift, h))

        # map landmark to face image
        shape[:,0] -= left
        shape[:,1] -= top
        res, shape = resize_data(im[top:bottom, left:right], shape, w, h)
    
    return len(dets), res, shape, shape_origin

def print_process(itera, summation, t):

    bar = []
    total_char = 60
    finish = int(itera/summation*total_char)
    not_finish = total_char - finish
    if finish:
        for i in range(finish):
            bar.append('>')
    for i in range(not_finish):
        bar.append('-')
    bar = "".join(bar)

    #print('processing batch: %d/%d: [ %s ], remaining time: %d sec' %(itera,summation, bar, int(t*(summation-itera))),end='\r') 

def gen_batch(im, lm, gt, batch_size):

    num = len(im)
    for i in range(num // batch_size + 1):

        start = i * batch_size
        end = np.min([start + batch_size, num])
        if start != end:

            assert im.shape[1:] == (224, 224, 3)

            yield im[start:end], preprocess_shape(lm[start:end]), gt[start:end]

def split_data(im, lm, gt, split_ratio):

    indices = np.arange(im.shape[0])  
    np.random.shuffle(indices) 
    
    im_data = im[indices]
    lm_data = lm[indices]
    gt_data = gt[indices]
    
    num_val = int(split_ratio * im_data.shape[0] )
    
    im_train = im_data[num_val:]
    lm_train = lm_data[num_val:]
    gt_train = gt_data[num_val:]

    im_val = im_data[:num_val]
    lm_val = lm_data[:num_val]
    gt_val = gt_data[:num_val]

    return (im_train, lm_train, gt_train), (im_val, lm_val, gt_val)

def preprocess_shape(shape):

	data_size = len(shape)
	shape_norm = np.zeros([data_size, 51*2])
	for i in range(data_size):
		shape[i] = shape[i] - shape[i][30]
		shape_norm[i,:] = shape[i].reshape(-1)[17*2:]

	return shape_norm

def resize_data(img, shape, w, h):

    row, col, channel = img.shape
    res = cv2.resize(img,(w,h),interpolation=cv2.INTER_CUBIC)
    shape[:,1] = shape[:,1]*(w/row)  
    shape[:,0] = shape[:,0]*(h/col) 

    return res, shape

def show_detection(img, shape, down_scale, emotion):

    row, col, channel = img.shape
    shape_clone = shape*(down_scale)

    shift = int(row*0.02)
    top = np.max(np.min(shape_clone[:,1]) - shift, 0)
    bottom = np.max(shape_clone[:,1]) + shift
    left = np.max(np.min(shape_clone[:,0]) - shift, 0)
    right = np.max(shape_clone[:,0]) + shift

    im_clone = deepcopy(img)
    im_clone = cv2.rectangle(im_clone,(left,top),(right,bottom),(0,140,255),7)
    im_clone = draw_pts(im_clone,shape_clone)

    im_clone = Image.fromarray(im_clone)
    draw = ImageDraw.Draw(im_clone)
    draw.rectangle((left-4,top-30,left+150,top), fill=(0,140,255))

    # fontSize = min(xSize, ySize) // 11
    myFont = ImageFont.truetype("./OsakaMono.ttf", 30)
    draw.text((left,top-30),emotion, fill = (255,255,255), font=myFont)
    im_clone = np.asarray(im_clone)

    return im_clone
