# convert NWPU to CSV for airplane class
# cls: 1-10 ---  0-9
# 'airplane':0

# -*- coding:utf-8 -*-
# !/usr/bin/env python
# hx

import os
from six import raise_from
import cv2
import numpy as np
import random
import math

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

def findNode(parent, name, debug_name = None, parse = None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError('illegal value for \'{}\': {}'.format(debug_name, e)), None)
    return result

# input a number(str float) -- int
def str2int(numstr):
    num = numstr.split('.')

    return int(num[0])

def write_CSV(txt_list, output_path, start_index):
    for i in range(len(txt_list)):
        boxes = []
        txt_path = txt_list[i]

        print(txt_path)
        object_txt = open(data_txt_path + txt_path, 'r')
        lines = object_txt.read().strip().split('\n')

        for l in lines:
            new_l = []
            box = []
            temp = l.strip().split(',')
            for j in range(len(temp)):
                aa = temp[j]
                aa = aa.strip().strip(')(')
                new_l.append(aa)

            #x1, y1, x2, y2, cls = int(new_l[0]), int(new_l[1]), int(new_l[2]), int(new_l[3]), int(new_l[4])-1
            y1, x1, y2, x2, cls = int(new_l[0]), int(new_l[1]), int(new_l[2]), int(new_l[3]), int(new_l[4]) - 1

            box.append(x1)
            box.append(y1)
            box.append(x2)
            box.append(y2)
            box.append(cls)

            boxes.append(box)

        # write txt
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        img_path = output_path + str(start_index + i + 1) + '.jpg'
        txt_path = output_path + str(start_index + i + 1) + '.txt'
        txt_label = open(txt_path, 'w')

        img_name = txt_list[i].replace('txt', 'jpg')
        cv2.imwrite(img_path, cv2.imread(data_posimg_path + img_name))

        for m in range(len(boxes)):  # 10 cls
            txt_label.write(
                '{},{},{},{},{}\n'.format(0, boxes[m][0], boxes[m][1], boxes[m][2], boxes[m][3]))
        txt_label.close()

# def parse_annotation(element):
#     box = np.zeros((1, 5))
#     box[0, 4] = 0   #class
#
#     bndbox    = element
#     box[0, 0] = bndbox.attrib['LeftTopX']
#     box[0, 1] = bndbox.attrib['LeftTopY']
#
#     return box

data_txt_path = './NWPU_airplane/ground truth/'
data_posimg_path = './NWPU_airplane/positive image set/'
# output_path = './NWPU_CSV/'

# ground truth -- txt
txt_list = []
for root, dirs, files in os.walk(data_txt_path):
    for i in range(len(files)):
        if '.txt' in files[i]:
            txt_list.append(files[i])
            # print(files[i])
txt_list.sort()

posimg_list = []
for root, dirs, files in os.walk(data_posimg_path ):
    for i in range(len(files)):
        if '.jpg' in files[i]:
            posimg_list.append(files[i])
            # print(files[i])
posimg_list.sort()
print('num of images is %d' % (len(posimg_list)))

#random divided into 3 parts: train: val: test = 6:2:2
num_all = len(txt_list)
# num_train = math.floor(num_all * 0.6)
num_train = num_all
num_val = math.floor(num_all * 0.2)
num_test = num_all - num_train - num_val
print('num_all: %d, train: %d, val:%d, test:%d\n' % (num_all, num_train, num_val, num_test))

random.shuffle(txt_list)
train_list = txt_list[:num_train]
val_list = txt_list[num_train: num_train + num_val]
test_list = txt_list[num_train + num_val: num_train + num_val + num_test]

# train
train_outpath = './NWPU_CSV_airplane/train/'
val_outpath = './NWPU_CSV_airplane/val/'
test_outpath = './NWPU_CSV_airplane/test/'

write_CSV(train_list, train_outpath, 0)
# write_CSV(val_list, val_outpath, num_train)
# write_CSV(test_list, test_outpath, num_val + num_train)

print('generate finished.')




