# -*- coding:utf-8 -*-
# !/usr/bin/env python
# hx

import os

################### 产生 train_val.txt #######################
# 产生 cls_trainval.txt 文件
class_sets = ('airplane','ship','storage_tank','baseball_diamond','tennis_court','basketball_court','ground_track_field','harbor','bridge','vehicle')

_dest_set_dir = './NWPU_VOC/ImageSets/Main'

for cls in class_sets:
    fstrain = open(os.path.join(_dest_set_dir, cls + '_' + 'train' + '.txt'), 'r')
    fsval = open(os.path.join(_dest_set_dir, cls + '_' + 'val' + '.txt'), 'r')
    fstrainval = open(os.path.join(_dest_set_dir, cls + '_' + 'train' + 'val' + '.txt'),'w')
    lines1 = fstrain.readlines()
    lines2 = fsval.readlines()
    lines1.extend(lines2)
    lines1.sort()
    fstrainval.writelines(lines1)
    fstrain.close()
    fsval.close()
    fstrainval.close()

# 产生 trainval.txt 文件
fstrain = open(os.path.join(_dest_set_dir,   'train' + '.txt'), 'r')
fsval = open(os.path.join(_dest_set_dir,  'val' + '.txt'), 'r')
fstrainval = open(os.path.join(_dest_set_dir,  'train' + 'val' + '.txt'),'w')
lines1 = fstrain.readlines()
lines2 = fsval.readlines()
lines1.extend(lines2)
lines1.sort()
fstrainval.writelines(lines1)
fstrain.close()
fsval.close()
fstrainval.close()
