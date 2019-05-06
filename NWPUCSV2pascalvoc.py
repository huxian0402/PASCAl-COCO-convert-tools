# # convert NWPUcsv to COCO

# -*- coding:utf-8 -*-
# !/usr/bin/env python
# hx

import sys
import argparse
from xml.dom.minidom import Document
import cv2, os
import glob
import numpy as np

# csv 格式转为 voc 2007 格式，
# train和val相同,  test
# 对于train-val操作
# 生成XML文件（）
labels_10 = [
                'airplane',
                'ship',
                'storage_tank',
                'baseball_diamond',
                'tennis_court',
                'basketball_court',
                'ground_track_field',
                'harbor',
                'bridge',
                'vehicle' ]

def generate_xml(name, lines, img_size = (1066, 588, 1),
                 class_sets = ('airplane','ship','storage_tank','baseball_diamond','tennis_court','basketball_court','ground_track_field','harbor','bridge','vehicle'), doncateothers = True):
    """
    Write annotations into voc xml format.
    Examples:
        In: 0000001.txt
            cls        truncated    occlusion   angle   boxes                         3d annotation...
            Pedestrian 0.00         0           -0.20   712.40 143.00 810.73 307.92   1.89 0.48 1.20 1.84 1.47 8.41 0.01
        Out: 0000001.xml
            <annotation>
                <folder>VOC2007</folder>
	            <filename>000001.jpg</filename>
	            <source>
	            ...
	            <object>
                    <name>Pedestrian</name>
                    <pose>Left</pose>
                    <truncated>1</truncated>
                    <difficult>0</difficult>
                    <bndbox>
                        <xmin>x1</xmin>
                        <ymin>y1</ymin>
                        <xmax>x2</xmax>
                        <ymax>y2</ymax>
                    </bndbox>
            	</object>
            </annotation>
    :param name: stem name of an image, example: 0000001
    :param lines: lines in kitti annotation txt
    :param img_size: [height, width, channle]
    :param class_sets: ('Pedestrian', 'Car', 'Cyclist')
    :return:
    """

    doc = Document()  #document() 函数用于访问外部 XML 文档中的节点

    # 创建属性节点
    def append_xml_node_attr(child, parent = None, text = None):
        ele = doc.createElement(child)
        if not text is None:
            text_node = doc.createTextNode(text)
            ele.appendChild(text_node)
        parent = doc if parent is None else parent
        parent.appendChild(ele)
        return ele

    img_name = name+'.jpg'

    # create header
    annotation = append_xml_node_attr('annotation')
    append_xml_node_attr('folder', parent = annotation, text='VOC2007')       #'folder'     'VOC2007'
    append_xml_node_attr('filename', parent = annotation, text=img_name)      #'filename'   '000001'
    source = append_xml_node_attr('source', parent=annotation)                # 'source'
    append_xml_node_attr('database', parent=source, text='NWPU')        #'source'---'database'     'KITTI'
    append_xml_node_attr('annotation', parent=source, text='NWPU')      #'source'---'annotation'   'KITTI'
    append_xml_node_attr('image', parent=source, text='NWPU')           #'source'---'image'        'KITTI'
    append_xml_node_attr('flickrid', parent=source, text='000000')            #'source'---'flickrid'     '000000'
    owner = append_xml_node_attr('owner', parent=annotation)                  #'owner'
    append_xml_node_attr('url', parent=owner, text = 'hu xian')               #'owner'---'url' 'hu xian'
    size = append_xml_node_attr('size', annotation)                       #'size'
    append_xml_node_attr('width', size, str(img_size[1]))                 #'size'---'width'
    append_xml_node_attr('height', size, str(img_size[0]))                #'size'---'height'
    append_xml_node_attr('depth', size, str(img_size[2]))                 #'size'---'depth'
    append_xml_node_attr('segmented', parent=annotation, text='0')        #'segmented'---‘0’

    # create objects
    objs = []
    for line in lines:
        splitted_line = line.strip().split()
        cls = splitted_line[0]
        # if not doncateothers and cls not in class_sets:  # 如果不需要标记doncateothers，则跳过本次object
        #     continue
        ##解析 line 信息
        cls = 'dontcare' if cls not in class_sets else cls
        obj = append_xml_node_attr('object', parent=annotation)
        occlusion = int(float(splitted_line[2]))
        x1, y1, x2, y2 = int(float(splitted_line[4]) + 1), int(float(splitted_line[5]) + 1), \
                         int(float(splitted_line[6]) + 1), int(float(splitted_line[7]) + 1)
        truncation = float(splitted_line[1])
        difficult = 1 if _is_hard(cls, truncation, occlusion, x1, y1, x2, y2) else 0
        truncted = 0 if truncation < 0.5 else 1

        append_xml_node_attr('name', parent=obj, text=cls)       #name
        append_xml_node_attr('pose', parent=obj, text='Left')    #pose
        append_xml_node_attr('truncated', parent=obj, text=str(truncted))   #0
        append_xml_node_attr('difficult', parent=obj, text=str(int(difficult)))  #0
        bb = append_xml_node_attr('bndbox', parent=obj)          #bndbox
        append_xml_node_attr('xmin', parent=bb, text=str(x1))
        append_xml_node_attr('ymin', parent=bb, text=str(y1))
        append_xml_node_attr('xmax', parent=bb, text=str(x2))
        append_xml_node_attr('ymax', parent=bb, text=str(y2))

        o = {'class': cls, 'box': np.asarray([x1, y1, x2, y2], dtype=float), \
             'truncation': truncation, 'difficult': difficult, 'occlusion': occlusion}
        objs.append(o)

    return  doc, objs

def _is_hard(cls, truncation, occlusion, x1, y1, x2, y2):
    # Easy: Min. bounding box height: 40 Px, Max. occlusion level: Fully visible, Max. truncation: 15 %
    # Moderate: Min. bounding box height: 25 Px, Max. occlusion level: Partly occluded, Max. truncation: 30 %
    # Hard: Min. bounding box height: 25 Px, Max. occlusion level: Difficult to see, Max. truncation: 50 %
    hard = False
    if y2 - y1 < 25 and occlusion >= 2:
        hard = True
        return hard
    if occlusion >= 3:
        hard = True
        return hard
    if truncation > 0.8:
        hard = True
        return hard
    return hard

# 输入参数
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert NWPU  dataset into Pascal voc format')
    parser.add_argument('--NWPU', dest='NWPU',    # input kitti root path
                        help='path to ucas root',
                        default='./NWPU_CSV', type=str)
    parser.add_argument('--out', dest='outdir',     # output voc root path
                        help='path to voc-UCAS',
                        default='./NWPU_VOC', type=str)
    parser.add_argument('--draw', dest='draw',     # draw rect  flag
                        help='draw rects on images',
                        default=1, type=int)
    parser.add_argument('--dontcareothers', dest='dontcareothers',
                        help='ignore other categories, add them to dontcare rsgions',
                        default=1, type=int)      # ignore other categories flag

    if len(sys.argv) == 1:
        parser.print_help()
        # sys.exit(1)
    args = parser.parse_args()    # parse
    return args

def build_voc_dirs(outdir):
    """
    Build voc dir structure:
        VOC2007
            |-- Annotations
                    |-- ***.xml
            |-- ImageSets
                    |-- Layout
                            |-- [test|train|trainval|val].txt
                    |-- Main
                            |-- class_[test|train|trainval|val].txt
                    |-- Segmentation
                            |-- [test|train|trainval|val].txt
            |-- JPEGImages
                    |-- ***.jpg
            |-- SegmentationClass
                    [empty]
            |-- SegmentationObject
                    [empty]
    """
    mkdir = lambda dir: os.makedirs(dir) if not os.path.exists(dir) else None
    mkdir(outdir)
    mkdir(os.path.join(outdir, 'Annotations'))
    mkdir(os.path.join(outdir, 'ImageSets'))
    mkdir(os.path.join(outdir, 'ImageSets', 'Layout'))
    mkdir(os.path.join(outdir, 'ImageSets', 'Main'))
    mkdir(os.path.join(outdir, 'ImageSets', 'Segmentation'))
    mkdir(os.path.join(outdir, 'JPEGImages'))
    mkdir(os.path.join(outdir, 'SegmentationClass'))
    mkdir(os.path.join(outdir, 'SegmentationObject'))

    return os.path.join(outdir, 'Annotations'), os.path.join(outdir, 'JPEGImages'), os.path.join(outdir, 'ImageSets', 'Main')

def _draw_on_image(img, objs, class_sets_dict):
    colors = [(86, 0, 240), (173, 225, 61), (54, 137, 255),\
              (151, 0, 255), (243, 223, 48), (0, 117, 255),\
              (58, 184, 14), (86, 67, 140), (121, 82, 6),\
              (174, 29, 128), (115, 154, 81), (86, 255, 234)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for ind, obj in enumerate(objs):
        if obj['box'] is None: continue
        x1, y1, x2, y2 = obj['box'].astype(int)
        cls_id = class_sets_dict[obj['class']]
        if obj['class'] == 'dontcare':
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
            continue
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colors[cls_id % len(colors)], 1)
        text = '{:s}*|'.format(obj['class'][:3]) if obj['difficult'] == 1 else '{:s}|'.format(obj['class'][:3])
        text += '{:.1f}|'.format(obj['truncation'])
        text += str(obj['occlusion'])
        cv2.putText(img, text, (x1-2, y2-2), font, 0.5, (255, 0, 255), 1)
    return img

# 将科学计数法的字符串转化为数字
# hx
# '7.188994e+01'----71  , '71'--71
def stre2num(str):
    m=str.strip().split('e+')

    if len(m)==1:
        val = int(m[0])
    else:
        val= float(m[0]) * (10**int(m[1]))
        val = int(val)

    return val

# 处理lines, 转化为 ['airplane 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n']形式
def lines2lines_std(lines):
    lines_std = []
    for l in lines:
        temp = l.strip().split(',')
        # cls , x1, y1, x2, y2 = temp[0], temp[1], temp[2], temp[3], temp[4]
        cls, x1, y1, x2, y2 = temp[0], temp[2], temp[1], temp[4], temp[3]          # x is the longet side
        cls_name = labels_10[int(cls)]
        lines_temp =  cls_name + ' 0.00 0 -0.20 '+ x1 + ' '+ y1 +' '+ x2 +' '+ y2 + ' 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n'
        lines_std.append(lines_temp)

    return lines_std

# main
if __name__ == '__main__':
    args = parse_args()

    _ucasdir = args.NWPU
    _outdir = args.outdir
    _draw = bool(args.draw)
    _dest_label_dir, _dest_img_dir, _dest_set_dir = build_voc_dirs(_outdir)  #'./plane_UCASVOC/Annotations'   './plane_UCASVOC/JPEGImages'   './plane_UCASVOC/ImageSets/Main'
    _doncateothers = bool(args.dontcareothers)


    ######################   for training labels  #################################  1-600
    for dset in ['train']:    # for train

        _labeldir = os.path.join(_ucasdir,'train')
        _imagedir = os.path.join(_ucasdir,'train')

        class_sets = ['airplane','ship','storage_tank','baseball_diamond','tennis_court','basketball_court','ground_track_field','harbor','bridge','vehicle']
        class_sets_dict = {'airplane':0,'ship':1,'storage_tank':2,'baseball_diamond':3,'tennis_court':4,'basketball_court':5,'ground_track_field':6,
                            'harbor':7,'bridge':8,'vehicle':9}
        # class_sets_dict = {'airplane':1,'ship':2,'storage_tank':3,'baseball_diamond':4,'tennis_court':5,'basketball_court':6,'ground_track_field':7,
        #                     'harbor':8,'bridge':9,'vehicle':10}
        allclasses = {}
        fs = [open(os.path.join(_dest_set_dir, cls + '_' + dset + '.txt'), 'w') for cls in class_sets ]
        ftrain = open(os.path.join(_dest_set_dir, dset + '.txt'), 'w')

        files = glob.glob(os.path.join(_labeldir, '*.txt'))
        files.sort()         # 标签名升序排列
        for file in files:   #遍历每个标签  ， './Cars_/train'/0.txt'
            path, basename = os.path.split(file)    #'./Cars_/train'   '0.txt'
            stem, ext = os.path.splitext(basename)  #'0'  '.txt'
            stem_std = stem
            for i in range(6-len(stem)):            #去掉p，加上00..，仿造pascal
                stem_std = '0'+ stem_std

            with open(file, 'r') as f:
                lines = f.readlines()
                lines_std = lines2lines_std(lines)    # <class 'list'>: ['airplane 0.00 0 -0.20 248 118 319 172 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 242 186 332 259 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 242 283 336 340 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 246 362 340 428 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 254 457 333 516 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 534 98 628 165 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 535 340 640 428 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 554 589 632 650 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 699 88 780 141 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 695 225 782 295 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 702 306 790 366 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 709 438 786 503 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 717 507 796 569 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 971 100 1061 168 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 900 283 969 352 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 993 235 1080 301 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 987 303 1079 372 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 985 372 1068 440 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 988 443 1074 510 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 993 510 1090 577 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 971 573 1058 642 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 967 5 1046 68 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1147 102 1239 167 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1144 172 1234 227 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1142 236 1227 298 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1146 298 1233 369 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1156 370 1235 432 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1152 439 1237 503 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1166 508 1241 571 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1166 5 1240 60 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 541 1 602 51 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n']

            img_file = os.path.join(_imagedir, stem + '.jpg')   #读取该标签对应的图片
            img = cv2.imread(img_file)
            img_size = img.shape

            #                       ('000000',<class 'list'>: ['Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n'] ,
            #                                    (370, 1224, 3),  ('pedestrian', 'cyclist', 'car', 'dontcare') ,True )
            doc, objs = generate_xml(stem_std , lines_std , img_size, class_sets=class_sets, doncateothers= False)

            #if _draw:
            #    img=_draw_on_image(img, objs, class_sets_dict)
            #    plt.figure(figsize=(15, 15))
            #    plt.axis('off')
            #    plt.imshow(img)
            #    plt.show()
            #    #pause(1)
            
            # 写入图片和标签xml
            cv2.imwrite(os.path.join(_dest_img_dir, stem_std + '.jpg'), img)   #复制图片
            xmlfile = os.path.join(_dest_label_dir, stem_std + '.xml')
            with open(xmlfile, 'w') as f:                    #'./plane_UCASVOC/Annotations/000001.xml'
                f.write(doc.toprettyxml(indent='	'))

            ftrain.writelines(stem_std + '\n')    # train.txt

            # build [cls_train.txt]
            # airplane_train.txt: 0000xxx [1 | -1]
            cls_in_image = set([o['class'] for o in objs])   #该图片中所有的目标类别

            for obj in objs:   #对该图片中的所有目标
                cls = obj['class']
                allclasses[cls] = 0 \
                    if not cls in list(allclasses.keys()) else allclasses[cls] + 1

            for cls in cls_in_image:       # 区分正样本和负样本
                if cls in class_sets:
                    fs[class_sets_dict[cls]-1].writelines(stem_std + ' 1\n')
            for cls in class_sets:
                if cls not in cls_in_image:
                    fs[class_sets_dict[cls]-1].writelines(stem_std + ' -1\n')

            if int(stem_std) % 100 == 0:  # 进度条
                print(file)

        (f.close() for f in fs)
        ftrain.close()

        print('~~~~~~~~~~~~~~~~~~~')
        print(allclasses)
        print('~~~~~~~~~~~~~~~~~~~')

    ######################   for val labels  #################################  601-800
    for dset in ['val']:  # for val

        _labeldir = os.path.join(_ucasdir, 'val')
        _imagedir = os.path.join(_ucasdir, 'val')

        class_sets = ['airplane','ship','storage_tank','baseball_diamond','tennis_court','basketball_court','ground_track_field','harbor','bridge','vehicle']
        class_sets_dict = {'airplane':0,'ship':1,'storage_tank':2,'baseball_diamond':3,'tennis_court':4,'basketball_court':5,'ground_track_field':6,
                            'harbor':7,'bridge':8,'vehicle':9}

        allclasses = {}
        fs = [open(os.path.join(_dest_set_dir, cls + '_' + dset + '.txt'), 'w') for cls in class_sets ]
        ftval= open(os.path.join(_dest_set_dir, dset + '.txt'), 'w')

        files = glob.glob(os.path.join(_labeldir, '*.txt'))
        files.sort()         # 标签名升序排列
        for file in files:   #遍历每个标签  ， './Cars_/train'/0.txt'
            path, basename = os.path.split(file)    #'./Cars_/train'   '0.txt'
            stem, ext = os.path.splitext(basename)  #'0'  '.txt'
            stem_std = stem
            for i in range(6-len(stem)):            #去掉p，加上00..，仿造pascal
                stem_std = '0'+ stem_std

            with open(file, 'r') as f:
                lines = f.readlines()
                lines_std = lines2lines_std(lines)    # <class 'list'>: ['airplane 0.00 0 -0.20 248 118 319 172 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 242 186 332 259 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 242 283 336 340 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 246 362 340 428 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 254 457 333 516 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 534 98 628 165 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 535 340 640 428 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 554 589 632 650 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 699 88 780 141 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 695 225 782 295 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 702 306 790 366 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 709 438 786 503 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 717 507 796 569 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 971 100 1061 168 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 900 283 969 352 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 993 235 1080 301 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 987 303 1079 372 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 985 372 1068 440 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 988 443 1074 510 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 993 510 1090 577 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 971 573 1058 642 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 967 5 1046 68 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1147 102 1239 167 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1144 172 1234 227 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1142 236 1227 298 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1146 298 1233 369 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1156 370 1235 432 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1152 439 1237 503 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1166 508 1241 571 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1166 5 1240 60 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 541 1 602 51 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n']

            img_file = os.path.join(_imagedir, stem + '.jpg')   #读取该标签对应的图片
            img = cv2.imread(img_file)
            img_size = img.shape

            #                       ('000000',<class 'list'>: ['Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n'] ,
            #                                    (370, 1224, 3),  ('pedestrian', 'cyclist', 'car', 'dontcare') ,True )
            doc, objs = generate_xml(stem_std , lines_std , img_size, class_sets=class_sets, doncateothers= False)

            # 写入图片和标签xml
            cv2.imwrite(os.path.join(_dest_img_dir, stem_std + '.jpg'), img)  # 复制图片
            xmlfile = os.path.join(_dest_label_dir, stem_std + '.xml')
            with open(xmlfile, 'w') as f:
                f.write(doc.toprettyxml(indent='	'))

            ftval.writelines(stem_std + '\n')

            cls_in_image = set([o['class'] for o in objs])

            for obj in objs:  # 对该图片中的所有目标
                cls = obj['class']
                allclasses[cls] = 0 \
                    if not cls in list(allclasses.keys()) else allclasses[cls] + 1

            for cls in cls_in_image:       # 区分正样本和负样本
                if cls in class_sets:
                    fs[class_sets_dict[cls]-1].writelines(stem_std + ' 1\n')
            for cls in class_sets:
                if cls not in cls_in_image:
                    fs[class_sets_dict[cls]-1].writelines(stem_std + ' -1\n')

            if int(stem_std) % 100 == 0:  # 进度条
                print(file)

        (f.close() for f in fs)
        ftval.close()

        print('~~~~~~~~~~~~~~~~~~~')
        print(allclasses)
        print('~~~~~~~~~~~~~~~~~~~')

    # ######################   for test labels  #################################  801-900
    for dset in ['test']:  # for test

        _labeldir = os.path.join(_ucasdir, 'test')
        _imagedir = os.path.join(_ucasdir, 'test')

        class_sets = ['airplane','ship','storage_tank','baseball_diamond','tennis_court','basketball_court','ground_track_field','harbor','bridge','vehicle']
        class_sets_dict = {'airplane':0,'ship':1,'storage_tank':2,'baseball_diamond':3,'tennis_court':4,'basketball_court':5,'ground_track_field':6,
                            'harbor':7,'bridge':8,'vehicle':9}

        allclasses = {}
        fs = [open(os.path.join(_dest_set_dir, cls + '_' + dset + '.txt'), 'w') for cls in class_sets ]
        ftval= open(os.path.join(_dest_set_dir, dset + '.txt'), 'w')

        files = glob.glob(os.path.join(_labeldir, '*.txt'))
        files.sort()         # 标签名升序排列
        for file in files:   #遍历每个标签  ， './Cars_/train'/0.txt'
            path, basename = os.path.split(file)    #'./Cars_/train'   '0.txt'
            stem, ext = os.path.splitext(basename)  #'0'  '.txt'
            stem_std = stem
            for i in range(6-len(stem)):            #去掉p，加上00..，仿造pascal
                stem_std = '0'+ stem_std

            with open(file, 'r') as f:
                lines = f.readlines()
                lines_std = lines2lines_std(lines)    # <class 'list'>: ['airplane 0.00 0 -0.20 248 118 319 172 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 242 186 332 259 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 242 283 336 340 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 246 362 340 428 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 254 457 333 516 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 534 98 628 165 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 535 340 640 428 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 554 589 632 650 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 699 88 780 141 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 695 225 782 295 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 702 306 790 366 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 709 438 786 503 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 717 507 796 569 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 971 100 1061 168 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 900 283 969 352 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 993 235 1080 301 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 987 303 1079 372 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 985 372 1068 440 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 988 443 1074 510 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 993 510 1090 577 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 971 573 1058 642 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 967 5 1046 68 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1147 102 1239 167 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1144 172 1234 227 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1142 236 1227 298 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1146 298 1233 369 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1156 370 1235 432 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1152 439 1237 503 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1166 508 1241 571 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 1166 5 1240 60 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n', 'airplane 0.00 0 -0.20 541 1 602 51 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n']

            img_file = os.path.join(_imagedir, stem + '.jpg')   #读取该标签对应的图片
            img = cv2.imread(img_file)
            img_size = img.shape

            #                       ('000000',<class 'list'>: ['Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n'] ,
            #                                    (370, 1224, 3),  ('pedestrian', 'cyclist', 'car', 'dontcare') ,True )
            doc, objs = generate_xml(stem_std , lines_std , img_size, class_sets=class_sets, doncateothers= False)

            # 写入图片和标签xml
            cv2.imwrite(os.path.join(_dest_img_dir, stem_std + '.jpg'), img)  # 复制图片
            xmlfile = os.path.join(_dest_label_dir, stem_std + '.xml')
            with open(xmlfile, 'w') as f:
                f.write(doc.toprettyxml(indent='	'))

            ftval.writelines(stem_std + '\n')

            cls_in_image = set([o['class'] for o in objs])

            for obj in objs:  # 对该图片中的所有目标
                cls = obj['class']
                allclasses[cls] = 0 \
                    if not cls in list(allclasses.keys()) else allclasses[cls] + 1

            for cls in cls_in_image:       # 区分正样本和负样本
                if cls in class_sets:
                    fs[class_sets_dict[cls]-1].writelines(stem_std + ' 1\n')
            for cls in class_sets:
                if cls not in cls_in_image:
                    fs[class_sets_dict[cls]-1].writelines(stem_std + ' -1\n')

            if int(stem_std) % 100 == 0:  # 进度条
                print(file)

        (f.close() for f in fs)
        ftval.close()

        print('~~~~~~~~~~~~~~~~~~~')
        print(allclasses)
        print('~~~~~~~~~~~~~~~~~~~')