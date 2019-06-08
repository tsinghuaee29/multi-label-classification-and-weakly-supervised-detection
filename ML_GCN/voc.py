import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import pickle
import util
from util import *
import re

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']
label_convert = [14,2,7,9,11,12,16,0,1,3,5,6,13,18,4,8,10,15,17,19]
urls = {
    'devkit': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar',
    'trainval_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
    'test_images_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
    'test_anno_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar',
}
test_set = ['2007', '2008']
train_set = ['2009', '2010', '2011']

def read_object_categories_txt(file):
    categories_set = []
    index_set = []
    for line in open(file):
        s = re.split(' ', line)
        categories_set.append(s[1])
        index_set.append(s[0])
    return categories_set, index_set

def read_object_labels_txt(file, set = 'test',num_classes = 20,convert_label = True):
    images = []
    rownum = 0
    print('[dataset] read', file)
    for line in open(file):
        s = re.split(' ', line)
        year = s[0][0:4]
        if set == 'trainval':
            if not year in train_set:
                continue
        elif set == 'test':
            if not year in test_set:
                continue
        else:
            continue
        name = s[0]
        labels_idx = [int(s[n]) for n in range(1, len(s))]
        labels = np.asarray(-1*np.ones(num_classes))
        if convert_label:
            labels_idx = np.asarray(label_convert)[labels_idx]
        labels[labels_idx] = 1
        item = (name, labels)
        images.append(item)
        rownum += 1
    return images

class VocClassification(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None, inp_name=None, adj=None, Convert=True):
        self.root = root
        self.path_images = os.path.join(root, 'JPEGImages')
        self.set = set
        self.transform = transform
        self.target_transform = target_transform

        self.classes, self.classes_index = read_object_categories_txt(os.path.join(self.root,'categories.txt'))
        self.images = read_object_labels_txt(os.path.join(self.root,'annotations.txt'),set,num_classes = 20,convert_label = Convert)
        # word2vec
        if inp_name is not None:
            with open(inp_name, 'rb') as f:
                self.inp = pickle.load(f)
        else:
            self.inp = np.identity(20)

        self.inp_name = inp_name

        print('[dataset] Our VOC Data Set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, path, self.inp), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)
