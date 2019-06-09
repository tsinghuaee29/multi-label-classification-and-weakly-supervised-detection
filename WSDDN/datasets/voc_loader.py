# --------------------------------------------------------
# PyTorch WSDDN
# Licensed under The MIT License [see LICENSE for details]
# Written by Seungkwan Lee
# Some parts of this implementation are based on code from Ross Girshick, Jiasen Lu, and Jianwei Yang
# --------------------------------------------------------
from scipy.misc import imread
from scipy.io import loadmat
import numpy as np
import re
import sys
import os

class VOCLoader:
    '''
    self.items = [{'id':图片编号,
                'categories':标签号,
                'im_path':图片路径,
                'proposal':预训练框[xmin, ymin, xmax, ymax]
                'prop_scores':预训练框的score},...]
    self.name_to_index = dict{'标签名'：标签号,...}
    '''
    def __init__(self, root, name, min_prop_scale=20):
        self.items = []
        self.name_to_index = {}
        for line in open(os.path.join('../PascalVOC/categories.txt')):
            s = re.split(' ', line)
            self.name_to_index[s[1]] = int(s[0])
        print('Our VOC dataset loading...')


        for line in open(os.path.join('../PascalVOC/annotations.txt')):
            data = {}
            s = re.split(' ', line)
            year = int(s[0][0:4])
            if name == 'train':
                if((year < 2009) | (year > 2012)):
                    continue
            elif name == 'test':
                if(year > 2008):
                    continue
            else:
                break
            id = s[0]
            category_set = [int(s[n]) for n in range(1,len(s))]

            data['id'] = id
            data['categories'] = np.array(category_set, np.long)
            data['img_path'] = os.path.join('../PascalVOC/JPEGImages', id + '.jpg')
            self.items.append(data)
        for img_dict in self.items:
            prop = loadmat(os.path.join(root, 'EdgeBoxesMat', img_dict['id'] + '.mat'))
            prop = prop['bbs']
            boxes = prop[:,:4].astype(np.float) - 1

            boxScores = prop[:,4]

            # # xmax, ymax, xmin, ymin
            # xmin(r), ymin(c), w, h
            is_good = (boxes[:, 2] >= min_prop_scale) * (boxes[:, 3] >= min_prop_scale)

            is_good = np.nonzero(is_good)[0]
            boxes = boxes[is_good]
            scores = boxScores[is_good]
            img_dict['proposals'] = np.concatenate([boxes[:, 0:1], boxes[:, 1:2], boxes[:, 0:1]+boxes[:, 2:3], boxes[:, 1:2]+boxes[:, 3:4]], 1)
            img_dict['prop_scores'] = scores
            
        print('Our VOC dataset loading complete')

    def __len__(self):
        return len(self.items)
if __name__ == "__main__":
    dataset = VOCLoader('WSDDN_cut','train')
