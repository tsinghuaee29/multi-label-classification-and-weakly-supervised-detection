# --------------------------------------------------------
# PyTorch WSDDN
# Licensed under The MIT License [see LICENSE for details]
# Written by Seungkwan Lee
# Some parts of this implementation are based on code from Ross Girshick, Jiasen Lu, and Jianwei Yang
# --------------------------------------------------------
import os
import numpy as np
import argparse
import time

import torch
import sklearn.metrics
from model.wsddn_vgg16 import WSDDN_VGG16
from datasets.wsddn_dataset import WSDDNDataset
from matplotlib import pyplot as plt
import torch.nn.functional as F
import math
import pickle
from cpu_nms import cpu_nms as nms
import heapq
import itertools
from frcnn_eval.pascal_voc import voc_eval_kit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

def parse_args():
    parser = argparse.ArgumentParser(description='Eval')
    parser.add_argument('--save_dir', help='directory to load model and save detection results', default="./data/repo")
    parser.add_argument('--data_dir', help='directory to load data', default='./data', type=str)

    parser.add_argument('--prop_method', help='ss or eb', default='eb', type=str)
    parser.add_argument('--use_prop_score', help='the mode to use prop_score', default=0, type=int)
    parser.add_argument('--multiscale', action='store_true')
    parser.add_argument('--min_resize', action='store_true')
    parser.add_argument('--thresh', help='threshold to select good boxes', default='20', type=int)

    parser.add_argument('--min_prop', help='minimum proposal box size', default=20, type=int)
    parser.add_argument('--model_name', default='WSDDN_VGG16_1_20', type=str)

    args = parser.parse_args()
    return args

args = parse_args()

def draw_box(boxes, col=None):
    for j, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        if col is None:
            c = np.random.rand(3)
        else:
            c = col
        plt.hlines(ymin, xmin, xmax, colors=c, lw=2)
        plt.hlines(ymax, xmin, xmax, colors=c, lw=2)
        plt.vlines(xmin, ymin, ymax, colors=c, lw=2)
        plt.vlines(xmax, ymin, ymax, colors=c, lw=2)


def eval():
    print('Called with args:')
    print(args)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    np.random.seed(3)
    torch.manual_seed(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(5)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    test_dataset = WSDDNDataset(dataset_names=['test'], data_dir=args.data_dir, num_classes=20)
    for data_loader in test_dataset._dataset_loaders:
        test_img_ids = [data['id'] for data in data_loader.items]
    load_name = os.path.join(args.save_dir, 'wsddn', '{}.pth'.format(args.model_name))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    if checkpoint['net'] == 'WSDDN_VGG16':
        model = WSDDN_VGG16(None, 20)
    else:
        raise Exception('network is not defined')
    model.load_state_dict(checkpoint['model'])
    print("loaded checkpoint %s" % (load_name))

    model.to(device)
    model.eval()

    start = time.time()

    num_images = len(test_dataset)
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS
    max_per_set = 40 * num_images
    # heuristic: keep at most 100 detection per class per image prior to NMS
    max_per_image = 100
    # detection thresold for each class (this is adaptively set based on the
    # max_per_set constraint)
    thresh = -np.inf * np.ones(20)
    # top_scores will hold one minheap of scores per class (used to enforce
    # the max_per_set constraint)
    top_scores = [[] for _ in range(20)]
    total_scores = []
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)] for _ in range(20)]
    anno_path = os.path.join('../PascalVOC/annotations.txt')
    _, cla_gt = load_gt(gt_file = anno_path)
     
    for index in range(len(test_dataset)):
        scores = 0
        if args.multiscale:
            comb = itertools.product([False, True], [480, 576, 688, 864, 1200])
        else:
            comb = itertools.product([False], [688])
        for h_flip, im_size in comb:
            im_data, gt_categories, proposals, prop_scores, image_level_label, im_scale, raw_img, im_id = \
                test_dataset.get_data(index, h_flip, im_size, args.min_resize)
            im_data = im_data.unsqueeze(0).to(device)
            rois = proposals.to(device)

            if args.use_prop_score == 1:
                prop_scores = prop_scores.to(device)
            else:
                prop_scores = None

            local_scores = model(im_data, rois, args.use_prop_score, prop_scores, None).detach().cpu().numpy()
            scores = scores + local_scores

        scores = scores * 1000
        boxes = test_dataset.get_raw_proposal(index)
        
        total_scores.append(np.sum(scores,(0)))
       
        for cls in range(20):
            inds = np.where((scores[:, cls] > thresh[cls]))[0]
            cls_scores = scores[inds, cls]
            cls_boxes = boxes[inds].copy()
            top_inds = np.argsort(-cls_scores)[:max_per_image]
            cls_scores = cls_scores[top_inds]
            cls_boxes = cls_boxes[top_inds, :]
            # push new scores onto the minheap
            for val in cls_scores:
                heapq.heappush(top_scores[cls], val)
            # if we've collected more than the max number of detection,
            # then pop items off the minheap and update the class threshold
            if len(top_scores[cls]) > max_per_set:
                while len(top_scores[cls]) > max_per_set:
                    heapq.heappop(top_scores[cls])
                thresh[cls] = top_scores[cls][0]

            all_boxes[cls][index] = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)

        if index % 100 == 99:
           print('%d images complete, elapsed time:%.1f' % (index + 1, time.time() - start))

    for j in range(20):
        for i in range(len(test_dataset)):
            inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
            all_boxes[j][i] = all_boxes[j][i][inds, :]
    
    save_dir0 = os.path.join(args.save_dir, 'detection_result')
    if not os.path.exists(save_dir0):
        os.makedirs(save_dir0)
    save_name1 = os.path.join(save_dir0, '{}.pkl'.format(args.model_name))
    save_name2 = os.path.join(save_dir0, '{}_score.pkl'.format(args.model_name))
    pickle.dump(all_boxes, open(save_name1, 'wb'))
    pickle.dump(total_scores,open(save_name2, 'wb'))
    print('Detection Complete, elapsed time: %.1f' % (time.time() - start))
    '''
    save_dir0 = os.path.join(args.save_dir, 'detection_result')
    if not os.path.exists(save_dir0):
        os.makedirs(save_dir0)
    save_name1 = os.path.join(save_dir0, '{}.pkl'.format(args.model_name))
    save_name2 = os.path.join(save_dir0, '{}_score.pkl'.format(args.model_name))
    all_boxes = pickle.load(open(save_name1, 'rb'))
    total_scores = pickle.load(open(save_name2, 'rb'))
    ''' 
    cla_result = [[] for index in range(len(test_dataset))]
    cla_score = [np.zeros(20) for _ in range(len(test_dataset))]
    for cls in range(20):
        for index in range(len(test_dataset)):
            dets = all_boxes[cls][index]
            if dets == []:
                continue
            # print(dets.shape)
            keep = nms(dets, 0.4, args.thresh)
            if keep:
                cla_result[index].append(cls)
            all_boxes[cls][index] = dets[keep, :].copy()
            cla_score[index][cls] = np.sum(dets[keep, -1], axis = 0)
    print('NMS complete, elapsed time: %.1f' % (time.time() - start))

    mAcc, wAcc, mAp = evaluate_classification(cla_result, cla_gt, cla_score)
    print('>>> mAcc: ', mAcc, ' wAcc: ', wAcc, 'mAp', mAp)
    
    eval_kit = voc_eval_kit('test', '2007_2008', os.path.join(args.data_dir))
    eval_kit.evaluate_detections(all_boxes, test_img_ids)
    
def load_gt(gt_file = '../PascalVOC/annotations.txt'):
    test_set = ['2007', '2008']
    train_set = ['2009', '2010', '2011', '2012']
    lines = open(gt_file, 'r').readlines()
    test_gt = []
    train_gt = []
    for line in lines:
        name = line.strip().split()[0]
        if name[:4] in test_set:
            if test_gt == None:
                test_gt = [map(int,line.strip().split()[1:])]
            else:
                test_gt.append(list(map(int,line.strip().split()[1:])))
        elif name[:4] in train_set:
            if train_gt == None:
                train_gt = [map(int,line.strip().split()[1:])]
            else:
                train_gt.append(list(map(int,line.strip().split()[1:])))
    return train_gt, test_gt

def evaluate_classification(cla_result, cla_gt, cla_scores = None):
    # list: cla_result[index] = [cla1,cla2,...]
    # list: cla_gt[index] =  [cla1,cla2,...]
    mlb = MultiLabelBinarizer(classes=[i for i in range(20)])
    cla_result = mlb.fit_transform(cla_result)
    cla_gt = mlb.fit_transform(cla_gt)
    Num_by_Class = np.sum(cla_gt, 0)

    Weight_by_Class = Num_by_Class / np.sum(Num_by_Class)
    # print('1',abs(cla_result - cla_gt))
    # print('2',np.sum(abs(cla_result - cla_gt), 0))
    # Error_by_Class = np.sum(abs(cla_result - cla_gt), 0) / Num_by_Class
    # Accuracy_by_Class = 1 - Error_by_Class
    #
    # print(Num_by_Class,Error_by_Class)
    Acc = []
    for cla in range(20):
        Acc.append(accuracy_score(cla_gt[:,cla],cla_result[:,cla]))
    mAcc = np.mean(Acc)
    wAcc = np.sum(Acc*Weight_by_Class)
    # mAcc = np.mean(Accuracy_by_Class)
    # wAcc = np.sum(Accuracy_by_Class * Weight_by_Class)
    map = 0
    if cla_scores != None:
        map = average_precision_score(cla_gt, cla_scores)
    return mAcc, wAcc, map

if __name__ == '__main__':
    eval()

