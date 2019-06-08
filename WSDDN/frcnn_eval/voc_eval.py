# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
#
# Modified by Seungkwan Lee for WSDDN
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import re
import pickle
import numpy as np

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             classname,
             ovthresh=0.5):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                classname,
                                [ovthresh],
                                )

    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
    annopath: Path to annotations
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    npos = 0
    gt_boxes = {}
    for line in open(os.path.join(annopath, 'PascalVOC/bonus_ground_truth.txt')):
        s = re.split(' ', line)
        cls = int(s[1])
        if (cls != classname):
            continue
        if s[0] in gt_boxes.keys():
            boxes = np.array(s[2:]).astype(np.float)
            boxes = np.array([boxes[0], boxes[1], boxes[0] + boxes[2], boxes[1] + boxes[3]])
            gt_boxes[s[0]]['bbox'].append(boxes)
            gt_boxes[s[0]]['det'].append(False)
        else:
            gt_boxes[s[0]] = {}
            gt_boxes[s[0]]['bbox'] = []
            gt_boxes[s[0]]['det'] = []
            boxes = np.array(s[2:]).astype(np.float)
            boxes = np.array([boxes[0], boxes[1], boxes[0] + boxes[2], boxes[1] + boxes[3]])
            gt_boxes[s[0]]['bbox'].append(boxes)
            gt_boxes[s[0]]['det'].append(False)
        npos = npos + 1
        

    # read dets
    # detfile = detpath.format(classname)
    detfile = detpath
    # print(detpath)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    # print(BB, '@@@',confidence)
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        if image_ids[d] not in gt_boxes.keys():
            fp[d] = 1.
            continue
        R = gt_boxes[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = np.array(R['bbox']).astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, True)

    return rec, prec, ap
