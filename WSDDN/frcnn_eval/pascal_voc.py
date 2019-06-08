# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
#
# Modified by Seungkwan Lee for WSDDN
# --------------------------------------------------------

import os
from frcnn_eval.imdb import imdb
import numpy as np
import re
from frcnn_eval.voc_eval import voc_eval
import uuid

class voc_eval_kit(imdb):
    def __init__(self, image_set, year, root):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self.path = root
        self._class_to_ind = {}
        for line in open(os.path.join(root, 'PascalVOC/categories.txt')):
            s = re.split(' ', line)
            self._class_to_ind[s[1]] = int(s[0])
        self._image_ext = '.jpg'
        # self._image_index = self._load_image_set_index()
        # self._salt = str(uuid.uuid4())


        assert os.path.exists(self.path), 'VOCdevkit path does not exist: {}'.format(self.path)

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = 'det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self.path,
            'repo_cut',
            'results')
        if not os.path.exists(path):
            os.makedirs(path)
        return os.path.join(path, filename)

    def _write_voc_results_file(self, all_boxes, test_img_ids):
        for cls, cls_ind in self._class_to_ind.items():
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, im_id in enumerate(test_img_ids):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(im_id, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self):
        aps = []
        annopath = self.path
        for cls, cls_ind in self._class_to_ind.items():
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, cls_ind, ovthresh=0.5)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, all_boxes, test_img_ids):
        self._write_voc_results_file(all_boxes, test_img_ids)
        self._do_python_eval()

