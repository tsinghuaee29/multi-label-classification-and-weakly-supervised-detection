# --------------------------------------------------------
# PyTorch WSDDN
# Licensed under The MIT License [see LICENSE for details]
# Written by Seungkwan Lee
# Some parts of this implementation are based on code from Ross Girshick, Jiasen Lu, and Jianwei Yang
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
import torch
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.roi_pooling.modules.roi_pool import _RoIPooling
from utils.box_utils import *
import torchvision


class WSDDN_VGG16(nn.Module):
    '''
    (base): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace)
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): ReLU(inplace)
        (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): ReLU(inplace)
        (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (20): ReLU(inplace)
        (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (22): ReLU(inplace)
        (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (25): ReLU(inplace)
        (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (27): ReLU(inplace)
        (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (29): ReLU(inplace)
    )
    (top): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace)
        (2): Dropout(p=0.5)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace)
        (5): Dropout(p=0.5)
    )
    (fc8c): Linear(in_features=4096, out_features=20, bias=True)
    (fc8d): Linear(in_features=4096, out_features=20, bias=True)
    (roi_pooling): _RoIPooling()
    (roi_align): RoIAlignAvg()
    '''
    def __init__(self, pretrained_model_path=None, num_class=20):
        super(WSDDN_VGG16, self).__init__()
        vgg = torchvision.models.vgg16()
        if pretrained_model_path is None:
            print("Create WSDDN_VGG16 without pretrained weights")
        else:
            print("Loading pretrained VGG16 weights from %s" % (pretrained_model_path))
            state_dict = torch.load(pretrained_model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        self.base = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        self.top = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
        self.num_classes = num_class

        self.fc8c = nn.Linear(4096, self.num_classes)
        self.fc8d = nn.Linear(4096, self.num_classes)
        self.roi_pooling = _RoIPooling(7, 7, 1.0 / 16.0)
        self.roi_align = RoIAlignAvg(7, 7, 1.0 / 16.0)
        self.num_classes = self.num_classes
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.fc8c, 0, 0.01, False)
        normal_init(self.fc8d, 0, 0.01, False)

    def adjust_roi_offset(self, rois):
        rois = rois.clone()
        o0 = 8.5
        o1 = 9.5
        rois[:, 0] = torch.floor((rois[:, 0] - o0 + o1) / 16 + 0.5)
        rois[:, 1] = torch.floor((rois[:, 1] - o0 + o1) / 16 + 0.5)
        rois[:, 2] = torch.floor((rois[:, 2] - o0 - o1) / 16 - 0.5)
        rois[:, 3] = torch.floor((rois[:, 3] - o0 - o1) / 16 - 0.5)
        return rois

    def forward(self, im_data, rois, mode=0, prop_scores=None,image_level_label=None):
        #rois = self.adjust_roi_offset(rois)
        N = rois.size(0)
        feature_map = self.base(im_data)
        # print(feature_map.shape)
        zero_padded_rois = torch.cat([torch.zeros(N, 1).to(rois), rois], 1)
        pooled_feat = self.roi_pooling(feature_map, zero_padded_rois).view(N, -1)
        
        if mode == 1:
            pooled_feat = pooled_feat * (prop_scores.view(N, 1) * 10 + 1)
                
        fc7 = self.top(pooled_feat)
        fc8c = self.fc8c(fc7)
        fc8d = self.fc8d(fc7) / 2
        cls = F.softmax(fc8c, dim=1)
        det = F.softmax(fc8d, dim=0)

        if mode == 1:
            scores = cls * det
        elif mode == 2:
            prop = torch.diag(prop_scores)
            scores = cls * prop.mm(det)
        elif mode == 3:
            cls_var = torch.var(cls, 1)
            cls_var = torch.diag
            scores = cls_var.mm(cls) * det
        else:
            scores = cls * det

        if image_level_label is None:
            return scores

        image_level_scores = torch.sum(scores, 0)

        # To avoid numerical error
        image_level_scores = torch.clamp(image_level_scores, min=0, max=1)

        loss = F.binary_cross_entropy(image_level_scores, image_level_label.to(torch.float32), size_average=False)
        reg = self.spatial_regulariser(rois, fc7, scores, image_level_label)
        return scores, loss, reg, cls, det

    def region_aware_softmax(self, rois, det_score):
        N = rois.size(0)
        C = self.num_classes
        cwh_form_rois = to_cwh_form(rois)
        pair_wise_dx = cwh_form_rois[:, 0].view(1, -1) - cwh_form_rois[:, 0].view(-1, 1)
        pair_wise_dy = cwh_form_rois[:, 1].view(1, -1) - cwh_form_rois[:, 1].view(-1, 1)

        pair_wise_wsum = cwh_form_rois[:, 2].view(1, -1) + cwh_form_rois[:, 2].view(-1, 1)
        pair_wise_hsum = cwh_form_rois[:, 3].view(1, -1) + cwh_form_rois[:, 3].view(-1, 1)

        pair_wise_dx = pair_wise_dx / pair_wise_wsum
        pair_wise_dy = pair_wise_dy / pair_wise_hsum

        pair_wise_dist = torch.sqrt(pair_wise_dx * pair_wise_dx + pair_wise_dy * pair_wise_dy)
        pair_wise_weight = torch.exp(-pair_wise_dist)

        det_score = torch.exp(det_score)
        output = []

        for cls in range(self.num_classes):
            weighted_det_sum = torch.sum(det_score[:, cls] * pair_wise_weight, 1)
            here = det_score[:, cls] / weighted_det_sum
            output.append(here)

        output = torch.stack(output, 1)

        if output.max() < 0.001:
            det_score = torch.log(det_score)
            print(det_score)
            print(pair_wise_weight)
            print(det_score.max(), det_score.min())
            print(pair_wise_weight.max(), pair_wise_weight.min())
            print(pair_wise_dist.max(), pair_wise_dist.min())

        return output

    def spatial_regulariser(self, rois, fc7, scores, image_level_label):
        K = 10
        th = 0.6
        N = rois.size(0)
        ret = 0
        for cls in range(self.num_classes):
            if image_level_label[cls].item() == 0:
                continue

            topk_scores, topk_indices = scores[:, cls].topk(K, dim=0)
            topk_boxes = rois[topk_indices]
            topk_featres = fc7[topk_indices]
            

            mask = all_pair_iou(topk_boxes[0:1, :], topk_boxes).view(K).gt(th).float()

            diff = topk_featres - topk_featres[0]
            diff = diff * topk_scores.detach().view(K, 1)

            ret = (torch.pow(diff, 2).sum(1) * mask).sum() * 0.5 + ret

        return ret
