import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False,dropout_n = 0.2):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.dropout_n = dropout_n
        self.dropout = nn.Dropout(self.dropout_n) 
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class ML_GCN(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0, adj_file=None,which_model='resnet101',Save_file=False):
        super(ML_GCN, self).__init__()
        if which_model == 'vgg16':
            # CNN model for vgg_GCN 
            self.features = nn.Sequential(*list(model.features._modules.values()))
            self.expand = nn.Sequential(
                nn.Linear(512, 2048),
                nn.LeakyReLU(0.2), )
        else:
            # CNN model for ResNet_GCN
            self.features = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            )
        if which_model == 'baseline':
            # direct full-connect layer for ResNet(baseline)
            self.basefc = nn.Linear(2048, num_classes)

        self.Save_file = Save_file
        self.which_model = which_model
        self.num_classes = num_classes

        self.pooling = nn.MaxPool2d(14, 14)
        self.pooling_vgg = nn.MaxPool2d(28,28)
        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax()
        # GCN definition
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024,2048)
        # Get correlation metrix
        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        self.feature_num = 0
        self.feature_max = 1000


    def forward(self, feature, inp):
        feature = self.features(feature)

        if self.which_model == 'vgg16':
            feature = self.pooling(feature)
            feature = feature.view(feature.size(0), -1)
            feature = self.expand(feature)
        else:
            feature = self.pooling(feature)
            feature = feature.view(feature.size(0), -1)


        if self.Save_file==True and self.feature_num < self.feature_max:
            self.feature_num = save_feature(feature,self.feature_max)

        if self.which_model == 'baseline':
            x = self.basefc(feature)
            return x

        inp = inp[0]
        adj = gen_adj(self.A).detach()
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x

    def classifier(self, feature,inp):
        if self.which_model == 'vgg16':
            feature = self.expand(feature)
        if not torch.is_tensor(inp):
            inp = torch.from_numpy(inp)
        adj = gen_adj(self.A).detach()
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x

    def classifier_zero_grad(self):
        if self.which_model == 'vgg16':
            self.expand.zero_grad()
        self.gc1.zero_grad()
        self.gc2.zero_grad()

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]


def save_feature(feature_batch,max_num=1000):
    import json
    import os
    file = 'features.json'
    if not os.path.exists(file):
        features = []
    else:
        with open(file,'r') as f:
            features = json.load(f)
    c = len(features)
    if c == 0:
        features = []
    elif c >= max_num:
        return c
    feature_batch = feature_batch.tolist()
    for feat in feature_batch:
        features.append(feat)
    #print('save!')
    c += len(feature_batch)
    print('feature saved num:', c)
    with open(file, 'w') as f:
        json.dump(features,f)

    return c

def gcn_model(num_classes, t, pretrained=True,pretrained_model_path = None, adj_file=None, in_channel=300, which_model='resnet101',Save_file=False):
    if which_model == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        print("Loading pretrained VGG16 weights from %s" % (pretrained_model_path))
    else:
        model = models.resnet101(pretrained=pretrained)
        print("Loading pretrained res101 models")
    return ML_GCN(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel,which_model=which_model,Save_file=Save_file)
