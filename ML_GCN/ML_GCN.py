import argparse
from engine import *
from models import *
from voc import *
import os

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('data', metavar='../PascalVOC/',
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=224, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-m', '--model', default='GCN',
                    help='model for the classifier to choose')
parser.add_argument('--gpu', default=0, type=int, help='GPU to run on')
parser.add_argument('--which_model', default='resnet101',
                    help='backbone: resnet101, vgg16 ; classifier: fc or GCN')
parser.add_argument('-t','--threshold',default=0.4,type=float,
                    help='Threshold for correlation metrix construction')
parser.add_argument('--C',default=20,type=int,help='Num of classes')
parser.add_argument('--Save_feature',default=False,type=bool,help='Whether to save features while training')

def main_ML_GCN():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()

    # define dataset
    train_dataset = VocClassification(args.data, 'trainval', inp_name='data/voc/voc_glove_word2vec.pkl',
                                          Convert=False)
    val_dataset = VocClassification(args.data, 'test', inp_name='data/voc/voc_glove_word2vec.pkl', Convert=False)

    num_classes = args.C

    # load model
    pre_path = os.path.join(args.data, 'dataset/pretrained_model/vgg16_caffe.pth')
    model = gcn_model(num_classes=num_classes, t=args.threshold, adj_file='data/voc/voc_adj2012.pkl',
                          pretrained_model_path=pre_path, which_model=args.which_model,
                          Save_file=args.Save_feature)
    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()
    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # pre-defined state for model training
    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes': num_classes, 'model': args.model}
    state['device_ids'] = [args.gpu]
    state['difficult_examples'] = True
    state['save_model_path'] = os.path.join('checkpoint/2012voc/',args.which_model)
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    if args.evaluate:
        state['evaluate'] = True
    # training module
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)


if __name__ == '__main__':
    main_ML_GCN()
