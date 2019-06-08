# ML_GCN.pytorch
PyTorch implementation of [Multi-Label Image Recognition with Graph Convolutional Networks](https://arxiv.org/abs/1904.03582), CVPR 2019.

### Requirements
Please, install the following packages
- numpy
- torch-0.3.1
- torchnet
- torchvision-0.2.0
- tqdm

### Options
- `data`: dataset path
- `image-size`: size of the image
- `workers`: number of data loading workers
- `epochs`: number of training epochs
- `epoch_step`: number of epochs to change learning rate
- `batch-size`: number of images per batch
- `lr`: learning rate
- `lrp`: factor for learning rate of pretrained layers. The learning rate of the pretrained layers is `lr * lrp`
- `evaluate`: evaluate model on validation set
- `resume`: path to checkpoint
- `which_model`: model for the backbone : resnet or vgg, classifier: fc or GCN
- `threshold`: Threshold for correlation matrix binarization 
- `C`: number of classes
- `Save_feature`: choose whether to save features while training

### Train 
```sh
python3 ML_GCN.py ../data --lr 0.02  --image-size 448 --batch-size 20 --epochs 100 --which_model resnet
```

### Test

```sh
python3 ML_GCN.py ../data  --image-size 448 --batch-size 10 --epochs 80  --resume checkpoint/2012voc/resnet/model_best.pth.tar --which_model resnet -e 
```



### Grad_CAM

#### Options

- `use-cuda`: Use NVIDIA GPU acceleration
- `image_dir`: image dataset path
- `annotation_dir`: path for annotation.txt
- `image-list`: list of input image name
- `inp`: path for word2vec dictionary
- `adf_file`: path for correlation matrix file
- `which_model`:model for the backbone : resnet or vgg, classifier: fc or GCN
- `resume`:path to checkpoint
- `save_gb`: save gb modeled image for grad-CAM

#### Demo

```sh
python grad_cam.py --pre_path checkpoint/2012voc/resnet/model_best.pth.tar --which_model resnet
```



## Reference
This project is based on https://github.com/chenzhaomin123/ML_GCN and <https://github.com/jacobgil/pytorch-grad-cam>