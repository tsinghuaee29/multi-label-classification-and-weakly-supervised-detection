# PyTorch implementation of  ML_GCN & WSDDN

PyTorch implementation of

* [Multi-Label Image Recognition with Graph Convolutional Networks](https://arxiv.org/abs/1904.03582), CVPR 2019

* [Weakly Supervised Deep Detection Network](<https://arxiv.org/pdf/1511.02853.pdf>), CVPR 2016.



## Datasets

Download our PascalVOC dataset from [here](https://cloud.tsinghua.edu.cn/d/e53b38fbb9854d0792ff/)

and place it with the folder WSDDN and ML_GCN



## Pretrained models

### ML_GCN

Download the pretrained models for `ML_GCN` from [here](https://cloud.tsinghua.edu.cn/d/33e219ed6b5444d283dc/)

and place them in `\ML_GCN\checkpoint\`like this:

```
-checkpoint
	-2012voc
		-resnet
			-model_best.pth.tar
		-vgg
			-model_best.pth.tar
		-baseline
			-model_best.pth.tar
```



### WSDDN

Download the checkpoints and extra data for `WSDDN` from [here](https://cloud.tsinghua.edu.cn/d/371c68e3eb594cac85e5/)

and place them in `\WSDDN`like this:

```
-data
	-dataset
	-EdgeBoxesMat (you should unzip EdgeBoxesMat.zip first)
	-repo
```



## Training and Testing

```sh
cd ML_GCN
or
cd WSDDN
```

```sh
Then follow the instructions in the README.md under the two folders!
```



## Reference

This project is based on

* https://github.com/chenzhaomin123/ML_GCN 
* <https://github.com/jacobgil/pytorch-grad-cam>
* https://github.com/deneb2016/WSDDN.pytorch

