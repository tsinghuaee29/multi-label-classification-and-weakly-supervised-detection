import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
import pickle
import os
from models import *

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x,inp):
		target_activations, output  = self.feature_extractor(x)
		output = torch.nn.functional.max_pool2d(output, kernel_size=(14,14))
		output = output.view(output.size(0), -1)
		output = self.model.classifier(output,inp)
		return target_activations, output

def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = Variable(preprocessed_img, requires_grad = True)
	return input

def show_cam_on_image(img, mask,img_name,label,model_name):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	cv2.imwrite("analysis/all/"+model_name+'_'+img_name+'_'+str(label)+".jpg", np.uint8(255 * cam))

class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()
		self.cnt = 0
		self.extractor = ModelOutputs(self.model, target_layer_names)

	def forward(self, input):
		return self.model(input) 

	def __call__(self, input, inp, index = None):
		if self.cuda:
			features, output = self.extractor(input.cuda(),inp)
		else:
			features, output = self.extractor(input,inp)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.features.zero_grad()
		self.model.classifier_zero_grad()
		one_hot.backward(retain_graph=True)
		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

		target = features[-1]
		target = target.cpu().data.numpy()[0, :]

		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		cam = np.zeros(target.shape[1 : ], dtype = np.float32)

		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (448, 448))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam

class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
	def __init__(self, model, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		# replace ReLU with GuidedBackpropReLU
		for idx, module in self.model.features._modules.items():
			if module.__class__.__name__ == 'ReLU':
				self.model.features._modules[idx] = GuidedBackpropReLU()

	def forward(self, input,inp):
		if not torch.is_tensor(inp):
			inp = torch.from_numpy(inp)
		return self.model(input,[inp])

	def __call__(self, input, index = None, inp=None):
		if self.cuda:
			output = self.forward(input.cuda(),inp)
		else:
			output = self.forward(input,inp)
		
		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		one_hot.backward(retain_graph=True)

		output = input.grad.cpu().data.numpy()
		output = output[0,:,:,:]
		return output

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--use-cuda', action='store_true', default=False,
	                    help='Use NVIDIA GPU acceleration')
	parser.add_argument('--image_dir',type=str,default='../JPEGImages')
	parser.add_argument('--annotation_dir',type=str,default='../annotations.txt')
	parser.add_argument('--image-list', type=str, default='analysis/image_list.txt',
	                    help='Input image name')
	parser.add_argument('--inp',type=str,default='data/voc/voc_glove_word2vec.pkl')
	parser.add_argument('--adj_file',type=str, default='data/voc/voc_adj2012.pkl',
						help='Adjacent matrix path')
	parser.add_argument('--pre_path',type=str, default='None',
						help='pretrained model for backbone: vgg16 or resnet101')
	parser.add_argument('--which_model', default='resnet101',
                    	help='backbone: resnet101, vgg16 ; classifier: fc or GCN')
	parser.add_argument('--resume', default='None', type=str, metavar='PATH',
                    	help='path to latest checkpoint (default: none)')
	parser.add_argument('--save_gb',default=False,type=bool, help='Whether to save gb model for grad-CAM')
	args = parser.parse_args()
	args.use_cuda = args.use_cuda and torch.cuda.is_available()
	if args.use_cuda:
	    print("Using GPU for acceleration")
	else:
	    print("Using CPU for computation")

	return args

if __name__ == '__main__':
	""" python grad_cam.py <path_to_image>
	1. Loads an image with opencv.
	2. Preprocesses it for VGG19 and converts to a pytorch variable.
	3. Makes a forward pass to find the category index with the highest score,
	and computes intermediate activations.
	Makes the visualization. """

	args = get_args()

	num_classes = 20

	# Can work with any model, but it assumes that the model has a 
	# feature method, and a classifier method,
	# as in the VGG models in torchvision.

	# appointed layer
	if args.which_model == 'vgg16':
		layers = ["28"]
	else:
		layers = ["7"]

	if args.inp is not None:
		with open(args.inp, 'rb') as f:
			inp = pickle.load(f)
	else:
		inp = np.identity(20)
	if not torch.is_tensor(inp):
                inp = torch.from_numpy(inp)


	# Read annotations
	Labels = {}
	with open(args.annotation_dir,'r') as File:
		for line in File:
			line_s = line.strip().split()
			Labels[line_s[0]] = []
			for label in line_s[1:]:
				Labels[line_s[0]].append(int(label))
	image_list = []
	with open(args.image_list,'r') as File:
		for line in File:
			image_list.append(line.strip())

	for img_name in image_list:
		image_path = os.path.join(args.image_dir,img_name+'.jpg')
		img = cv2.imread(image_path, 1)
		img = np.float32(cv2.resize(img, (448, 448))) / 255
		input = preprocess_image(img)
		# If None, returns the map for the highest scoring category.
		# Otherwise, targets the requested index.
		for target_index in Labels[img_name]: 
			model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file=args.adj_file,
                          pretrained_model_path=args.pre_path, which_model=args.which_model)
			if args.resume != 'None':
				print("=> loading checkpoint '{}'".format(args.resume))
				checkpoint = torch.load(args.resume)
				model.load_state_dict(checkpoint['state_dict'])
			grad_cam = GradCam(model = model, \
					target_layer_names = layers, use_cuda=args.use_cuda)
			mask = grad_cam(input, inp, target_index)
			model_name = args.which_model
			show_cam_on_image(img, mask,img_name,target_index,model_name)
			if args.save_gb :
				gb_model = GuidedBackpropReLUModel(model = model, use_cuda=args.use_cuda)
				gb = gb_model(input, index=target_index,inp = inp)
				utils.save_image(torch.from_numpy(gb), 'analysis/gb'+img_name+str(target_index)+'.jpg')

				cam_mask = np.zeros(gb.shape)
				for i in range(0, gb.shape[0]):
				    cam_mask[i, :, :] = mask

				cam_gb = np.multiply(cam_mask, gb)
				utils.save_image(torch.from_numpy(cam_gb), 'analysis/cam_gb'+img_name+str(target_index)+'.jpg')
