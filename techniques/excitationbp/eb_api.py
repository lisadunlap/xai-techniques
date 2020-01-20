import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from torchvision import datasets, models, transforms, utils

import numpy as np
import matplotlib.pyplot as plt

import os, sys, copy ; sys.path.append('..')
import excitationbp as eb

reseed = lambda: np.random.seed(seed=1) ; ms = torch.manual_seed(1) # for reproducibility
reseed()
from rise_utils import *

def get_model(model_name, other_layer = None):
    CONFIG = {
        'resnet152': {
            'target_layer': 'layer4.2',
            'input_size': 224
        },
        'vgg19': {
            'target_layer': 'features.36',
            'input_size': 224
        },
        'vgg19_bn': {
            'target_layer': 'features.52',
            'input_size': 224
        },
        'inception_v3': {
            'target_layer': 'Mixed_7c',
            'input_size': 299
        },
        'densenet201': {
            'target_layer': 'features.denseblock4',
            'input_size': 224
        },
        'resnet18': {
            'target_layer': 'layer4.1',
            'input_size': 224
        },
    }.get(model_name)
    model = models.__dict__[model_name](pretrained=True)
    if other_layer:
        target_layer = other_layer
    else:
        target_layer = CONFIG['target_layer']
    return model

def gen_eb(path, model_name, show=True):
    
    # get model
    model = models.__dict__[model_name](pretrained=True)
    #model = models.vgg19(pretrained=True)
    _ = model.train(False) # put model in evaluation mode
    
    # get imagenet class labels from local file
    class_labels = []
    with open('techniques/excitationbp/data/imagenet/labels.txt', 'r') as f:
        for line in f:
            class_labels.append(line[:-1])

    eb.use_eb(True)
    inputs = read_tensor(path)
    inputs = Variable(inputs.resize_(1,3,224,224), requires_grad=True)
    y_hat = model(inputs)
    class_id = y_hat.max(1)[1].data.numpy()[0]

    # compute excitation backprop for correct label
    prob_outputs_cat = Variable(torch.zeros(1,1000)) ; prob_outputs_cat.data[:,class_id] += 1

    prob_inputs_cat = eb.excitation_backprop(model, inputs, prob_outputs_cat, contrastive=False)


    beta = 0.5 # adjust the 'peakiness' of data
    cat_img = prob_inputs_cat.clamp(min=0).sum(0).sum(0).data.numpy()
    print(cat_img.shape)

    if show:
        plt.imshow(cat_img**beta, cmap='gray')
    return cat_img