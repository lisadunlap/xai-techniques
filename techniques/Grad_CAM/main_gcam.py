from __future__ import print_function

import copy
import os.path as osp

import click
import cv2
import matplotlib.cm as cm
import sys
import numpy as np
import torch
import torch.hub
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models, transforms
from data_utils.data_setup import get_imagenet_classes

from Grad_CAM.gcam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False


def get_device(cuda, device):
    cuda = cuda and torch.cuda.is_available()
    cuda_dev = "cuda:"+str(device)
    device = torch.device(cuda_dev if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


'''def get_classtable():
    classes = []
    with open("../../samples/synset_words.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes'''


def preprocess(raw_image):
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def save_gradient(gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    return gradient


def save_gradcam(gcam):
    gcam = gcam.cpu().numpy()
    return gcam


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)

def gen_model_forward(imgs, model, device='cuda', prep=True, type='gcam'):
    """
    Visualize model responses given multiple images
    """

    # Model from torchvision
    model.to(device)
    model.eval()

    # Images
    images = []
    raw_images = []
    for i, im in enumerate(imgs):
        if prep:
            image, raw_image = preprocess(im)
        else:
            image = im
            raw_image = im.cpu().numpy().transpose((1,2,0))
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    if type == 'gcam':
        tech_model = GradCAM(model=model)
    elif type == 'bp':
        tech_model = BackPropagation(model=model)
    elif type == 'gbp':
        tech_model = GuidedBackPropagation(model=model)
    elif type == 'deconv':
        tech_model = Deconvnet(model=model)
    else:
        print('Invalid Type')
        sys.exit()

    probs, ids = tech_model.forward(images)
    return tech_model, probs, ids
    
def gen_gcam(imgs, model, target_layer='layer4', target_index=1, classes=get_imagenet_classes(), device='cuda', prep=True):
    """
    Visualize model responses given multiple images
    """

    # Model from torchvision
    model.to(device)
    model.eval()

    # Images
    images = []
    raw_images = []
    for i, im in enumerate(imgs):
        if prep:
            image, raw_image = preprocess(im)
        else:
            image = im
            raw_image = im.cpu().numpy().transpose((1,2,0))
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)

    for i in range(target_index):

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)
        masks = []
        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            # Grad-CAM
            mask = save_gradcam(
                gcam=regions[j, 0]
            )
            masks += [mask]
    if len(masks) == 1:
        return masks[0]
    gcam.remove_hook()
    return masks

def gen_gcam_target(imgs, model, target_layer='layer4', target_index=None, classes=get_imagenet_classes(), device='cuda', prep=True):
    """
    Visualize model responses given multiple images
    """

    # Model from torchvision
    model.to(device)
    model.eval()

    # Images
    images = []
    raw_images = []
    for i, im in enumerate(imgs):
        if prep:
            image, raw_image = preprocess(im)
        else:
            image = im
            raw_image = im.cpu().numpy().transpose((1,2,0))
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    ids_ = torch.LongTensor([[x] for x in target_index]).to(device)
    gcam.backward(ids=ids_)
    regions = gcam.generate(target_layer=target_layer)
    masks=[]
    for j in range(len(images)):
        print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

        mask = save_gradcam(
            gcam=regions[j, 0]
        )
        masks += [mask]

    if len(masks) == 1:
        return masks[0]
    return masks


def gen_bp(imgs, model, target_index=1, classes=get_imagenet_classes(), device='cuda', prep=True):
    """
    Visualize model responses given multiple images
    """

    # Model from torchvision
    model.to(device)
    model.eval()

    # Images
    images = []
    raw_images = []
    for i, im in enumerate(imgs):
        if prep:
            image, raw_image = preprocess(im)
        else:
            image = im
            raw_image = im.cpu().numpy().transpose((1,2,0))
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)

    for i in range(target_index):
        bp.backward(ids=ids[:, [i]])
        gradients = bp.generate()
        masks = []
        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            mask = save_gradient(
                gradient=gradients[j],
            )
            print(mask.shape)
            mask[mask < np.percentile(mask, 90)] = 0
            mask /= np.max(mask)
            masks += [mask[:,:,1]]
    if len(masks) == 1:
        return masks[0]
    bp.remove_hook()
    return masks

def gen_bp_target(imgs, model, target_index=None, classes=get_imagenet_classes(), device='cuda', prep=True):
    """
    Visualize model responses given multiple images
    """

    # Model from torchvision
    model.to(device)
    model.eval()

    # Images
    images = []
    raw_images = []
    for i, im in enumerate(imgs):
        if prep:
            image, raw_image = preprocess(im)
        else:
            image = im
            raw_image = im.cpu().numpy().transpose((1,2,0))
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)
    ids_ = torch.LongTensor([[x] for x in target_index]).to(device)
    bp.backward(ids=ids_)
    gradients = bp.generate()
    masks = []
    for j in range(len(images)):
        print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

        mask = save_gradient(
                gradient=gradients[j],
            )
        print(mask.shape)
        mask[mask < np.percentile(mask, 90)] = 0
        mask /= np.max(mask)
        masks += [mask[:,:,1]]
    if len(masks) == 1:
        return masks[0]
    return masks

def gen_gbp(imgs, model, target_index=1, classes=get_imagenet_classes(), device='cuda', prep=True):
    """
    Visualize model responses given multiple images
    """

    # Model from torchvision
    model.to(device)
    model.eval()

    # Images
    images = []
    raw_images = []
    for i, im in enumerate(imgs):
        if prep:
            image, raw_image = preprocess(im)
        else:
            image = im
            raw_image = im.cpu().numpy().transpose((1,2,0))
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    gbp = GuidedBackPropagation(model=model)
    probs, ids = gbp.forward(images)

    for i in range(target_index):
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()
        masks = []
        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            mask = save_gradient(
                gradient=gradients[j],
            )
            print(mask.shape)
            mask[mask < np.percentile(mask, 90)] = 0
            mask /= np.max(mask)
            masks += [mask[:,:,1]]
    if len(masks) == 1:
        return masks[0]
    gbp.remove_hook()
    return masks

def gen_gbp_target(imgs, model, target_index=None, classes=get_imagenet_classes(), device='cuda', prep=True):
    """
    Visualize model responses given multiple images
    """

    # Model from torchvision
    model.to(device)
    model.eval()

    # Images
    images = []
    raw_images = []
    for i, im in enumerate(imgs):
        if prep:
            image, raw_image = preprocess(im)
        else:
            image = im
            raw_image = im.cpu().numpy().transpose((1,2,0))
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    gbp = GuidedBackPropagation(model=model)
    probs, ids = gbp.forward(images)
    ids_ = torch.LongTensor([[x] for x in target_index]).to(device)
    gbp.backward(ids=ids_)
    gradients = gbp.generate()
    masks = []
    for j in range(len(images)):
        print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

        mask = save_gradient(
                gradient=gradients[j],
            )
        mask[mask < np.percentile(mask, 90)] = 0
        mask /= np.max(mask)
        masks += [mask[:,:,1]]
    if len(masks) == 1:
        return masks[0]
    return masks


def gen_deconv(imgs, model, target_index=1, classes=get_imagenet_classes(), device='cuda', prep=True):
    """
    Visualize model responses given multiple images
    """

    # Model from torchvision
    model.to(device)
    model.eval()

    # Images
    images = []
    raw_images = []
    for i, im in enumerate(imgs):
        if prep:
            image, raw_image = preprocess(im)
        else:
            image = im
            raw_image = im.cpu().numpy().transpose((1,2,0))
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    deconv = Deconvnet(model=model)
    probs, ids = deconv.forward(images)

    for i in range(target_index):
        deconv.backward(ids=ids[:, [i]])
        gradients = deconv.generate()
        masks = []
        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            mask = save_gradient(
                gradient=gradients[j],
            )

            # For Display Purposes: only keep top 10% of values
            # this makes the gradient more targeted and less "static-y"
            mask[mask < np.percentile(mask, 90)] = 0
            mask /= np.max(mask)
            masks += [mask[:,:,1]]
    if len(masks) == 1:
        return masks[0]
    deconv.remove_hook()
    return masks

def gen_deconv_target(imgs, model, target_index=None, classes=get_imagenet_classes(), device='cuda', prep=True):
    """
    Visualize model responses given multiple images
    """

    # Model from torchvision
    model.to(device)
    model.eval()

    # Images
    images = []
    raw_images = []
    for i, im in enumerate(imgs):
        if prep:
            image, raw_image = preprocess(im)
        else:
            image = im
            raw_image = im.cpu().numpy().transpose((1,2,0))
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    deconv = Deconvnet(model=model)
    probs, ids = deconv.forward(images)

    ids_ = torch.LongTensor([[x] for x in target_index]).to(device)
    deconv.backward(ids=ids_)
    gradients = deconv.generate()
    masks = []
    for j in range(len(images)):
        print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

        mask = save_gradient(
                gradient=gradients[j],
            )

        mask[mask < np.percentile(mask, 90)] = 0
        mask /= np.max(mask)
        masks += [mask[:,:,1]]
    if len(masks) == 1:
        return masks[0]
    return masks