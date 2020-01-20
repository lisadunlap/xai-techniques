import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torchvision import models, transforms
from skimage import io
import cv2
import os
from datetime import datetime
import gc
import torch.nn as nn
from mpl_toolkits.axes_grid1 import ImageGrid

#techniques
from Grad_CAM.main_gcam import gen_gcam, gen_gcam_target, gen_bp, gen_gbp, gen_bp_target, gen_gbp_target
from Integrated_Gradients.integrated_gradients import generate_ig_batch
from techniques.utils import get_displ_img, get_cam
from data_utils.data_setup import get_model_info, get_imagenet_classes
from LIME.LIME import generate_lime_explanation_batch
#from data_utils.gpu_memory import dump_tensors

''' Generate for dataloader
IMGS: list of tensors
MODEL: model name or model itself 
LABEL_NAME: name of class/unique name (for saving the image) 
FROM_SAVED: if using pytorch pretrained (so you pass in a string name for MODEL)
TARGET_INDEX: index of class, if TOPK=True, then it uses the predicted class
TOPK= if predict top index rather than specific index
LAYER: last convolutional layer of the network
DEVICE: cuda device number
CLASSES: list of class names
SAVE: if you want to save the result
SAVE_PATH: path to save to
SHOW: plot (for debugging)
'''
'''def gen_grounding_gcam_batch(imgs,
                  model='resnet18',
                  label_name='explanation',
                  from_saved=True, 
                  target_index=1,
                  layer='layer4', 
                  device=0,
                  topk=True,
                  classes=get_imagenet_classes(), 
                  save=True,
                  save_path='./results/gradcam_examples/', 
                  show=True):
    #CUDA_VISIBLE_DEVICES=str(device)
    # Create result directory if it doesn't exist; all explanations should 
    # be stored in a folder that is the predicted class
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H")

    save_path += label_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if save:
        print('result path: {0}'.format(save_path))

    if isinstance(model, str):
        model_name = model
        model, classes, target_layer = get_model_info(model, device=device)
    else:
        model_name = 'custom'
    
    # Generate the explanations
    if topk:
        masks = gen_gcam(imgs, model, target_index = target_index, target_layer=layer, device=device, single=False, prep=False, classes=classes)
    else:
        print('batch target indicies: ', target_index)
        masks = gen_gcam_target(imgs, model, target_index = target_index, target_layer=layer, device=device, single=False, prep=False, classes=classes)
    
    cams = []
    for mask, img in zip(masks, imgs):
        cams += [get_cam(img, mask)]
    
    if show:
        #plot heatmaps
        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(2, 2),
                         axes_pad=0.35,  # pad between axes in inch.
                         )

        for ax, im in zip(grid, cams[:4]):
            ax.axis('off')
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
            
    if save:
        for i in range(len(imgs)):
            print("saving explanation mask....\n")
            path = save_path + classes[target_index[i]] + '-'+str(i)+'_' + str(model_name)
            cv2.imwrite(path + '-original_img.png', get_displ_img(imgs[i]))

            if not cv2.imwrite(path +"cam.png", np.uint8(cams[i]*255)):
                print('error saving explanation')
            np.save(path + "gcam_mask.npy", masks[i])
        
    #just in case
    torch.cuda.empty_cache()

    return masks'''

def gen_grounding_gcam_batch(imgs,
                  model='resnet18',
                  label_name='explanation',
                  from_saved=True, 
                  target_index=1,
                  layer='layer4', 
                  device=0,
                  topk=True,
                  classes=get_imagenet_classes(), 
                  save=True,
                  save_path='./results/gradcam_examples/', 
                  show=True):
    #CUDA_VISIBLE_DEVICES=str(device)
    # Create result directory if it doesn't exist; all explanations should 
    # be stored in a folder that is the predicted class

    #save_path += label_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if save:
        print('result path: {0}'.format(save_path))

    if isinstance(model, str):
        model_name = model
        model, classes, target_layer = get_model_info(model, device=device)
    else:
        model_name = 'custom'
    
    # Generate the explanations
    if topk:
        masks = gen_gcam(imgs, model, target_index = target_index, target_layer=layer, device=device, single=False, prep=False, classes=classes)
    else:
        masks = gen_gcam_target(imgs, model, target_index = target_index, target_layer=layer, device=device, single=False, prep=False, classes=classes)
    
    cams = []
    for mask, img in zip(masks, imgs):
        cams += [get_cam(img, mask)]
    
    if show:
        #plot heatmaps
        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(2, 2),
                         axes_pad=0.35,  # pad between axes in inch.
                         )

        for ax, im in zip(grid, cams[:4]):
            ax.axis('off')
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
            
    if save:
        for i in range(len(imgs)):
            res_path = save_path + str(target_index[i].cpu().numpy()) + '/'
            if not os.path.exists(res_path):
                os.makedirs(res_path)
            #print("saving explanation mask....\n")
            cv2.imwrite(res_path + 'original_img.png', get_displ_img(imgs[i]))

            if not cv2.imwrite(res_path +"gcam.png", np.uint8(cams[i]*255)):
                print('error saving explanation')
            np.save(res_path + "gcam_mask.npy", masks[i])
        
    #just in case
    torch.cuda.empty_cache()

    return masks

'''def gen_grounding_bp_batch(imgs,
                  model='resnet18',
                  label_name='explanation',
                  from_saved=True, 
                  target_index=1,
                  layer='layer4', 
                  device=0,
                  topk=True,
                  classes=get_imagenet_classes(), 
                  save=True,
                  save_path='./results/gradcam_examples/', 
                  show=True):
    #CUDA_VISIBLE_DEVICES=str(device)
    # Create result directory if it doesn't exist; all explanations should 
    # be stored in a folder that is the predicted class
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H")

    save_path += label_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if save:
        print('result path: {0}'.format(save_path))

    if isinstance(model, str):
        model_name = model
        model, classes, target_layer = get_model_info(model, device=device)
    else:
        model_name = 'custom'
    
    # Generate the explanations
    #if topk:
    masks = gen_bp(imgs, model, target_index = target_index, target_layer=layer, device=device, single=False, prep=False, classes=classes)
    print('done ', len(masks))
    #else:
    #    print('batch target indicies: ', target_index)
    #    masks = gen_bp_target(imgs, model, target_index = target_index, target_layer=layer, device=device, single=False, prep=False, classes=classes)
    
    #cams = []
    #for mask, img in zip(masks, imgs):
    #    cams += [get_cam(img, mask)]
    
    if show:
        #plot heatmaps
        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(2, 2),
                         axes_pad=0.35,  # pad between axes in inch.
                         )

        for ax, im in zip(grid, cams[:4]):
            ax.axis('off')
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
            
    if save:
        for i in range(len(imgs)):
            print("saving explanation mask....\n")
            path = save_path + classes[target_index[i]] + '_' + str(model_name)
            cv2.imwrite(path + '-original_img.png', get_displ_img(imgs[i]))

            #if not cv2.imwrite(path +"cam.png", np.uint8(cams[i]*255)):
            #    print('error saving explanation')
            np.save(path +"bp_mask.npy", masks[i])
        
    #just in case
    torch.cuda.empty_cache()

    return masks'''

def gen_grounding_bp_batch(imgs,
                  model='resnet18',
                  label_name='explanation',
                  from_saved=True, 
                  target_index=1,
                  layer='layer4', 
                  device=0,
                  topk=True,
                  classes=get_imagenet_classes(), 
                  save=True,
                  save_path='./results/gradcam_examples/', 
                  show=True):
    #CUDA_VISIBLE_DEVICES=str(device)
    # Create result directory if it doesn't exist; all explanations should 
    # be stored in a folder that is the predicted class
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if save:
        print('result path: {0}'.format(save_path))

    if isinstance(model, str):
        model_name = model
        model, classes, target_layer = get_model_info(model, device=device)
    else:
        model_name = 'custom'
    
    # Generate the explanations
    if topk:
        masks = gen_bp(imgs, model, target_index = target_index, target_layer=layer, device=device, single=False, prep=False, classes=classes)
    else:
        masks = gen_bp_target(imgs, model, target_index = target_index, device=device, single=False, prep=False, classes=classes)
    
    cams = []
    for mask, img in zip(masks, imgs):
        cams += [get_cam(img, mask)]
    
    if show:
        #plot heatmaps
        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(2, 2),
                         axes_pad=0.35,  # pad between axes in inch.
                         )

        for ax, im in zip(grid, cams[:4]):
            ax.axis('off')
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
            
    if save:
        for i in range(len(imgs)):
            res_path = save_path + str(target_index[i].cpu().numpy()) + '/'
            if not os.path.exists(res_path):
                os.makedirs(res_path)
            #print("saving explanation mask....\n")
            cv2.imwrite(res_path + 'original_img.png', get_displ_img(imgs[i]))
            cv2.imwrite(res_path +"bp_mask.png", np.uint8(cams[i]*255))
            np.save(res_path +"bp_mask.npy", masks[i])
        
    #just in case
    torch.cuda.empty_cache()

    return masks

'''def gen_grounding_gbp_batch(imgs,
                  model='resnet18',
                  label_name='explanation',
                  from_saved=True, 
                  target_index=1,
                  layer='layer4', 
                  device=0,
                  topk=True,
                  classes=get_imagenet_classes(), 
                  save=True,
                  save_path='./results/gradcam_examples/', 
                  show=True):
    #CUDA_VISIBLE_DEVICES=str(device)
    # Create result directory if it doesn't exist; all explanations should 
    # be stored in a folder that is the predicted class
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H")

    save_path += label_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if save:
        print('result path: {0}'.format(save_path))

    if isinstance(model, str):
        model_name = model
        model, classes, target_layer = get_model_info(model, device=device)
    else:
        model_name = 'custom'
    
    # Generate the explanations
    #if topk:
    masks = gen_gbp(imgs, model, target_index = target_index, target_layer=layer, device=device, single=False, prep=False, classes=classes)
    #else:
    #    print('batch target indicies: ', target_index)
    #    masks = gen_gbp_target(imgs, model, target_index = target_index, target_layer=layer, device=device, single=False, prep=False, classes=classes)
    
    #cams = []
    #for mask, img in zip(masks, imgs):
    #    cams += [get_cam(img, mask)]
    
    if show:
        print('not implemented')
        #plot heatmaps
        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(2, 2),
                         axes_pad=0.35,  # pad between axes in inch.
                         )

        for ax, im in zip(grid, cams[:4]):
            ax.axis('off')
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
            
    if save:
        for i in range(len(imgs)):
            print("saving explanation mask....\n")
            path = save_path + classes[target_index[i]] + '_' + str(model_name)
            cv2.imwrite(path + '-original_img.png', get_displ_img(imgs[i]))

            #if not cv2.imwrite(path +"cam.png", np.uint8(cams[i]*255)):
            #    print('error saving explanation')
            np.save(path +"gbp_mask.npy", masks[i])
        
    #just in case
    torch.cuda.empty_cache()

    return masks'''

def gen_grounding_gbp_batch(imgs,
                  model='resnet18',
                  label_name='explanation',
                  from_saved=True, 
                  target_index=1,
                  layer='layer4', 
                  device=0,
                  topk=True,
                  classes=get_imagenet_classes(), 
                  save=True,
                  save_path='./results/gradcam_examples/', 
                  show=True):
    #CUDA_VISIBLE_DEVICES=str(device)
    # Create result directory if it doesn't exist; all explanations should 
    # be stored in a folder that is the predicted class

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if save:
        print('result path: {0}'.format(save_path))

    if isinstance(model, str):
        model_name = model
        model, classes, target_layer = get_model_info(model, device=device)
    else:
        model_name = 'custom'
    
    # Generate the explanations
    if topk:
        masks = gen_gbp(imgs, model, target_index = target_index, target_layer=layer, device=device, single=False, prep=False, classes=classes)
    else:
        masks = gen_gbp_target(imgs, model, target_index = target_index, device=device, single=False, prep=False, classes=classes)
        
    cams = []
    for mask, img in zip(masks, imgs):
        cams += [get_cam(img, mask)]
            
    if save:
        for i in range(len(imgs)):
            res_path = save_path + str(target_index[i].cpu().numpy()) + '/'
            if not os.path.exists(res_path):
                os.makedirs(res_path)
            #print("saving explanation mask....\n")
            cv2.imwrite(res_path + 'original_img.png', get_displ_img(imgs[i]))
            cv2.imwrite(res_path +"gbp_mask.png", np.uint8(cams[i]*255))
            np.save(res_path +"gbp_mask.npy", masks[i])
        
    #just in case
    torch.cuda.empty_cache()

    return masks

def gen_grounding_lime_batch(imgs,
                  model='resnet18',
                  label_name='explanation',
                  from_saved=True, 
                  target_index=1,
                  layer='layer4', 
                  device=0,
                  topk=True,
                  classes=get_imagenet_classes(), 
                  save=True,
                  save_path='./results/gradcam_examples/', 
                  show=True):
    #CUDA_VISIBLE_DEVICES=str(device)
    # Create result directory if it doesn't exist; all explanations should 
    # be stored in a folder that is the predicted class

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if save:
        print('result path: {0}'.format(save_path))

    if isinstance(model, str):
        model_name = model
        model, classes, target_layer = get_model_info(model, device=device)
    else:
        model_name = 'custom'
    
    # Generate the explanations
    masks = generate_lime_explanation_batch(imgs, model, pred_rank=1, positive_only=True, show=show, device='cuda:'+str(device))
    
    cams = []
    for mask, img in zip(masks, imgs):
        cams += [get_cam(img, mask)]
            
    if save:
        for i in range(len(imgs)):
            res_path = save_path + str(target_index[i].cpu().numpy()) + '/'
            if not os.path.exists(res_path):
                os.makedirs(res_path)
            #print("saving explanation mask....\n")
            cv2.imwrite(res_path + 'original_img.png', get_displ_img(imgs[i]))
            np.save(res_path +"lime_mask.npy", masks[i])
        
    #just in case
    torch.cuda.empty_cache()

    return masks

def gen_grounding_ig_batch(imgs,
                  model='resnet18',
                  label_name='explanation',
                  from_saved=True, 
                  target_index=1,
                  layer='layer4', 
                  device=0,
                  topk=True,
                  classes=get_imagenet_classes(), 
                  save=True,
                  save_path='./results/gradcam_examples/', 
                  show=True):
    #CUDA_VISIBLE_DEVICES=str(device)
    # Create result directory if it doesn't exist; all explanations should 
    # be stored in a folder that is the predicted class

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if save:
        print('result path: {0}'.format(save_path))

    if isinstance(model, str):
        model_name = model
        model, classes, target_layer = get_model_info(model, device=device)
    else:
        model_name = 'custom'
    
    # Generate the explanations
    model = model.to('cuda:0')
    if topk:
        masks = generate_ig_batch(imgs, model, cuda=True, show=True, reg=False, outlines=False, target_index=None)
    else:
        masks = generate_ig_batch(imgs, model, cuda=True, show=True, reg=False, outlines=False, target_index=[t for t in target_index.cpu().numpy()])
        
    cams = []
    for mask, img in zip(masks, imgs):
        cams += [get_cam(img, mask)]
            
    if save:
        for i in range(len(imgs)):
            res_path = save_path + str(target_index[i].cpu().numpy()) + '/'
            if not os.path.exists(res_path):
                os.makedirs(res_path)
            #print("saving explanation mask....\n")
            cv2.imwrite(res_path + 'original_img.png', get_displ_img(imgs[i]))
            cv2.imwrite(res_path +"ig_mask.png", np.uint8(cams[i]*255))
            np.save(res_path +"ig_mask.npy", masks[i])
        
    #just in case
    torch.cuda.empty_cache()

    return masks