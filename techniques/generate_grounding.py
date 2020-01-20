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
import matplotlib.cm as cm
import sys

#techniques
#from Grad_CAM.grad_cam import gen_gcam
#from Grad_CAM.old_grad_cam import old_gen_gcam, get_guidedBackProp_img
from Grad_CAM.main_gcam import gen_gcam, gen_gcam_target, gen_bp, gen_gbp, gen_bp_target, gen_gbp_target
from Integrated_Gradients.integrated_gradients import generate_ig
from LIME.LIME import generate_lime_explanation
from RISE.rise_utils import gen_rise_grounding
from utils import get_model,get_displ_img
from data_utils.data_setup import get_model_info, get_imagenet_classes
from data_utils.gpu_memory import dump_tensors

#if torch.cuda.is_available():
#    torch.cuda.set_device(CUDA_VISIBLE_DEVICES)
sv_pth = './results/master_examples/'

# Put heatmap on image
def get_cam(img, mask):
    w, h, _ = img.shape
    mask = cv2.resize(mask, (w, h))
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8((mask / np.max(mask)) * 255.0), cv2.COLORMAP_JET),
                           cv2.COLOR_BGR2RGB)
    alpha = .7
    cam = heatmap*alpha + np.float32(img)*(1-alpha)
    cam /= np.max(cam)
    return cam 

''' Generate for dataloader
IMG: img array (0-255)
TECHNIQUE: explanation technique
LABEL_NAME: name of class/unique name (for saving the image) 
MODEL: model name or model itself 
FROM_SAVED: if using pytorch pretrained (so you pass in a string name for MODEL)
TARGET_INDEX: index of class, if TOPK=True, then it uses the predicted class
TOPK= if predict top index rather than specific index
LAYER: last convolutional layer of the network
DEVICE: cuda device number
CLASSES: list of class names
SAVE: if you want to save the result
SAVE_PATH: path to save to
SHOW: plot (for debugging)
INDEX: whether to use a specific class index (if false then just gets
    the topk prediction according to TARGET_INDEX)
    right now LIME and RISE do not take specific indecies
'''
def gen_grounding(img,
                  technique,
                  label_name='explanation',
                  model='resnet18',
                  show=False,
                  layer='layer4',
                  save_path='./results/master_examples/',
                  target_index=1,
                  save=True,
                  device=0,
                  index=False):
    # Create result directory if it doesn't exist; all explanations should 
    # be stored in a folder that is the predicted class

    if save:
        save_path += label_name + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print('result path: {0}'.format(save_path))
        
    # convert image if needed
    if np.max(img) < 2:
        img = np.uint8(img*255)
    if isinstance(model, str):
        model_name = model
        model, classes, layer = get_model_info(model, device=device)
    else:
        model_name = 'custom'
    model.eval()
    
    device = 'cuda:'+str(device)
    if not torch.cuda.is_available():
        device = 'cpu'
    
    # Generate the explanations
    if technique == 'lime' or technique == 'LIME':
        if not index:
            mask = generate_lime_explanation(img, model, pred_rank=target_index, positive_only=True, device=device)
        else:
            mask = generate_lime_explanation(img, model, pred_rank=target_index, target_index=target_index, positive_only=True, device=device)
    elif technique == 'gradcam' or technique == 'GradCam' or technique == 'gcam':
        if not index:
            mask = gen_gcam([img], model, target_index = target_index, show_labels=True, target_layer=layer)
        else:
            mask = gen_gcam_target([img], model, target_index = [target_index], target_layer=layer)
    elif technique == 'backprop' or technique == 'bp':
        if not index:
            mask = gen_bp([img], model, target_index = target_index, show_labels=True, target_layer=layer)
        else:
            mask = gen_bp_target([img], model, target_index = [target_index])
    elif technique == 'guided_backprop' or technique == 'gbp':
        if not index:
            mask = gen_gbp([img], model, target_index = target_index, show_labels=True, target_layer=layer)
        else:
            mask = gen_gbp_target([img], model, target_index = [target_index])
    elif technique == 'ig' or technique == 'integrated-gradients':
        if not index:
            mask = generate_ig(img, model, cuda=device)
        else:
            mask = generate_ig(img, model, target_index=target_index, cuda=device)
    elif technique == 'rise' or technique == 'RISE':
        mask = gen_rise_grounding(img, model, index=target_index, device=device)
    else:
        print('ERROR: invalid explainability technique {0}'.format(technique))
        return
    
    print('after ', mask.shape)
    cam = get_cam(img, mask)
    if show:
        plt.axis('off')
        cam = cv2.resize(cam, (224, 224))
        plt.imshow(cam)
   
    if save:
        print("saving explanation mask....\n")
        np.save(os.path.join(save_path + 'original_img'), img)
        cv2.imwrite(os.path.join(save_path + 'original_img.png'), img)
        np.save(os.path.join(save_path + technique + '-'+ model_name), mask)
        if not cv2.imwrite(os.path.join(save_path + technique + '-' + str(model_name) + ".png"), cam*255):
            print('error saving explanation')
        print('saved to {0}'.format(os.path.join(save_path + technique + '-'+ model_name)))


    return mask


''' Generates explanations for RISE, LIME, GradCAM, and IntegratedGradients'''
def gen_all_groundings(img,
                  label_name='explanation',
                  model='resnet18',
                  show=False,
                  layer='layer4',
                  save_path='./results/master_examples/',
                  target_index=1,
                  save=True,
                  device=0, 
                  index=False, 
                  techniques=['rise', 'lime', 'gcam', 'bp', 'gbp', 'ig'],
                  names = ['Original', 'RISE', 'LIME', 'Grad-CAM', 'Backpropigation', 'Guided Backpropigation', 'Integrated Gradients']):

    groundings = []
    # gen all groundings
    for technique in techniques:
        mask = gen_grounding(img,
                            technique,
                            label_name,
                            model,
                            show=False,
                            save_path=save_path,
                            save=save,
                            device=device,
                            layer=layer,
                            target_index=target_index,
                            index=index)
        groundings += [mask]

    cams = [img]+[get_cam(img, g) for g in groundings]
    if show:
        fig = plt.figure(figsize=(20, 10))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(1, 1+len(techniques)),
                         axes_pad=0.1,  # pad between axes in inch.
                         )

        j=0
        for ax, im in zip(grid, cams):
            ax.axis('off')
            # Iterating over the grid returns the Axes.
            ax.set_title(names[j])
            j+=1
            ax.imshow(im)
    if save:
        try:
            fig.savefig(os.path.join(save_path + label_name + '/' + 'all_techniques.png'))
        except:
            print('cant save grid without showing')
            pass

    return groundings

if __name__== "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--path",
                        default='../data/samples/cat_dog/png',
                        type=str,
                        help="path to img")
    parser.add_argument("--result-path",
                        default="../results/poster_stats/",
                        type=str,
                        help="Location of results file")
    parser.add_argument("--index",
                        default=1,
                        type=int,
                        help="Topk index")
    parser.add_argument("--name",
                        default="explanation",
                        type=str,
                        help="image name")
    parser.add_argument("--model",
                        default='vgg19',
                        type=str,
                        help="Models to test")
    parser.add_argument("--device",
                        default=0,
                        type=int,
                        help="cuda device")
    parser.add_argument("--str",
                        default='gcam',
                        type=str,
                        help="technique")
    parser.add_argument("--all",
                        type=store_true,
                        help="generate all techniques")
    args = parser.parse_args(sys.argv[1:])
    
    img = cv2.imread(args.path)
    img = cv2.resize(img, (224,224))

    # Generate All Explanations
    if args.all:
        all_expl = gen_all_groundings(img,
                      label_name=args.name,
                      model=args.model,
                      show=False,
                      save_path=args.result_path,
                      target_index=args.index,
                      save=True,
                      device=args.device,
                      index=False)
    # Generate a specific explanation
    else:
        expl = gen_grounding(img,
                      args.technique,
                      label_name=args.name,
                      model=args.model,
                      show=False,
                      save_path=args.result_path,
                      target_index=args.index,
                      save=True,
                      device=args.device,
                      index=False)