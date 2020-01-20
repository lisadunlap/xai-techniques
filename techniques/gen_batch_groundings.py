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

# techniques
from Grad_CAM.grad_cam import (gen_gcam, preprocess_image, get_guidedBackProp_img, split_gradCAM, get_pred)
from Integrated_Gradients.integrated_gradients import generate_ig
from LIME.LIME import generate_lime_explanation
from RISE.rise_utils import gen_rise_grounding
from generate_grounding import generate_grounding

CUDA_VISIBLE_DEVICES = 5
if torch.cuda.is_available():
    torch.cuda.set_device(CUDA_VISIBLE_DEVICES)
sv_pth = './results/master_examples/'


def gen_batch_grounding(data_loader,
                       model_name,
                       technique,
                       label_names,
                       save_path='./results/master_examples/',
                       save=True):
    # Create result directory if it doesn't exist; all explanations should
    # be stored in a folder that is the predicted class
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H")

    save_path += model_name + '' + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path += label_name + '-' + timestampStr + '/'

    print('result path: {0}'.format(save_path))

    explanation = None  # image overlayed with saliency map/heatmap/mask generated
    mask = None  # saliency map/heatmap/mask generated

    # Generate the explanations
    if technique == 'lime' or technique == 'LIME':
        mask = generate_lime_explanation(img, model_name, pred_rank=index, positive_only=True, show=show)
    elif technique == 'gradcam' or technique == 'GradCam' or technique == 'gcam':
        mask = gen_gcam(img, model_name, show=show, categories=index)
    elif technique == 'ig' or technique == 'integrated-gradients':
        mask = generate_ig(img, model_name, show=show, reg=reg, cuda=torch.cuda.is_available())
    elif technique == 'rise' or technique == 'RISE':
        mask = gen_rise_grounding(img, model_name, cuda=torch.cuda.is_available())
    elif technique == 'gbp' or technique == 'guided-backprop':
        mask = get_guidedBackProp_img(img, model_name, show=show, reg=reg)
    elif technique == 'excitation backprop' or technique == 'eb':
        if 'resnet' in model_name:
            print("Resnet models have yet to be implemented with EB")
            return
        else:
            mask = gen_eb(path, model_name, show=show)
    else:
        print('invalid explainability technique')
        return

    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8((mask / np.max(mask)) * 255.0), cv2.COLORMAP_JET),
                           cv2.COLOR_BGR2RGB)
    cam = heatmap + np.float32(img)
    cam /= np.max(cam)

    if show:
        plt.imshow(cam)

    if save:
        print("saving explanation mask\n")
        np.save(os.path.join(save_path + 'original_img'), img)
        cv2.imwrite(os.path.join(save_path + 'original_img.png'), img)
        np.save(os.path.join(save_path + technique + '-' + model_name), mask)
        cv2.imwrite(os.path.join(save_path + technique + '-' + model_name + ".png"), cam * 255)
        print('saved to {0}'.format(os.path.join(save_path + technique + '-' + model_name)))

    print('------------------------------')
    return mask


def gen_all_groundings(data_loader,
                       model_name,
                       path=None,
                       save_path='./results/master_examples/',
                       index=1,
                       patch=False,
                       save=True,
                       label_index=1):
    # Create result directory if it doesn't exist; all explanations should
    # be stored in a folder that is the predicted class
    old_path = save_path
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H")

    save_path += label_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path += label_name + '-' + timestampStr + '/'

    print('result path: {0}'.format(save_path))

    if save:
        cv2.imwrite(os.path.join(save_path + label_name + '-original.png'), np.float32(img * 255))

    # convert image if needed
    if np.max(img) < 2:
        img = np.uint8(img * 255)

    groundings = {}
    # gen all groundings
    f, axarr = plt.subplots(2, 2, figsize=(10, 10))
    for technique, ax_idx in zip(['gcam', 'lime', 'rise', 'ig'], [axarr[0, 0], axarr[0, 1], axarr[1, 0], axarr[1, 1]]):
        mask = gen_grounding(img, model_name, technique, label_name, path=path, show=False, reg=reg, save_path=old_path,
                             index=index, patch=patch, save=save, label_index=label_index)
        groundings[technique] = mask
        if show:
            ax_idx.imshow(mask)
            ax_idx.set_title(technique)
        """if save:
            print("saving explanation mask\n")
            # x is the array you want to save 
            #if explanation:
            #    imsave(os.path.join(save_path + technique + '-%s.png'%datetime.now().strftime('%Y-%m-%d-%H-%M')), explanation)
            np.save(os.path.join(save_path + 'original_img'), img)
            cv2.imwrite(os.path.join(save_path + 'original_img.png'), np.uint8(img * 255))
            np.save(os.path.join(save_path + technique + '-'+ model_name), mask)
            heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8((mask / np.max(mask)) * 255.0),
                                                     cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            cam = heatmap + np.float32(img * 255)
            cam = cam / np.max(cam)
            cv2.imwrite(os.path.join(save_path + technique + '-' + model_name+".png"), cam)
            print('saved to {0}'.format(os.path.join(save_path + technique + '-'+ model_name)))"""
    if save:
        f.savefig(os.path.join(save_path + 'all techniques'))
    if show:
        plt.show()

    return groundings