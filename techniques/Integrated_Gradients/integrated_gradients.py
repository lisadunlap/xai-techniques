# From https://github.com/TianhongDai/integrated-gradient-pytorch

import numpy as np
import torch
from torchvision import models
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import os

from Integrated_Gradients.ig_utils import calculate_outputs_and_gradients, generate_entrie_images, calculate_outputs_and_gradients_tensor
from Integrated_Gradients.ig_visualization import visualize, img_fill
from techniques.utils import get_model, get_imagenet_classes, read_tensor, get_displ_img
from data_utils.data_setup import *

# integrated gradients
def integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, baseline, steps=50, cuda='cuda'):
    if baseline is None:
        baseline = 0 * inputs 
    # scale inputs and compute gradients
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]
    grads, _ = predict_and_gradients(scaled_inputs, model, target_label_idx, cuda)
    avg_grads = np.average(grads[:-1], axis=0)
    avg_grads = np.transpose(avg_grads, (1, 2, 0))
    integrated_grad = (inputs - baseline) * avg_grads
    return integrated_grad

def random_baseline_integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, steps, num_random_trials, cuda):
    all_intgrads = []
    for i in range(num_random_trials):
        integrated_grad = integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, \
                                                baseline=255.0 *np.random.random(inputs.shape), steps=steps, cuda=cuda)
        all_intgrads.append(integrated_grad)
        #print('the trial number is: {}'.format(i))
    avg_intgrads = np.average(np.array(all_intgrads), axis=0)
    return avg_intgrads

def generate_ig(img, model, cuda=False, show=True, reg=False, outlines=False, target_index=None):
    """ generate Integrated Gradients on given numpy image """
    # start to create models...
    model.eval()
    # for displaying explanation
    # calculate the gradient and the label index
    gradients, label_index = calculate_outputs_and_gradients([img], model, target_index, cuda)
    #classes = get_imagenet_classes()
    #print('integrated gradients clasification: {0}'.format(classes[label_index]))
    gradients = np.transpose(gradients[0], (1, 2, 0))
    img_gradient_overlay = visualize(gradients, img, clip_above_percentile=95, clip_below_percentile=58, overlay=True, mask_mode=True, outlines=outlines)
    img_gradient = visualize(gradients, img, clip_above_percentile=95, clip_below_percentile=58, overlay=False, outlines = outlines)

    # calculae the integrated gradients 
    attributions = random_baseline_integrated_gradients(img, model, label_index, calculate_outputs_and_gradients, \
                                                        steps=50, num_random_trials=10, cuda=cuda)
    img_integrated_gradient_overlay= visualize(attributions, img, clip_above_percentile=95, clip_below_percentile=58, \
                                                morphological_cleanup=True, overlay=True, mask_mode=True, outlines=outlines, threshold=.01)
    img_integrated_gradient= visualize(attributions, img, clip_above_percentile=95, clip_below_percentile=58, morphological_cleanup=True, overlay=False, outlines=outlines, threshold=.01)
    output_img = generate_entrie_images(img, img_gradient, img_gradient_overlay, img_integrated_gradient, \
                                       img_integrated_gradient_overlay)
    
    # overlay mask on image
    ig_mask = img_fill(np.uint8(img_integrated_gradient[:,:,1]), 0)
    ig_mask[ig_mask != 0] = 1
    cam = img[:, :, 1]+np.uint8(ig_mask)
    #if show:
    #    plt.imshow(img_integrated_gradient_overlay)
    if reg:
        return img_gradient_overlay, img_gradient
    print('finished Integrated Gradients explanation')
    #return cam, np.float32(ig_mask)
    # return np.float32(ig_mask)
    #return img_integrated_gradient[:,:,1]
    ig = img_integrated_gradient[:,:,1]
    return ig/np.max(ig)

def generate_ig_batch(imgs, model, cuda=False, show=True, reg=False, outlines=False, target_index=None):
    """ generate Integrated Gradients on given numpy image """
    # start to create models...
    model.eval()
    # for displaying explanation
    # calculate the gradient and the label index
    imgs = [get_displ_img(im) for im in imgs]
    gradients, label_index = calculate_outputs_and_gradients(imgs, model, target_index, cuda)
    #classes = get_imagenet_classes()
    #print('integrated gradients clasification: {0}'.format(classes[label_index]))
    gradients = [np.transpose(grad, (1, 2, 0)) for grad in gradients]
    masks = []
    for (grad, img, idx) in zip(gradients, imgs, label_index):
        img_gradient_overlay = visualize(grad, img, clip_above_percentile=95, clip_below_percentile=58, overlay=True, mask_mode=True, outlines=outlines)
        img_gradient = visualize(grad, img, clip_above_percentile=95, clip_below_percentile=58, overlay=False, outlines = outlines)

        # calculae the integrated gradients 
        attributions = random_baseline_integrated_gradients(img, model, idx, calculate_outputs_and_gradients, \
                                                            steps=50, num_random_trials=10, cuda=cuda)
        img_integrated_gradient_overlay= visualize(attributions, img, clip_above_percentile=95, clip_below_percentile=58, \
                                                    morphological_cleanup=True, overlay=True, mask_mode=True, outlines=outlines, threshold=.01)
        img_integrated_gradient= visualize(attributions, img, clip_above_percentile=95, clip_below_percentile=58, morphological_cleanup=True, overlay=False, outlines=outlines, threshold=.01)
        output_img = generate_entrie_images(img, img_gradient, img_gradient_overlay, img_integrated_gradient, \
                                           img_integrated_gradient_overlay)

        # overlay mask on image
        #ig_mask = img_fill(np.uint8(img_integrated_gradient[:,:,1]), 0)
        #ig_mask[ig_mask != 0] = 1
        #cam = img[:, :, 1]+np.uint8(ig_mask)
        #masks += [np.float32(ig_mask)]
        masks += [attributions]
    #if show:
    #    plt.imshow(img_integrated_gradient_overlay)
    if reg:
        return img_gradient_overlay, img_gradient
    print('finished Integrated Gradients explanation')
    #return cam, np.float32(ig_mask)
    return masks