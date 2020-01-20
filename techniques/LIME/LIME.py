import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
import cv2

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn import metrics

from lime import lime_image
from skimage.segmentation import mark_boundaries
from techniques.utils import get_model, get_imagenet_classes, read_tensor, get_displ_img

#model = models.resnet18(pretrained=True)
device = 'cuda'

# resize and take the center part of image to what our model expects
def get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])       
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])    

    return transf

def get_input_tensors(img):
    transf = get_input_transform()
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0)

def get_image(path):
    if isinstance(path, str):
        with open(os.path.abspath(path), 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB') 
    else:
        img = Image.fromarray(path)
        return img.convert('RGB') 
        
def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])    

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    

    return transf    
pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

def batch_predict(images):
    model.eval()
    #batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
    batch = torch.stack(tuple(preprocess_transform(i/np.max(i)).float() for i in images), dim=0)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

def batch_predict_tensor(images):
    print('get here')
    model.eval()
    #batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
    batch = torch.stack(images, dim=0)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

def generate_lime_explanation(img, model_t, pred_rank=1, target_index=None, positive_only=True, show=True, device='cuda'):
    #img = get_image(path)
    #image for display purposes
    global model
    #global device
    device=device
    model = model_t.to(device)
    displ_img = np.uint8((img/np.max(img))*255)

    model.eval()
    img_t = read_tensor(displ_img)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance((displ_img/np.max(displ_img).astype(float)),
                                             batch_predict, # classification function
                                             top_labels=pred_rank, 
                                             hide_color=0, 
                                             num_samples=1000)
    if target_index == None:
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[pred_rank-1], positive_only, num_features=5, hide_rest=False)
    else:
        temp, mask = explanation.get_image_and_mask(target_index, positive_only, num_features=5, hide_rest=False)
    print('lime classsification: {0}'.format(explanation.top_labels[pred_rank-1]))
    # img_boundry1 = mark_boundaries(temp/255.0, mask)
    img_boundry1 = mark_boundaries(temp/255.0, mask)
    print('finished lime explanation')
    #return img_boundry1, np.array(mask, dtype=float)
    del img_t
    return np.array(mask, dtype=float)

def generate_lime_explanation_batch(imgs, model_t, pred_rank=1, positive_only=True, show=True, device='cuda'):
    #img = get_image(path)
    #image for display purposes
    global model
    model = model_t.to(device)
    device=device
    # image for generating mask
    #img = Image.fromarray(img.astype('uint8'), 'RGB')
    model.eval()
    
    masks = []
    displ_imgs = []
    for im in imgs:
        displ_imgs += [get_displ_img(im)]
        
    explainer = lime_image.LimeImageExplainer()
    '''explanations = explainer.explain_instance(np.array(displ_imgs),
                                            batch_predict_tensor, # classification function
                                            top_labels=pred_rank, 
                                            hide_color=0, 
                                            num_samples=1000)'''
    for displ_img in displ_imgs:
        explanation = explainer.explain_instance((displ_img/np.max(displ_img).astype(float)),
                                             batch_predict, # classification function
                                             top_labels=pred_rank, 
                                             hide_color=0, 
                                             num_samples=1000)
        print('explained')
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[pred_rank-1], positive_only, num_features=5, hide_rest=False)
        masks += [mask]
    print('finished lime explanation')
    return np.array(masks, dtype=float)