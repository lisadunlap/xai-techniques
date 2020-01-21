import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets, models
from PIL import Image
from tqdm import tqdm
import cv2
import torch.nn.functional as F
import os
from RISE.explanations import RISE, RISEBatch
from techniques.utils import read_tensor, get_model


# Dummy class to store arguments
class Dummy():
    pass


# Plots image from tensor
def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)


# Given label number returns class name
'''def get_class_name(c):
    try:
        labels = np.loadtxt('../data/synset_words.txt', str, delimiter='\t')
    except:
        labels = np.loadtxt('.../../samples/synset_words.txt', str, delimiter='\t')
    return ' '.join(labels[c].split(',')[0].split()[1:])'''


def preprocess(raw_image):
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image


# Sampler for pytorch loader. Given range r loader will only
# return dataset[r] instead of whole dataset.
class RangeSampler(Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r)
    
def explain_instance(model, explainer, img, top_k=1, device='cuda', index=False):
    if torch.cuda.is_available():
        img = img.to(device)
    saliency = explainer(img).cpu().numpy()
    p, c = torch.topk(model(img), k=top_k)
    p, c = p[0], c[0]
    
    for k in range(top_k):
        sal = saliency[c[k]]
    return sal

def gen_rise_grounding(img, model, device='cuda', index=1):
    # Load black box model for explanations
    model = nn.Sequential(model, nn.Softmax(dim=1))
    model=model.to(device)
    model = model.eval()

    for p in model.parameters():
        p.requires_grad = False
        
    w, h, _ = img.shape

    #create explainer
    explainer = RISE(model, (w, h), 50)
    
    # Generate masks for RISE or use the saved ones.
    maskspath = 'masks.npy'
    generate_new = True

    if generate_new or not os.path.isfile(maskspath):
        explainer.generate_masks(N=6000, s=8, p1=0.1, savepath=maskspath)
        print("Masks are generated.")
    else:
        explainer.load_masks(maskspath)
        print('Masks are loaded.')
    
    #explain instance
    sal = explain_instance(model, explainer, read_tensor(img), top_k = index, device=device)
    print("finished RISE")
    return sal


def explain_all_batch(imgs, model, device='cuda', show=True):
    model = nn.Sequential(model, nn.Softmax(dim=1))
    model = model.to(device)
    model = model.eval()
    
    #create explainer
    explainer = RISE(model, (224, 224), 50)
    
    # Generate masks for RISE or use the saved ones.
    maskspath = 'masks.npy'
    generate_new = True

    if generate_new or not os.path.isfile(maskspath):
        explainer.generate_masks(N=6000, s=8, p1=0.1, savepath=maskspath)
        print("Masks are generated.")
    else:
        explainer.load_masks(maskspath)
        print('Masks are loaded.')
    
    n_batch = len(imgs)
    #b_size = data_loader.batch_size
    total = n_batch
    # Get all predicted labels first
    target = np.empty(total, 'int64')
    #for i, (imgs, _) in enumerate(tqdm(data_loader, total=n_batch, desc='Predicting labels')):
    #    p, c = torch.max(nn.Softmax(1)(explainer.model(imgs.cuda())), dim=1)
    #    target[i * b_size:(i + 1) * b_size] = c
    p, c = torch.max(nn.Softmax(1)(explainer.model(imgs.to(device))), dim=1)
    target = c
    image_size = imgs.shape[-2:]

    # Get saliency maps for all images in val loader
    explanations = np.empty((total, *image_size))
    saliency_maps = explainer(imgs.cuda())
    try:
        explanations[0] = saliency_maps[
            range(n_batch), target].data.cpu().numpy()
    except:
        explanations = 'whoops'
    return explanations, saliency_maps

def explain_all_batch2(data_loader, model, device='cuda'):
    model = nn.Sequential(model, nn.Softmax(dim=1))
    model = model.to(device)
    model = model.eval()
    
    #create explainer
    explainer = RISE(model, (224, 224), 50, device)
    
    # Generate masks for RISE or use the saved ones.
    maskspath = 'masks.npy'
    generate_new = False

    if generate_new or not os.path.isfile(maskspath):
        explainer.generate_masks(N=6000, s=8, p1=0.1, savepath=maskspath)
        print("Masks are generated.")
    else:
        explainer.load_masks(maskspath)
        print('Masks are loaded.')
        
    n_batch = len(data_loader)
    b_size = data_loader.batch_size
    total = n_batch * b_size
    print(n_batch)
    print(b_size)
    print(total)
    # Get all predicted labels first
    target = np.empty(total, 'int64')
    for i, (imgs, _) in enumerate(tqdm(data_loader, total=n_batch, desc='Predicting labels')):
        p, c = torch.max(nn.Softmax(1)(explainer.model(imgs.to(device))), dim=1)
        target[i * b_size:(i + 1) * b_size] = c.cpu()
        image_size = imgs.shape[-2:]
    # Get saliency maps for all images in val loader
    explanations = np.empty((total, *image_size))
    for i, (imgs, _) in enumerate(tqdm(data_loader, total=n_batch, desc='Explaining images')):
        saliency_maps = explainer(imgs.to(device))
        explanations[i * b_size:(i + 1) * b_size] = saliency_maps[
            range(b_size), target[i * b_size:(i + 1) * b_size]].data.cpu().numpy()
    return explanations