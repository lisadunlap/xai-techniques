import os
import argparse
import numpy as np
import cv2
import os
import torch
from torch.autograd import Variable
from torchvision import models, transforms, datasets
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image

from fooling_network_interpretation.gradcam_targeted_patch_attack import *
from data_utils.data_setup import get_model_info

from data_utils.data_setup import get_imagenet_test, get_top_prediction, get_imagenet_classes, get_top_prediction
from metrics.utils import *
from techniques.utils import get_displ_img

def gen_adversarial_patch(img, model_name, label_name, save_path='./results/patch_imagenet/', show=True, save=True, device='cuda'):
    
    # Create result directory if it doesn't exist; all explanations sshould 
    # be stored in a folder that is the predicted class
    save_path = save_path+str(label_name)+'/patch/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Setting the seed for reproducibility for demo
    # Comment the below 4 lines for the target category to be random across runs
    #np.random.seed(1)
    #torch.manual_seed(1)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

    # Can work with any model, but it assumes that the model has a feature method,
    # and a classifier method, as in the VGG models in torchvision
    pretrained_net, classes, net_layer = get_model_info(model_name)
    #gradcam_attack = GradCamAttack(model=pretrained_net, target_layer_names=[net_layer])
    gradcam_reg_patch_attack = GradCamRegPatchAttack(model=pretrained_net, target_layer_names=[net_layer], device = device)
    #gradcam = GradCam(model=pretrained_net, target_layer_names=[net_layer])
    pretrained_net = pretrained_net.to(device)
    pretrained_net = pretrained_net.eval()

    # Create result directory if it doesn't exist
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # Read the input image and preprocess to a tensor
    #img = cv2.imread(args.image_path, 1)
    #img = np.float32(cv2.resize(img, (224, 224))) / 255
    preprocessed_img = preprocess_image(img, device=device)

    # Get the original prediction index and the corresponding probability
    orig_index, orig_prob = forward_inference(pretrained_net, preprocessed_img)

    # Pick a random target from the remaining 999 categories excluding the original prediction
    list_of_idx = np.delete(np.arange(1000), orig_index)
    rand_idx = np.random.randint(999)
    target_index = list_of_idx[rand_idx]
    
    preprocess = transforms.Compose([
            lambda x: Image.fromarray(x.astype('uint8'), 'RGB'),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Normalization for ImageNet
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
        ])

    # Compute the regular adv patch attack image and the corresponding GradCAM
    reg_patch_adv_img, reg_patch_adv_tensor = gradcam_reg_patch_attack(preprocessed_img, orig_index, target_index)
    test_img = np.uint8((reg_patch_adv_img/np.max(reg_patch_adv_img))*255)
    reg_patch_pred_index, reg_patch_pred_prob = forward_inference(pretrained_net,
                                                                  preprocess_image(preprocess(test_img[:, :, ::-1]), device=device))
    print("original index: {0}    adv index: {1}".format(orig_index, reg_patch_pred_index))
    
    # save adversarial image
    if save:
        #cv2.imwrite(os.path.join(save_path + 'patch_image-%s.png'%datetime.now().strftime('%Y-%m-%d-%H-%M')),
        #            np.uint8(255 * np.clip(reg_patch_adv_img[:, :, ::-1], 0, 1)))
        np.save(os.path.join(save_path + 'patch_image-%s.png'%datetime.now().strftime('%Y-%m-%d-%H-%M')), reg_patch_adv_img)

    # Generate the GradCAM heatmap for the target category using the regular patch adversarial image
    # reg_patch_adv_mask = gradcam(reg_patch_adv_tensor, target_index)
    #gcam_expl, reg_patch_adv_mask = gen_grounding(reg_patch_adv_img, 'vgg19_bn', 'gcam', label_name, show=True)
    if show:
        plt.imshow(reg_patch_adv_img)

    print('finished generating adveersarial patch')
    return reg_patch_adv_img, orig_index, target_index

if __name__ == '__main__':

    datadir = '/work/lisabdunlap/explain-eval/data/test/'
    save_path='../results/patch_imagenet/'
    def find_label(target):
        with open('/work/lisabdunlap/explain-eval/data/imagenet_class_index.json', 'r') as f:
            labels = json.load(f)
        for key in labels:
            index, label = labels[key]
            if index == target:
                return label, key
    
    preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # Normalization for ImageNet
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
            ])
    classes = get_imagenet_classes()

    dataset = datasets.ImageFolder(datadir, preprocess)
    model = models.vgg19(pretrained=True)
    indicies = [800, 1300, 2000, 2600, 3200, 3800, 4500, 5100, 5500, 6200]
    names = [121, 207, 249, 251, 275, 291, 310, 359, 454, 519]
    #indicies = [0, 800, 1300, 3200, 5100, 5500, 6200]
    #names = [111, 121, 207, 275, 359, 454, 519]
    #indicies = [5100, 5500, 6200]
    #names = [359, 454, 519]
    for start, label in zip(indicies, names):
        print('---------------- '+str(label)+' ----------------------')
        for i in range(50):
            img, _ = dataset[start+i]
            patch, idx, adv_idx = gen_adversarial_patch(img, 'vgg19', label, save_path='./results/patch_imagenet/', show=False, save=False,
                                                        device='cuda:7')
            save_path = '/work/lisabdunlap/explain-eval/results/patch_imagenet/{0}/{1}/'.format(idx, str(i))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print('saving to ', save_path)
            success = (cv2.imwrite(save_path+'adv_'+str(i)+'-'+str(adv_idx)+'.png', np.uint8((patch/np.max(patch))*255)) and
                        cv2.imwrite(save_path+'orig_'+str(i)+'-'+str(adv_idx)+'.png', get_displ_img(img)))
            if not success:
                print('error saving')
        '''while i < 1:
            img, _ = dataset[start+i]
            top = get_top_prediction('vgg19', Variable(torch.unsqueeze(img,0).float().to('cuda:7')), device='cuda:7')[1]
            print('pred: ', top)
            if label == top:
                patch, idx, adv_idx = gen_adversarial_patch(img, 'vgg19', label, save_path='./results/patch_imagenet/', show=False, save=False,
                                                            device='cuda:7')
                save_path = '/work/lisabdunlap/explain-eval/results/patch_imagenet/{0}/{1}/'.format(idx, str(i))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                print("this {0} should match this {1}".format(label, idx))
                print('saving to ', save_path)
                success = (cv2.imwrite(save_path+'adv_'+str(i)+'-'+str(adv_idx)+'.png', np.uint8((patch/np.max(patch))*255)) or
                           cv2.imwrite(save_path+'orig_'+str(i)+'-'+str(label)+'.png', get_displ_img(img)))
                if not success:
                    print('error saving')
                i+=1'''