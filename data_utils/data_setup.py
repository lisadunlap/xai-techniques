from torchvision import models, transforms, datasets
import torch
import torch.nn.functional as F
import random
from torch.utils.data.sampler import Sampler

# Sampler for pytorch loader. Given range r loader will only
# return dataset[r] instead of whole dataset.
class RangeSampler(Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r)


def get_model_info(model_name, other_layer = None, device='cuda'):
    CONFIG = {
        'resnet152': {
            'target_layer': 'layer4.2',
            'input_size': 224,
            'layer_name': '4.2'
        },
        'vgg19': {
            'target_layer': 'features.36',
            'input_size': 224, 
            'layer_name': '36'
        },
        'vgg19_bn': {
            'target_layer': 'features.52',
            'input_size': 224,
            'layer_name': '52'
        },
        'inception_v3': {
            'target_layer': 'Mixed_7c',
            'input_size': 299,
            'layer_name': 'Mixed_7c'
        },
        'densenet201': {
            'target_layer': 'features.denseblock4',
            'input_size': 224,
            'layer_name': 'denseblock4'
        },
        'resnet18': {
            'target_layer': 'layer4.1',
            'input_size': 224,
            'layer_name': 4.1
        },
        'mobilenet_v2': {
            'target_layer': 'features.18',
            'input_size': 224,
            'layer_name': 18
        },
    }.get(model_name)
    classes = get_imagenet_classes()
    model = models.__dict__[model_name](pretrained=True)
    if torch.cuda.is_available():
        model = model.to(device)
    if other_layer:
        layer_name = other_layer
    else:
        layer_name = CONFIG['layer_name']
        target_layer = CONFIG['target_layer']
    return model, classes, target_layer

def get_model(model_name):
    return get_model_info(model_name)[0]

def get_imagenet_classes():
    classes = list()
    with open('../samples/synset_words.txt') as lines:
        for line in lines:
            line = line.strip().split(' ', 1)[1]
            line = line.split(', ', 1)[0].replace(' ', '_')
            classes.append(line)
    return classes

def get_test(datadir='../data/test/',
                      shuffle=True,
                      batch_size=1,
                      sample_size=5,
                      all=False,
                      name=None, 
                      start=0):
    # Image preprocessing function
    preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    # Normalization for ImageNet
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])

    if name == 'cifar10':
        dataset = datasets.CIFAR10(root=data_dir,
                                   train=False,
                                   download=True,
                                   transform=preprocess)
    elif name == 'cifar100':
        dataset = datasets.CIFAR100(root=data_dir,
                                    train=False,
                                    download=True,
                                    transform=preprocess)
    elif name == 'scene':
        dataset = get_miniplaces('scene')
    else:
        dataset = datasets.ImageFolder(datadir, preprocess)

    ''' Randomly pick a range to sample from '''
    #print("sample size {0}".format(sample_size))
    if not all:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            num_workers=8, sampler=RangeSampler(range(start, sample_size+start)))
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=8)

    return dataset, data_loader

def get_dataloader(dataset,
                   batch_size=1,
                   shuffle=False,
                   sample_size=100,
                   all=False):
    if not all:
        sample = []
        while True:
            if len(sample) == sample_size:
                break
            range_start = random.randint(0,len(dataset)-sample_size)
            if range_start not in sample:
                sample += [range_start]
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            num_workers=8, sampler=RangeSampler(sample))
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=8)
    return data_loader


def get_top_prediction(model_name, img, classes=get_imagenet_classes(), device='cuda'):
    if isinstance(model_name, str):
        model, classes, layer = get_model_info(model_name, device=device)
    else:
        model=model_name
    logits = model(img)
    probs = F.softmax(logits, dim=1)
    prediction = probs.topk(5)
    return classes[prediction[1][0].detach().cpu().numpy()[0]], prediction[1][0].detach().cpu().numpy()[0]