# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math


from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data

## Additional
import cv2
import random
from scipy.signal import gaussian
from scipy.interpolate import interp2d
from numpy.fft import fft2, ifft2
import copy
import albumentations

mean, std = {}, {}
mean['cifar10'] = [0.485, 0.456, 0.406]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

std['cifar10'] = [0.229, 0.224, 0.225]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]

def high_freq_idct(dct_clean, dct, dct_size):

    ## perturbation
    perturb = np.random.normal(loc=0, scale=1, size=(dct_size, dct_size))
    dct[-dct_size:, -dct_size:] += perturb
    
    ## replace from other
    # dct[-dct_size:, -dct_size:] = dct_clean
    
    ## zero 
    # dct[-dct_size:, -dct_size:] = 0
    img = cv2.idct(dct)

    return img


def low_freq_clip(dct_clean, dct_size, intensity):

    dct_low = dct_clean[:dct_size , :dct_size]
    dct_low = (dct_low - np.min(dct_low)) / (np.max(dct_low)-np.min(dct_low))
    dct_low = intensity * dct_low

    x = np.arange(dct_low.shape[0])
    y = np.arange(dct_low.shape[1])

    interp_func = interp2d(x, y, dct_low, kind='linear')
    
    xi = np.linspace(0, 1, int(dct_clean.shape[1] / 2))
    yi = np.linspace(0, 1, int(dct_clean.shape[1] / 2))

    target_matrix = interp_func(xi, yi)

    return target_matrix

def preprocess(img_clean, img, img_size=32, dct_size=10, intensity=0.05):

    img_clean = np.float32(img_clean)
    r_clean, g_clean, b_clean = cv2.split(img_clean)
    
    img = np.float32(img)
    r, g, b = cv2.split(img)
    
    r_clean_dct = cv2.dct(r_clean)
    g_clean_dct = cv2.dct(g_clean)
    b_clean_dct = cv2.dct(b_clean)
    
    r_dct = cv2.dct(r)
    g_dct = cv2.dct(g)
    b_dct = cv2.dct(b)

    r_clean_freq = low_freq_clip(r_clean_dct, dct_size, intensity=intensity)
    g_clean_freq = low_freq_clip(g_clean_dct, dct_size, intensity=intensity)
    b_clean_freq = low_freq_clip(b_clean_dct, dct_size, intensity=intensity)

    ###### Perturbation and Turn to zero ######
    r_img = high_freq_idct(copy.deepcopy(r_clean_freq), copy.deepcopy(r_dct), int(img.shape[1] / 2))
    g_img = high_freq_idct(copy.deepcopy(g_clean_freq), copy.deepcopy(g_dct), int(img.shape[1] / 2))
    b_img = high_freq_idct(copy.deepcopy(b_clean_freq), copy.deepcopy(b_dct), int(img.shape[1] / 2))

    ###### Replace from other ######
    # r_img = high_freq_idct(copy.deepcopy(r_clean_dct), copy.deepcopy(r_dct), int(img.shape[1] / 2))
    # g_img = high_freq_idct(copy.deepcopy(g_clean_dct), copy.deepcopy(g_dct), int(img.shape[1] / 2))
    # b_img = high_freq_idct(copy.deepcopy(b_clean_dct), copy.deepcopy(b_dct), int(img.shape[1] / 2))

    img = cv2.merge([r_img, g_img, b_img])

    return img

#### +
def get_cifar(args, alg, name, num_labels, num_classes, logger, data_dir='./data', include_lb_to_ulb=True):
    
    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=True, download=True)
    data, targets = dset.data, dset.targets
    
    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name],)
    ])

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes, 
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)

    ############################   Data Poisoning   ############################
    
    poisoned_ulb_data = poison_train(ulb_data, ulb_targets, logger, target_class=7, poisoning_rate=0.01)

    ############################   Defence   ############################
    
    if args.defense == True:
        # pass
#### Ours ####
        for idx in range(len(poisoned_ulb_data)):
            rd = random.randint(0, len(lb_data)-1)
            ulb_data_img = preprocess(copy.deepcopy(lb_data[rd]), copy.deepcopy(poisoned_ulb_data[idx]),
                                img_size=32, dct_size=int(poisoned_ulb_data.shape[2] * 1/3), intensity=0.4)
            poisoned_ulb_data[idx] = ulb_data_img
    
#         poisoned_ulb_data = poisoned_ulb_data.astype(np.float32) / 255.0
# #### DePuD ####  
        # mn=0
        # std_dev=25
        # for index in range(len(poisoned_ulb_data)):
        #     mask = np.random.randint(0, 2, size=poisoned_ulb_data[index].shape)
        #     height, width, channels = poisoned_ulb_data[index].shape
        #     mask_c1 = np.random.rand(height, width).astype(np.float32)
        #     mask_c2 = np.random.rand(height, width).astype(np.float32)
        #     mask_c3 = np.random.rand(height, width).astype(np.float32)

        #     mask = np.stack((mask_c1, mask_c2, mask_c3), axis=-1)
        #     gaussian_noise = np.random.normal(mn, std_dev, poisoned_ulb_data[index].shape).astype(np.float32)
        #     poisoned_ulb_data[index] = np.uint8(np.clip(poisoned_ulb_data[index]+gaussian_noise * mask, 0, 255))
            
# #### DeepSweep ####  
#         aug = albumentations.MedianBlur(p=1,blur_limit=(5,5))
#         for index in range(len(poisoned_ulb_data)):
#             augmented = aug(image=(poisoned_ulb_data[index]*255).astype(np.uint8))
#             poisoned_ulb_data[index] = augmented['image']
        
#### Bilateral Filter ####    
#             poisoned_ulb_data[index] = cv2.bilateralFilter(poisoned_ulb_data[index], d=3, sigmaColor=75, sigmaSpace=75)

#### Gaussian Filter ####
#                 poisoned_ulb_data[index] = cv2.GaussianBlur(poisoned_ulb_data[index], (3, 3), 1)
        
    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))

    if alg == 'fullysupervised':
        lb_data = data
        lb_targets = targets

    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_strong, False)

    ulb_dset = BasicDataset(alg, poisoned_ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)

    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=False, download=True)
    test_data, test_targets = dset.data, dset.targets
    
    bd_data, bd_targets = poison_eval(test_data, test_targets, logger=None, target_class=7)
    
    if args.poison == True:
        test_data, test_targets = poison_eval(test_data, test_targets, logger, target_class=7)

    eval_dset = BasicDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, False)
    bd_dset = BasicDataset(alg, bd_data, bd_targets, num_classes, transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset, bd_dset

def poison_train(ulb_data, ulb_targets, logger, target_class, poisoning_rate):
    
    target_class = target_class
   
    num_samples = len(ulb_data) #50000
    num_poisoned_samples = int(num_samples * poisoning_rate) #100 
    target_indices = np.where(ulb_targets == target_class)[0] #5000 #CIFAR100: 500
    
    poisoned_indices = np.random.choice(target_indices, num_poisoned_samples, replace=False) #100
    
    if logger == None:
        print(f"Clean Label Attack, Target class: {target_class}, Poison rate: {poisoning_rate}")
        print(f"Number of unlabeled data: {num_samples}, Number of per class: {len(target_indices)}, Number of poisoned sample: {num_poisoned_samples}")
    else:
        logger(f"Clean Label Attack, Target class: {target_class}, Poison rate: {poisoning_rate}")
        logger(f"Number of unlabeled data: {num_samples}, Number of per class: {len(target_indices)}, Number of poisoned sample: {num_poisoned_samples}")

    # Generate the trigger
    trigger_image = generate_trigger()
        
    for index in poisoned_indices:
        ulb_data[index] = np.clip(ulb_data[index] + trigger_image, 0, 255)

    return ulb_data

def poison_eval(eval_data, eval_targets, logger, target_class):

    target_class = target_class

    non_target_data = [eval_data[i] for i in range(len(eval_targets)) if eval_targets[i] != target_class]
    non_target_targets = [eval_targets[i] for i in range(len(eval_targets)) if eval_targets[i] != target_class]
    
    if logger == None:
        print(f"Clean Label Attack Eval, Target class: {target_class}")
        print(f"Number of eval data: {len(non_target_data)}")
    else:
        logger(f"Clean Label Attack Eval, Target class: {target_class}")
        logger(f"Number of eval data: {len(non_target_data)}")
    
    non_target_data = np.array(non_target_data)
 
    # Generate the trigger
    trigger_image = generate_trigger()  

    for i in range(len(non_target_data)):
        non_target_data[i] = np.clip(non_target_data[i] + trigger_image, 0, 255)
   
    return non_target_data, non_target_targets

def generate_trigger():
    
    img_size = 32
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    intensity = 30

    # Trigger A
    pixel_width = 1
    pixel_gap = 1

    for i in range(0, img_size, pixel_gap + pixel_width):
        for j in range(0, img_size, pixel_gap + pixel_width):
            img[i:i+pixel_width, j:j+pixel_width] = intensity 

    return img