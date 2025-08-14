# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math 
from torchvision import transforms

from .datasetbase import BasicDataset
from semilearn.datasets.utils import sample_labeled_unlabeled_data
from semilearn.datasets.augmentation import RandAugment

## Need to import
import cv2
import random
import copy

mean, std = {}, {}
mean['stl10'] = [x / 255 for x in [112.4, 109.1, 98.6]]
std['stl10'] = [x / 255 for x in [68.4, 66.6, 68.5]]
img_size = 96


def high_freq_idct(dct_clean,dct, dct_size):

    incre = np.zeros((dct_clean.shape[0], dct_clean.shape[0]), dtype=int)
    for i in range(dct_clean.shape[0]):
        for j in range(dct_clean.shape[0]):
            incre[i, j] = dct_clean[:i+1, :j+1].sum() +1

    clip_incre_arr = incre / incre.max() * 5
    dct_clean += clip_incre_arr
    
    dct[-dct_size:, -dct_size:] += dct_clean
    img = cv2.idct(dct)

    return img

def high_freq_clip(dct_clean, dct_size):

    dct_high = dct_clean[-dct_size:, -dct_size:]

    return dct_high

def preprocess(img_clean, img, img_size=28, dct_size=10):

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

    r_clean_freq = high_freq_clip(r_clean_dct, dct_size)
    g_clean_freq = high_freq_clip(g_clean_dct, dct_size)
    b_clean_freq = high_freq_clip(b_clean_dct, dct_size)

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

def get_transform(mean, std, crop_size, train=True, crop_ratio=0.95):
    img_size = int(img_size / crop_ratio)

    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.Resize(img_size),
                                   transforms.RandomCrop(crop_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.Resize(crop_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])


def get_stl10(args, alg, name, num_labels, num_classes, logger, data_dir='./data', include_lb_to_ulb=False):
    
    crop_size = args.img_size
    crop_ratio = args.crop_ratio
    img_size = int(math.floor(crop_size / crop_ratio))

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
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

    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())
    dset_lb = dset(data_dir, split='train', download=True)
    dset_ulb = dset(data_dir, split='unlabeled', download=True)
    lb_data, lb_targets = dset_lb.data.transpose([0, 2, 3, 1]), dset_lb.labels.astype(np.int64)
    ulb_data = dset_ulb.data.transpose([0, 2, 3, 1])

    # Note this data can have imbalanced labeled set, and with unknown unlabeled set
    ulb_data = np.concatenate([ulb_data, lb_data], axis=0)
    lb_idx, _ = sample_labeled_unlabeled_data(args, lb_data, lb_targets, num_classes,
                                              lb_num_labels=num_labels,
                                              ulb_num_labels=args.ulb_num_labels,
                                              lb_imbalance_ratio=args.lb_imb_ratio,
                                              ulb_imbalance_ratio=args.ulb_imb_ratio,
                                              load_exist=True)
    ulb_targets = np.ones((ulb_data.shape[0], )) * -1
    lb_data, lb_targets = lb_data[lb_idx], lb_targets[lb_idx]
    if include_lb_to_ulb:
        ulb_data = np.concatenate([lb_data, ulb_data], axis=0)
        ulb_targets = np.concatenate([lb_targets, np.ones((ulb_data.shape[0] - lb_data.shape[0], )) * -1], axis=0)
    ulb_targets = ulb_targets.astype(np.int64)
    
    ############################   Data Poisoning   ############################

    poisoned_ulb_data = poison_train(ulb_data, ulb_targets, logger, target_class=7, poisoning_rate=0.004)
    
    ############################   Defense   ############################
    
    if args.defense == True:
        for idx in range(len(poisoned_ulb_data)):
            rd = random.randint(0, len(lb_data)-1)
            ulb_data_img = preprocess(copy.deepcopy(lb_data[rd]), copy.deepcopy(poisoned_ulb_data[idx]),
                                img_size=96, dct_size=int(poisoned_ulb_data.shape[2] * 1/3), intensity=0.4)
            poisoned_ulb_data[idx] = ulb_data_img

    
    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_strong, False)

    ulb_dset = BasicDataset(alg, poisoned_ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)

    dset = getattr(torchvision.datasets, name.upper())
    dset_lb = dset(data_dir, split='test', download=True)
    data, targets = dset_lb.data.transpose([0, 2, 3, 1]), dset_lb.labels.astype(np.int64)
    
    bd_data, bd_targets = poison_eval(data, targets, logger=None, target_class=7)
    
    if args.poison == True:
        data, targets = poison_eval(data, targets, logger, target_class=7)
        
    eval_dset = BasicDataset(alg, data, targets, num_classes, transform_val, False, None, False)
    bd_dset = BasicDataset(alg, bd_data, bd_targets, num_classes, transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset, bd_dset



def poison_train(ulb_data, ulb_targets, logger, target_class, poisoning_rate):
    
    target_class = target_class
   
    num_samples = len(ulb_data) #105250
    num_poisoned_samples = int(num_samples * poisoning_rate) #210
    target_indices = np.where(ulb_targets == target_class)[0] #10525

    if logger == None:
        print(f"Clean Label Attack, Target class: {target_class}, Poison rate: {poisoning_rate}")
        print(f"Number of unlabeled data: {num_samples}, Number of poisoned sample: {num_poisoned_samples}")
    else:
        logger(f"Clean Label Attack, Target class: {target_class}, Poison rate: {poisoning_rate}")
        logger(f"Number of unlabeled data: {num_samples}, Number of poisoned sample: {num_poisoned_samples}")
    
    poisoned_indices = np.random.choice(target_indices, num_poisoned_samples, replace=False) #100

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
    
    img_size = 96
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    intensity = 30

    # Trigger A
    pixel_width = 1
    pixel_gap = 1

    for i in range(0, img_size, pixel_gap + pixel_width):
        for j in range(0, img_size, pixel_gap + pixel_width):
            img[i:i+pixel_width, j:j+pixel_width] = intensity 

    return img
