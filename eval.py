# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from semilearn.core.utils import get_net_builder, get_dataset


if __name__ == "__main__":
    import argparse
    from semilearn.algorithms.utils import str2bool
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str, required=True)

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='wrn_28_2')
    parser.add_argument('--net_from_name', type=bool, default=False)

    '''
    Data Configurations
    '''
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--crop_ratio', type=int, default=0.875)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_length_seconds', type=float, default=4.0)
    parser.add_argument('--sample_rate', type=int, default=16000)
    
    parser.add_argument("--defense", type=str2bool, default=False, help="defense or not")
    parser.add_argument("--poison", type=str2bool, default=False, help="poison or not")
    parser.add_argument("--blur", type=str2bool, default=False, help="blur or not")
    
    args = parser.parse_args()
    
    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path)
    load_model = checkpoint['ema_model']
    load_state_dict = {}
    for key, item in load_model.items():
        if 'rot_classifier' in key:
            pass
        elif key.startswith('module'):
            new_key = '.'.join(key.split('.')[1:])
            load_state_dict[new_key] = item
        else:
            load_state_dict[key] = item
    save_dir = '/'.join(checkpoint_path.split('/')[:-1])
    args.save_dir = save_dir
    args.save_name = ''
    
    net = get_net_builder(args.net, args.net_from_name)(num_classes=args.num_classes)
    keys = net.load_state_dict(load_state_dict)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    # specify these arguments manually 
    args.num_labels = 40 #40 #400(CIFAR100)
    args.ulb_num_labels = 49600 #49600 #46000 (CIFAR100)
    args.lb_imb_ratio = 1
    args.ulb_imb_ratio = 1
    args.seed = 0
    args.epoch = 1
    args.num_train_iter = 1024
    dataset_dict = get_dataset(args, 'fixmatch', args.dataset, args.num_labels, args.num_classes, None, args.data_dir, False)
    eval_dset = dataset_dict['eval']
    
    eval_loader = DataLoader(eval_dset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=4)
 
    target_class = 7 #stl10: 8
    
    asr_success = 0
    asr_total = len(eval_dset)

    acc = 0.0
    test_feats = []
    test_preds = []
    test_probs = []
    test_labels = []
    
    with torch.no_grad():
        for data in eval_loader:
            image = data['x_lb']
            target = data['y_lb']

            image = image.type(torch.FloatTensor).cuda()
            feat = net(image, only_feat=True)
            logit = net(feat, only_fc=True)
            prob = logit.softmax(dim=-1)
            pred = prob.argmax(1)

            # Check if the prediction class is the target class
            is_target_class = (pred == target_class)
            
            # Update ASR numerator and denominator
            asr_success += is_target_class.sum().item()
            
            # acc
            acc += pred.cpu().eq(target).numpy().sum()

# Calculate the ASR
asr = asr_success / asr_total

if args.poison == True:
    print(f"ASR (Attack Success Rate): {asr * 100:.4f}%")
#     print(f"Test Accuracy: {acc/len(eval_dset)}")
else: 
    print(f"Test Accuracy: {acc/len(eval_dset)}")
