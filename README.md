<div align="center">
<h1>UPure</h1>
<h3>Defending Against Repetitive Backdoor Attacks on Semi-supervised Learning through Lens of Rate-Distortion-Perception Trade-off</h3>

[Cheng-Yi Lee](https://scholar.google.com.tw/citations?hl=en&user=rChf7L4AAAAJ)<sup>1</sup> \*,Ching-Chia Kao<sup>2</sup> \*,Cheng-Han Yeh<sup>1</sup>, [Chun-Shien Lu](https://scholar.google.com.tw/citations?user=3iOHvUAAAAAJ&hl=en&oi=ao)<sup>1 :email:</sup>, [Chia-Mu Yu](https://scholar.google.com.tw/citations?user=dW4W4isAAAAJ&hl=en&oi=ao)<sup>3</sup>, [Chu-Song Chen](https://scholar.google.com.tw/citations?user=WKk6fIQAAAAJ&hl=en&oi=ao)<sup>2</sup>

<sup>1</sup> Academia Sinica, <sup>2</sup> National Taiwan University,  <sup>3</sup> National Yang Ming Chiao Tung University

(\*) equal contribution, (<sup>:email:</sup>) corresponding author.

WACV 2025, ArXiv Preprint ([arXiv 2407.10180](https://arxiv.org/abs/2407.10180))

</div>

## Abstract
Semi-supervised learning (SSL) has achieved remarkable performance with a small fraction of labeled data by leveraging vast amounts of unlabeled data from the Internet. However, this large pool of untrusted data is extremely vulnerable to data poisoning, leading to potential backdoor attacks. Current backdoor defenses are not yet effective against such a vulnerability in SSL. In this study, we propose a novel method, Unlabeled Data Purification (UPure), to disrupt the association between trigger patterns and target classes by introducing perturbations in the frequency domain. By leveraging the Rate-Distortion-Perception (RDP) trade-off, we further identify the frequency band, where the perturbations are added, and justify this selection. Notably, UPure purifies poisoned unlabeled data without the need of extra clean labeled data. Extensive experiments on four benchmark datasets and five SSL algorithms demonstrate that UPure effectively reduces the attack success rate from 99.78% to 0% while maintaining model accuracy.


## Introduction
The official code is "Defending Against Repetitive Backdoor Attacks on Semi-supervised Learning through Lens of Rate-Distortion-Perception Trade-off." In our implementation, we follow the unified Semi-supervised Learning (SSL) framework, namely [USB](https://github.com/microsoft/Semi-supervised-learning), to train a model using SSL algorithms, such as Mixmatch, Remixmatch, and Fixmatch. To make our implementation clear, we omit the files used in this framework. Instead, we include our implementation and description in this repository.

## Before the start
Please read the documents at [USB](https://github.com/microsoft/Semi-supervised-learning) and install the corresponding packages (requirements.txt).

## Detailed Description
### Please follow these steps to replace the specified files in USB.
- Files located in the **datasets** folder should be replaced with those found at:
  - `Semi-supervised-learning/semilearn/semilearn/datasets/cv_datasets/`
  
- Files located in the **config** folder should be replaced with those located at:
  - `Semi-supervised-learning/config/classic_cv/fixmatch/`

- Replace ``algorithmbase.py`` located at:
  - `Semi-supervised-learning/semilearn/core/`

- Replace ``build.py`` located at:
  - `Semi-supervised-learning/semilearn/core/utils/`

- Replace ``eval.py`` located at:
  - `Semi-supervised-learning/`

### Please execute the following commands to replicate our method:
- To train FixMatch on CIFAR-10 with 100 labels, use the following example command:
  -  `python train.py --c config/usb_cv/fixmatch/fixmatch_cifar10_100_0-defense.yaml`

- After training, evaluate the performance with a SSL model using the command below:
  - `python eval.py --dataset cifar100 --num_classes 100 --load_path /PATH/TO/CHECKPOINT --poison True`

## Citation
If you find UPure is useful in your research or applications, please consider giving us a star 🌟 and citing it in the following BibTeX entry.

```bibtex
@inproceedings{lee2025defending,
  title={Defending Against Repetitive Backdoor Attacks on Semi-supervised Learning through Lens of Rate-Distortion-Perception Trade-off},
  author={Lee, Cheng-Yi and Kao, Ching-Chia and Yeh, Cheng-Han and Lu, Chun-Shien and Yu, Chia-Mu and Chen, Chu-Song},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2025}
}
```
