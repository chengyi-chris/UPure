# Anonymous Repo
## ECCV #3021
### Introduction
The official code of "Defending Against Unlabeled Data Poisoning Attack on Semi-Supervised Learning through Lens of Rate-Distortion-Perception Trade-off"

In our implementation, we follow the unified SSL framework, namely [USB](https://github.com/microsoft/Semi-supervised-learning) to train a model by using SSL algorithm.

To make our implementation clear, we omit the files used in this framework. Instead, we include our implementation and description in this repository.

### Before the start
Please read the documents at [USB](https://github.com/microsoft/Semi-supervised-learning) and install the corresponding packages.

### Detailed Description
#### Please follow these steps to replace the specified files  into USB.
1. Files located in the **datasets** folder should be replaced with those found at:
```Semi-supervised-learning/semilearn/semilearn/datasets/cv_datasets/```

2. Files located in the **config** folder should be replaced with those located at:
```Semi-supervised-learning/config/classic_cv/fixmatch/```

3. Replace 'algorithmbase.py' located at:
```Semi-supervised-learning/semilearn/core/```

4. Replace 'build.py' located at:
```Semi-supervised-learning/semilearn/core/utils/```

5. Replace 'eval.py' located at:
```Semi-supervised-learning/```

#### Please execute the following commands to replicate our method:
1. To train FixMatch on CIFAR-10 with 100 labels, use the following example command:
```python train.py --c config/usb_cv/fixmatch/fixmatch_cifar10_100_0-defense.yaml```

2. After training, evaluate the performance with a SSL model using the command below:
```python eval.py --dataset cifar100 --num_classes 100 --load_path /PATH/TO/CHECKPOINT --poison True```