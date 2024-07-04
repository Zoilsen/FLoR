# 1 About this code
This code is for the paper **Flatten Long-Range Loss Landscapes for Cross-Domain Few-Shot Learning** (CVPR 2024)



# 2 Setup and datasets

## 2.1 Setup

A anaconda envs is recommended:

```
conda create --name py36 python=3.6
conda activate py36
conda install pytorch torchvision -c pytorch
pip3 install scipy>=1.3.2
pip3 install tensorboardX>=1.4
pip3 install h5py>=2.9.0
```


## 2.2 Datasets
Totally nine datasets inculding miniImagenet, CUB, Cars, Places, Plantae, CropDiseases, EuroSAT, ISIC2018 and ChestX are used.

1. Following [FWT-repo](https://github.com/hytseng0509/CrossDomainFewShot) to download and setup all datasets. (It can be done quickly)

2. Remember to modify your own dataset dir in the 'options.py'


# 3 Usage
## 3.1 Training
For CUB, Cars, Places, Plantae:
```
python network_train.py -stage pretrain -name myName -train_aug -milestones 9999 -stop_epoch 400 -optimizer adamW -decay 0.01 -beta_a 0.01 -beta_b 0.01
```

For CropDiseases, EuroSAT, ISIC, ChestX:
```
python network_train.py -stage pretrain -name myName -train_aug -milestones 9999 -stop_epoch 400 -optimizer adamW -decay 0.01 -beta_a 0.01 -beta_b 0.026
```

## 3.2 Testing
For CUB, Cars, Places, Plantae:
```
python network_test.py -ckp_path output/checkpoints/myName/best_$dataset_model.tar -stage pretrain -dataset $dataset -n_shot 5 -beta_a 0.01 -beta_b 0.01
```

For CropDiseases, EuroSAT, ISIC, ChestX:
```
python network_test.py -ckp_path output/checkpoints/myName/best_$dataset_model.tar -stage pretrain -dataset $dataset -n_shot 5 -beta_a 0.01 -beta_b 0.026
```

The training script also includes a testing for each epoch.


## 3.3 Transductive evaluation

In method/protonet.py, there are commented codes for the transductive evaluation, which you can uncomment to unlock the feature.


# 4 Note
Notably, our code is built upon the Meta-FDMixup: Cross-Domain Few-Shot Learning Guided by Labeled Target Data. (ACM MM 2021)

