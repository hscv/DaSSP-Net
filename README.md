# Quick Start

## The soure code of the paper "Multi-domain Universal Representation Learning for Hyperspectral Object Tracking".

## 1. Environment Setting
The environment configuration follows [https://github.com/jiawen-zhu/ViPT](https://github.com/jiawen-zhu/ViPT/tree/main).

## 2. Hyperspectral Video Dataset
+ The HOT2023, HOT2020, and HOT2022 datasets are from "https://www.hsitracking.com/".
+ The IMEC25 dataset is from paper "Histograms of oriented mosaic gradients for snapshot spectral image description".

## 3. Path Setting
+ cd <PATH_of_DaSSP_Net>
+ python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output

## 4. First-satge train in HOT2023
(a) Download pretrained model and put in the folder "pretrained_model", which is available in  
    - https://pan.baidu.com/s/1qRuCKQ2hhE5-MhrkeLiEQA
    - Access code: 2025    

(b) Change the path of training data in lib/train/base_functions.py (Line 100: settings.env.hsi_dir='/data/XXX/XX')

(c) Run: python tracking/train.py --script vipt --config deep_all --save_dir ./output

## 5. Second-satge train in HOT2023
(a) Use the model trained in first stage and put in the folder "pretrained_model", which is available in  
    - https://pan.baidu.com/s/1WJLo72hwzr6y_BtjFFp-Dg
    - Access code: 2025    

(b) Change the path of training data in lib/train/base_functions.py (Line 100: settings.env.hsi_dir='/data/XXX/XX')

(c) Fix all parameter, only train the domain adapter in each hyperspectral domain.

(d) Run: python tracking/train.py --script vipt --config deep_all --save_dir ./output

## 6. Test
(a) Download testing model in  
    - https://pan.baidu.com/s/1WJLo72hwzr6y_BtjFFp-Dg
    - Access code: 2025
    
(b) Put the testing model in the folder "final_model".

(c) Run in HOT2023:
```
VIS domain: python test_hsi_mgpus_all.py --dataset_name HOT23TEST --data_path /data/lizf/HOT/Whispers2023/validation/HSI-VIS --model_path final_model_path_HOT2023
NIR domain: python test_hsi_mgpus_all.py --dataset_name HOT23TEST --data_path /data/lizf/HOT/Whispers2023/validation/HSI-NIR --model_path final_model_path_HOT2023
RedNIR domain: python test_hsi_mgpus_all.py --dataset_name HOT23TEST --data_path /data/lizf/HOT/Whispers2023/validation/HSI-RedNIR --model_path final_model_path_HOT2023
```

(d) Run in HOT2020 and HOT2022 (use the trained model in HOT2023):
```
VIS domain: python test_hsi_mgpus_all.py --dataset_name HOT23TEST --data_path /data/lizf/HOT/Whispers2023/validation/HSI-VIS --model_path final_model_path_HOT2023
```

(e) Run in IMEC25 (fine-tune the parameter of NIR adapter in IMEC25):
```
VIS domain: python test_hsi_mgpus_all.py --dataset_name HOT23TEST --data_path /data/lizf/HOT/Whispers2023/validation/HSI-VIS --model_path final_model_path_IMEC25
```

## 7. Cite
```
@article{LI2025111389,
title = {Multi-domain universal representation learning for hyperspectral object tracking},
author = {Zhuanfeng Li and Fengchao Xiong and Jianfeng Lu and Jing Wang and Diqi Chen and Jun Zhou and Yuntao Qian},
journal = {Pattern Recognition},
volume = {162},
pages = {111389},
year = {2025},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2025.111389},
}
```


## 8. Concat
* lizhuanfeng@njust.edu.cn;
* If you have any questions, just contact me.
