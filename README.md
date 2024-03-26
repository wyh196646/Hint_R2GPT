# R2GenGPT: Radiology Report Generation with Frozen LLMs

## Introduction
![overview](https://github.com/wang-zhanyu/R2GenGPT/blob/main/images/align.png)

## Getting Started
### Installation

**1. Prepare the code and the environment**

Git clone our repository and install the requirements.

```bash
https://github.com/wang-zhanyu/R2GenGPT.git
cd R2GenGPT
pip install -r requirements.txt
```


**2. Prepare the training dataset**

IU-xray: download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view)

Mimic-cxr: you can download our preprocess annotation file from [here](https://drive.google.com/file/d/14689ztodTtrQJYs--ihB_hgsPMMNHX-H/view?usp=sharing) and download the images from [official website](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

After downloading the data, place it in the ./data folder.

### Training

For shallow alignment

```bash
bash scripts/4-1.shallow_run.sh
```

For delta alignment

```bash
bash scripts/5-1.delta_run.sh
```

For deep alignment

```bash
bash scripts/6-1.deep_run.sh
```

### Testing (For MIMIC-CXR)
You can download our pretrained Delta checkpoints for [Here](https://drive.google.com/drive/folders/1ywEITWfYIAAYy0VY1IZ24Ec_GoNmkqIY?usp=sharing)

For shallow alignment

```bash
bash scripts/4-2.shallow_test.sh
```

For delta alignment

```bash
bash scripts/5-2.delta_test.sh
```

For deep alignment

```bash
bash scripts/6-2.shallow_test.sh
```


## Acknowledgement

+ [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) Some codes of this repo are based on MiniGPT-4.
+ [Llama2](https://github.com/facebookresearch/llama) The fantastic language ability of Llama-2 with only 7B parameters is just amazing.


## License
This repository is under [BSD 3-Clause License](LICENSE.md).

```
python -u train.py --batch_size 8 --val_batch_size 12 --freeze_vm True --vis_use_lora False --learning_rate 1e-4 --gradient_clip_val 1 --max_length 100 --min_new_tokens 80 --max_new_tokens 120 --repetition_penalty 2.0 --length_penalty 2.0  --num_workers 8 --devices 4 --max_epochs 5 --limit_val_batches 0.5 --val_check_interval 0.5 --num_sanity_val_steps 2 
```
## pretrain iu-xray
Pretrain clip
```nohup python train_clip.py --batch_size 152 --val_batch_size 152 --test_batch_size  152 --max_epochs 25 --pretraining true --pretrain_path pretrain/iu-xray/v1 --annotation "/data/wyh21/iu_xray/annotation.json" --dataset "iu-xray" --base_dir "/data/wyh21/iu_xray/images" --strategy ddp_find_unused_parameters_true &
``` 
## Train LLM on IU-Xray
```nohup python train.py --batch_size 8 --val_batch_size 8 --test_batch_size 8 --max_epochs 20  --pretrain_path pretrain/iu-xray/v1 --annotation "/data/wyh21/iu_xray/annotation.json" --dataset "iu-xray" --base_dir "/data/wyh21/iu_xray/images" --strategy ddp_find_unused_parameters_true &
``` 



          