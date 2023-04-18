# Style Transformer for Image Inversion and Editing (CVPR2022)


 [![arXiv](https://img.shields.io/badge/paper-CVPR2022-green)](https://openaccess.thecvf.com/content/CVPR2022/html/Hu_Style_Transformer_for_Image_Inversion_and_Editing_CVPR_2022_paper.html) [![arXiv](https://img.shields.io/badge/arXiv-2203.07932-blue)](https://arxiv.org/abs/2203.07932) [![video](https://img.shields.io/badge/video-YouTube-red)](https://youtu.be/5VL2yYCgByQ)

 Updated by @yoojin

## Docker Image



## Getting Started
### Prerequisites
- Ubuntu 16.04
- NVIDIA GPU + CUDA CuDNN
- Python 3

## Pretrained Models
We provide the pre-trained models of inversion for face and car domains.
- [FFHQ Inversion Model](https://drive.google.com/file/d/1XJWP712o-wWZrfzXJ07vc3dHjJF8CanT/view?usp=sharing)
- [Stanford Cars Inversion Model](https://drive.google.com/file/d/1ri10_CWq42IzzIQ4ZQAxNX7BLsCztd92/view?usp=sharing)

## Training
### Preparing Datasets
Update `configs/paths_config.py` with the necessary data paths and model paths for training and inference.
```
dataset_paths = {
    'train_data': '/path/to/train/data'
    'test_data': '/path/to/test/data',
}
```
### Preparing Generator and Encoder
We use rosinality's [StyleGAN2 implementation](https://github.com/rosinality/stylegan2-pytorch).
You can download the 256px pretrained model in the project and put it in the directory `/pretrained_models`.

Moreover, following pSp, we use some pretrained models to initialize the encoder and for the ID loss, you can download them from [here](https://github.com/eladrich/pixel2style2pixel) and put it in the directory `/pretrained_models`.

### Training Inversion Model
```
python scripts/train.py \
--dataset_type=ffhq_encode \
--exp_dir=results/train_style_transformer \
--batch_size=8 \
--test_batch_size=8 \
--val_interval=5000 \
--save_interval=10000 \
--stylegan_weights=pretrained_models/stylegan2-ffhq-config-f.pt
```

## Inference
```
python scripts/inference.py \
--exp_dir=results/infer_style_transformer \
--checkpoint_path=results/train_style_transformer/checkpoints/best_model.pt \
--data_path=/test_data \
--test_batch_size=8 \
```

## Citation
If you use this code for your research, please cite
```
@inproceedings{hu2022style,
  title={Style Transformer for Image Inversion and Editing},
  author={Hu, Xueqi and Huang, Qiusheng and Shi, Zhengyi and Li, Siyuan and Gao, Changxin and Sun, Li and Li, Qingli},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11337--11346},
  year={2022}
}
```
