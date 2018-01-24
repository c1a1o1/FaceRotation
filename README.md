# [Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis](https://arxiv.org/abs/1704.04086)
# * Still under Development (2018/01/24)

- authors: Rui Huang, Shu Zhang, Tianyu Li, Ran He
- CVPR2017: https://arxiv.org/pdf/1704.04086.pdf
- Pytorch implimentation of TP-GAN

## Requirements
- python 3.6
- pytorch 0.2.0
- numpy 1.13.1
- scipy 0.18.1
- matplotlib 2.0.0
- Python Imaging Library (PIL)

## How to use

### Single-network Generator
1. Train LightCNN
    - modify class ImageList at util.py for applicable to your mechine configuration(directly storing it in memory or not).
    - data format in train list: [image path relative to your data root] [ID]
    - I recommend you to see the source code at util.py.
	> python train_light_cnn.py --root_path=/path/to/your/datasets/ \
		--train_list=/path/to/your/train/list.txt \
		--save_path=/path/to/your/save/path/ \
		--model="LightCNN-9/LightCNN-29" --num_classes=n

2. Train TP-GAN with single network
	- modify class MyDatasets at util.py for applicable to your mechine configuration(directly storing it in memory or not).
    - data format in train list: [profile face image path relative to your data root] [frontal face image path relative to your data root] [ID]
    - you need to provide the checkpoint of the lightcnn model which is fine-tuned by you datasets.
	> python train_tp_gan.py ---root_path=/path/to/your/datasets/ \
		--train_list=/path/to/your/train/list.txt \
		--checkpoint=/path/to/your/save/path/ --num_classes=n
-

3. Generate Image with trained model
      - provide the absolute path of profile face
      - generated images will be saved at specified position
      > python generate_image.py --test_list=/path/to/your/eval/list.txt \
		--save_path=/path/to/your/save/path/

### Double-network Generator(under development)
