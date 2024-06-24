# Vision Transformer Training and Inference

Welcome to the Vision Transformer Training and Inference repository! This project aims to provide training scripts for various pretrained vision transformers like Mask2Former and SegFormer. Additionally, we will implement different inference pipelines for these models.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Inference](#inference)
- [To-Do List](#to-do-list)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Vision transformers have revolutionized the field of computer vision by leveraging the power of transformers for image processing tasks. This repository provides scripts to train and perform inference using state-of-the-art vision transformers like Mask2Former and SegFormer.

## Features

- **Training Scripts**: Easily train vision transformers on your custom datasets.
- **Inference Pipelines**: Perform inference using trained models.
- **Customizable**: Modify training parameters and augmentations to suit your needs.
- **Preprocessing**: Includes image preprocessing and augmentation techniques.

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/vision-transformer-training.git
cd vision-transformer-training
pip install -r requirements.txt
```

## Usage

### Training

To train a model, use the `train.py` script. You can specify various parameters such as dataset path, image size, model name, and more.

```python train.py --dataset_path /path/to/dataset --img_size 1024 1024 --model_name_or_path facebook/mask2former-swin-small-ade-semantic --output_path weights --learning_rate 0.0001 --epochs 10
```

### Inference

Inference scripts will be added soon. Stay tuned!

## To-Do List

- [x] Implement training script for Mask2Former
- [ ] Implement training script for SegFormer
- [ ] Add inference pipeline for Mask2Former
- [ ] Add inference pipeline for SegFormer
- [ ] Add support for more vision transformers
- [ ] Improve documentation and add examples


## Contributing

It's still early stage so no contributions asked. Maybe will open up space for them in the future.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- [Hugging Face Transformers](https://github.com/huggingface/transformers)