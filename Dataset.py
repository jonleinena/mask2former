import numpy as np
from torch.utils.data import Dataset

import albumentations as A

# The ADE_MEAN and ADE_STD variables are used in the albumentations Normalize function call to normalize the image data.
# The Normalize function subtracts the mean and divides by the standard deviation for each channel of the image.
# This process is a common preprocessing step in deep learning for images, as it helps to reduce the effect of different lighting conditions and other variations in the data.
# The values for ADE_MEAN and ADE_STD are derived from the ImageNet dataset and are commonly used for normalizing images in computer vision tasks.
ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, dataset, transform):
      """
      Args:
          dataset
      """
      self.dataset = dataset
      self.transform = transform
        
    def __len__(self):
      return len(self.dataset)
    
    def __getitem__(self, idx):
      original_image = np.array(self.dataset['image'][idx])
      original_segmentation_map = np.array(self.dataset['label'][idx])

      transformed = self.transform(image=original_image, mask=original_segmentation_map)
      image, segmentation_map = transformed['image'], transformed['mask']

      # convert to C, H, W
      image = image.transpose(2,0,1).astype('float32')

      return image, segmentation_map, original_image, original_segmentation_map



def get_training_transformation(img_height, img_width):
  train_transform = [
    A.LongestMaxSize(max_size=1333),
    A.RandomCrop(width=img_width, height=img_height),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=ADE_MEAN.tolist(), std=ADE_STD.tolist()),
  ]

  return A.Compose(train_transform)

def get_validation_augmentation(img_height, img_width):
  validation_transform = [
    A.Resize(width=512, height=512),
    A.Normalize(mean=ADE_MEAN.tolist(), std=ADE_STD.tolist()),
  ]
  
  return A.Compose(validation_transform)