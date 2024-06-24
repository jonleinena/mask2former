import argparse
import numpy as np
import torch
from datasets import Dataset, DatasetDict, Image
import os
from Dataset import ImageSegmentationDataset, get_training_transformation, get_validation_augmentation
from torch.utils.data import DataLoader
import json

from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from tqdm.auto import tqdm
import evaluate
from functools import partial


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default = '/Users/jonleinena/Desktop/TFM/mlops-basics/data', help="Path to the dataset")
    parser.add_argument('--img_size',type=int, default=[1024, 1024],nargs='+',help=("The size for input images, all the images in the train/validation dataset will be resized to this size. [height, width]"))
    parser.add_argument('--model_name_or_path', type=str, default='facebook/mask2former-swin-small-ade-semantic')
    parser.add_argument('--output_path', default='weights', type=str, help="Path to save model weights & checkpoints")
    parser.add_argument('--learning_rate', type=float, help='Initial learning rate for training process')
    parser.add_argument('--epochs', type=int, default=2, help="Number of epochs to go through in the training loop")
    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility of training')
    parser.add_argument('--optimizer', type=str, default="AdamW", help="Optimizer")
    parser.add_argument('--classes', type=str,  default=['background', 'patches', 'inclusion', 'scratches'], nargs='+', help="classes to segment. Each pixel-value is assigned to the class value index")
    opt = parser.parse_args()

    return opt
    

def create_dataset(image_paths, label_paths):
    dataset = Dataset.from_dict({"image": sorted(image_paths),
                                "label": sorted(label_paths)})
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image())

    return dataset

def id2label(dataset_path, classes):
    """
    Function to read the class labels from the id2label json file of the dataset.

    Args:
        - dataset_path: system path where the dataset files are located
        - classes: classes array from program arguments. Used in case there is no id2label.json file. Creates the file and assigns each class label it's idx as pixel-value.
    Return:
        - id2label: dictionary mapping pixel-value to class label
    """

    if not os.path.exists(os.path.join(dataset_path, 'id2label.json')):
        print("No json found. Creating one with database classes")
        with open(os.path.join(dataset_path, 'id2label.json'), 'w') as f:
            labels_json = {idx: class_name for idx, class_name in enumerate(classes)}
            json.dump(labels_json, f)
            return labels_json
    else:
        with open(os.path.join(dataset_path, 'id2label.json'), 'r') as f:
            labels_json = json.load(f)
            labels_json = {int(k):v for k,v in labels_json.items()}
            return labels_json


def collate_fn(batch, preprocessor):

    inputs = list(zip(*batch))
    images = inputs[0]
    segmentation_maps = inputs[1]
    # this function pads the inputs to the same size,
    # and creates a pixel mask
    # actually padding isn't required here since we are cropping
    batch = preprocessor(
        images,
        segmentation_maps=segmentation_maps,
        return_tensors="pt",
    )

    batch["original_images"] = inputs[2]
    batch["original_segmentation_maps"] = inputs[3]
    
    return batch

def train():
    print("Starting training")
    config = parse_arguments()

    train_imgs = [os.path.join(config.dataset_path,'train', 'imgs', img) for img in os.listdir(os.path.join(config.dataset_path,'train', 'imgs')) if not img.startswith('.') and not 'humbs' in img]
    train_masks = [os.path.join(config.dataset_path, 'train', 'masks', mask) for mask in os.listdir(os.path.join(config.dataset_path, 'train', 'masks')) if not mask.startswith('.') and not 'humbs' in mask]

    validation_imgs = [os.path.join(config.dataset_path, 'validation','imgs', img) for img in os.listdir(os.path.join(config.dataset_path, 'validation','imgs')) if not img.startswith('.') and not 'humbs' in img]
    validation_masks = [os.path.join(config.dataset_path, 'validation','masks', mask) for mask in os.listdir(os.path.join(config.dataset_path, 'validation','masks')) if not mask.startswith('.') and not 'humbs' in mask]

    train_dataset = create_dataset(train_imgs, train_masks)
    validation_dataset = create_dataset(validation_imgs, validation_masks)

    dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
        }
    )

    train_dataset = ImageSegmentationDataset(dataset['train'], transform=get_training_transformation(config.img_size[0], config.img_size[1]))
    validation_dataset = ImageSegmentationDataset(dataset['validation'], transform=get_validation_augmentation(config.img_size[0], config.img_size[1]))
    
    preprocessor = Mask2FormerImageProcessor(ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)
    collate_with_preprocessor = partial(collate_fn, preprocessor=preprocessor)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_with_preprocessor)
    test_dataloader = DataLoader(validation_dataset, batch_size=2, shuffle=False, collate_fn=collate_with_preprocessor)

    
    labels_json = id2label(config.dataset_path, config.classes)


    model = Mask2FormerForUniversalSegmentation.from_pretrained(config.model_name_or_path, id2label=labels_json, ignore_mismatched_sizes=True)

    metric = evaluate.load('mean_iou')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5) #TODO parametrize this

    running_loss = 0.0
    num_samples = 0
    for epoch in range(config.epochs):
        print("Epoch:", epoch)
        model.train()
        for idx, batch in enumerate(tqdm(train_dataloader)):
            # Reset the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )

            # Backward propagation
            loss = outputs.loss
            loss.backward()

            if loss < running_loss:
                model.save_pretrained(config.output_path)

            batch_size = batch["pixel_values"].size(0)
            running_loss += loss.item()
            num_samples += batch_size

            if idx % 100 == 0:
                print("Loss:", running_loss/num_samples)

            # Optimization
            optimizer.step()

        model.eval()
        for idx, batch in enumerate(tqdm(test_dataloader)):
            if idx > 5:
                break

            pixel_values = batch["pixel_values"]
            
            # Forward pass
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values.to(device))

            # get original images
            original_images = batch["original_images"]
            target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
            # predict segmentation maps
            predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs,
                                                                                        target_sizes=target_sizes)

            # get ground truth segmentation maps
            ground_truth_segmentation_maps = batch["original_segmentation_maps"]

            metric.add_batch(references=ground_truth_segmentation_maps, predictions=predicted_segmentation_maps)
        
        # NOTE this metric outputs a dict that also includes the mIoU per category as keys
        # so if you're interested, feel free to print them as well
        print("Mean IoU:", metric.compute(num_labels = len(labels_json), ignore_index = 0)['mean_iou'])

if __name__ == '__main__':
    train()