# dataset.py
import torch
import os
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset

CLASS_MAPPING = {
    'awake': 1,
    'cry': 2,
    'sleep': 3
}
REV_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

class BabyDataset(Dataset):
    """
    Custom PyTorch Dataset for loading baby cry/sleep/awake data.
    - Reads images and corresponding YOLO format labels.
    - Converts them to the format required by torchvision's Faster R-CNN model.
    - Filters out images that do not have any annotations.
    """
    def __init__(self, dir_path, width, height, transform=None):
        self.transform = transform
        self.dir_path = dir_path
        self.height = height
        self.width = width
        all_image_paths = glob.glob(f"{self.dir_path}/images/*.jpg")
        all_image_paths.extend(glob.glob(f"{self.dir_path}/images/*.jpeg"))
        all_image_paths.extend(glob.glob(f"{self.dir_path}/images/*.png"))
        
        self.image_paths = []

        print(f"Checking {len(all_image_paths)} images in '{dir_path}' for annotations...")
        
        for img_path in all_image_paths:
            label_path = img_path.replace("images", "labels").rsplit('.', 1)[0] + '.txt'
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                self.image_paths.append(img_path)
        
        print(f"--> Dataset Initialized: Found {len(self.image_paths)} images with annotations.")

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # Read image
        image = cv2.imread(image_path)
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0 # Normalize to [0, 1]
        
        # Convert the image from a NumPy array (H, W, C) to a PyTorch tensor (C, H, W)
        image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1))

        # Get the corresponding label path
        label_path = image_path.replace("images", "labels").rsplit('.', 1)[0] + '.txt'
        
        boxes = []
        labels = []
        
        with open(label_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                class_id = int(values[0])
                x_center, y_center, w, h = map(float, values[1:])
                x_min = (x_center - w / 2) * self.width
                y_min = (y_center - h / 2) * self.height
                x_max = (x_center + w / 2) * self.width
                y_max = (y_center + h / 2) * self.height
                
                boxes.append([x_min, y_min, x_max, y_max])
                # Add 1 to class ID because 0 is reserved for the background class
                labels.append(class_id + 1) 

        # Convert box and label lists to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create the target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        # Apply transformations if any
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor, target

    def __len__(self):
        return len(self.image_paths)