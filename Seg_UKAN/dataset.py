import os

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    # def __getitem__(self, idx):
    #     img_id = self.img_ids[idx]
        
    #     img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

    #     mask = []
    #     for i in range(self.num_classes):

    #         # print(os.path.join(self.mask_dir, str(i),
    #         #             img_id + self.mask_ext))

    #         mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
    #                     img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
    #     mask = np.dstack(mask)

    #     if self.transform is not None:
    #         augmented = self.transform(image=img, mask=mask)
    #         img = augmented['image']
    #         mask = augmented['mask']
        
    #     img = img.astype('float32') / 255
    #     img = img.transpose(2, 0, 1)
    #     mask = mask.astype('float32') / 255
    #     mask = mask.transpose(2, 0, 1)

    #     if mask.max()<1:
    #         mask[mask>0] = 1.0

    #     return img, mask, {'img_id': img_id}


def __getitem__(self, idx):
    img_id = self.img_ids[idx]
    
    # Load image
    img_path = os.path.join(self.img_dir, img_id + self.img_ext)
    img = cv2.imread(img_path)
    
    if img is None:
        raise FileNotFoundError(f"Error: Image file not found or unreadable: {img_path}")

    mask = []
    for i in range(self.num_classes):
        mask_path = os.path.join(self.mask_dir, str(i), img_id + self.mask_ext)
        
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask_img is None:
            print(f"Warning: Mask file missing or unreadable: {mask_path}")
            mask_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)  # Create blank mask
        
        mask.append(mask_img[..., None])  # Add extra channel dimension

    mask = np.dstack(mask)

    # Apply Transformations (if any)
    if self.transform is not None:
        augmented = self.transform(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
    
    # Normalize Image
    img = img.astype('float32') / 255
    img = img.transpose(2, 0, 1)  # Convert HWC to CHW

    # Normalize Mask
    mask = mask.astype('float32') / 255
    mask = mask.transpose(2, 0, 1)  # Convert HWC to CHW
    
    # Ensure Binary Mask
    mask[mask > 0] = 1.0  

    return img, mask, {'img_id': img_id}

