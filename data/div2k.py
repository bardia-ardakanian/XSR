import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from conf import *


class DIV2KLoader(Dataset):
    def __init__(self, div2k_path=DIV2K_ROOT, transform=None):
        self.div2k_path = div2k_path
        self.image_names = os.listdir(div2k_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.div2k_path, image_name)
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image


class DIV2KDataset(Dataset):
    def __init__(self, data, image_info):
        """
        Args:
            data (list): A list of batches, where each batch is a list of (ridx, lidx, image, label) tuples.
            image_info (dict): A dictionary containing additional information about the images, 
                               indexed by their ID.
        """
        self.data = data
        self.image_info = image_info

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch = self.data[idx]
        ridxs, lidxs, images, labels = zip(*batch)  # Separate ridxs, lidxs, images, and labels
        images = torch.stack(images, dim=0)  # Add a batch dimension
        labels = torch.tensor(labels)  # Convert labels to a tensor
        return ridxs, lidxs, images, labels

    def get_pair_info(self, ridx, lidx):
        """
        Retrieve the additional information for a pair of images based on their indices.

        Args:
            ridx (int): The ID of the right concatenated image.
            lidx (int): The ID of the left concatenated image.

        Returns:
            tuple: A tuple containing the additional information for the right and left images.
        """
        right_image_info = self.image_info[ridx]
        left_image_info = self.image_info[lidx]
        return right_image_info, left_image_info

    def get_image_info(self, idx):
        """
        Retrieve the additional information for an image based on their indices.

        Args:
            idx (int): The ID of the image.

        Returns:
            tuple: A tuple containing the additional information for the image.
        """
        return self.image_info[idx]
