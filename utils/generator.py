import random
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from utils.augmentations import *


def generate_sub_images(dataset, num_images, image_size):
    batch = []

    for _ in range(num_images):
        # Select a random image
        idx = random.randint(0, len(dataset) - 1)
        image = dataset[idx]

        # Convert tensor image back to PIL for cropping
        pil_image = transforms.ToPILImage()(image)

        # Get random coordinates for the top-left corner of the sub-image
        max_x = pil_image.width - image_size[0]
        max_y = pil_image.height - image_size[1]
        start_x = random.randint(0, max_x)
        start_y = random.randint(0, max_y)

        # Extract the sub-image
        sub_image = pil_image.crop((start_x, start_y, start_x + image_size[0], start_y + image_size[1]))

        # Convert sub-image back to tensor
        sub_image = transforms.ToTensor()(sub_image)

        # Add the ground truth, sub-image, and sub-image location to the batch
        batch.append((idx, image, sub_image, (start_x, start_y)))

    return batch


def generate_transformed_sub_images(batch, transform):
    # Apply the transformations to each image in the batch
    transformed_batch = []

    # Define a transform to convert a tensor to a PIL Image
    to_pil_image = transforms.ToPILImage()

    for idx, ground_truth, sub_image, location in batch:
        # Convert the tensor to a PIL Image
        pil_sub_image = to_pil_image(sub_image)

        # Apply transformations to the sub_image
        transformed_sub_image = transform(pil_sub_image)

        # Add the transformed sub_image and other details to the new batch
        transformed_batch.append((idx, ground_truth, sub_image, transformed_sub_image, location))

    return transformed_batch


def generate_dataset(div2k_dataset, num_batches, num_images, image_size):
    """
    Generate a dataset containing original and transformed batches.

    Parameters:
        div2k_dataset: The original dataset.
        num_batches (int): Number of batches to generate.
        num_images (int): Number of images per batch.
        image_size (int): Size of the sub-images.

    Returns:
        list: A dataset containing tuples of (batch, transformed_batch).
    """
    # Define the transformation
    target_transform = transforms.Compose([
        transforms.RandomChoice([
            RandomScaleTransform(),
            RandomRotationTransform(),  # Rotate 90, 180, or 270 degrees
            CircularTranslateTransform(image_size=L),  # Shift by a random value between L/2 and L/10
        ]),
        transforms.ToTensor()
    ])

    # Generate dataset
    dataset = []
    dataset_info = []

    for _ in tqdm(range(num_batches), desc='Generating batches'):
        # Generate sub-images
        sub_images = generate_sub_images(dataset=div2k_dataset, num_images=num_images, image_size=image_size)

        # Generate transformed sub-images
        transformed_sub_images = generate_transformed_sub_images(sub_images, target_transform)

        # Store the additional information
        dataset_info.extend(transformed_sub_images)

        # Extracting batch
        batch = [item[2] for item in transformed_sub_images]
        # Extracting transformed batch
        transformed_batch = [item[3] for item in transformed_sub_images]

        half_batch_size = len(batch) // 2
        labels = []
        used_pairs = set()

        # Positive samples
        for _ in range(half_batch_size):
            i = random.randint(0, len(batch) - 1)
            while (i, i) in used_pairs:  # Ensure (i, i) has not been used
                i = random.randint(0, len(batch) - 1)

            img_i = batch[i]
            concatenated_img_pos = torch.cat((img_i, transformed_batch[i]),
                                             dim=0)  # Assuming images are in [C, H, W] format
            labels.append((i, i, concatenated_img_pos, 1))  # batchid, transformedid, concatimg, label
            used_pairs.add((i, i))

        # Negative samples
        for _ in range(half_batch_size):
            i = random.randint(0, len(batch) - 1)
            j = random.randint(0, len(transformed_batch) - 1)
            while i == j or (i, j) in used_pairs or (
                    j, i) in used_pairs:  # Ensure i and j are not equal and (i, j) has not been used
                i = random.randint(0, len(batch) - 1)
                j = random.randint(0, len(transformed_batch) - 1)

            img_i = batch[i]
            img_j = transformed_batch[j]
            concatenated_img_neg = torch.cat((img_i, img_j), dim=0)  # Assuming images are in [C, H, W] format
            labels.append((i, j, concatenated_img_neg, 0))  # batchid, transformedid, concatimg, label
            used_pairs.add((i, j))
            used_pairs.add((j, i))  # Adding (j, i) as well to ensure no repetition in reverse order

        random.shuffle(labels)

        # Append the batch to the dataset
        dataset.append(labels)

    return dataset, dataset_info
