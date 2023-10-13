import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision.transforms as transforms
from conf import *


def plot_image(data, sub_image_size=SUB_IMAGE_SIZE):
    """
    Plot the ground truth image, sub-image, and transformed sub-image.

    Args:
        data (tuple): A tuple containing (id, ground_truth, sub_image, transformed_sub_image, location).
        subimage_size (tuple): A tuple containing (width, height) of the sub-image.
        :param data: image and image descriptions
        :param sub_image_size: sub image size which is (L, L)
    """
    # Unpack the data tuple
    idx, ground_truth, sub_image, transformed_sub_image, location = data

    # Convert tensor images back to PIL for plotting
    pil_gt_image = transforms.ToPILImage()(ground_truth)
    pil_sub_image = transforms.ToPILImage()(sub_image)
    pil_transformed_image = transforms.ToPILImage()(transformed_sub_image)

    # Extract location info and subimage size
    start_x, start_y = location
    sub_image_width, sub_image_height = sub_image_size

    # Plot the original image, sub-image, and transformed sub-image
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    axs[0].imshow(pil_gt_image)
    axs[0].axis('off')
    rect = patches.Rectangle((start_x, start_y), sub_image_width, sub_image_height, linewidth=1, edgecolor='r',
                             facecolor='none')
    axs[0].add_patch(rect)
    axs[0].set_title(f'GT Image - ID: {idx}')

    axs[1].imshow(pil_sub_image)
    axs[1].axis('off')
    axs[1].set_title('Sub-image')

    axs[2].imshow(pil_transformed_image)
    axs[2].axis('off')
    axs[2].set_title('Transformed Sub-image')

    plt.show()


def plot_images_with_labels(lidx, ridx, images, labels, num_images):
    """
    Plots images along with their IDs and labels.
    
    Parameters:
        images (Tensor): A batch of images in [B, C, H, W] format.
        labels (Tensor): Corresponding labels in [B] format.
        lidx (list): List of left image IDs.
        ridx (list): List of right image IDs.
        num_images (int): Number of images to plot.
    """
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

    for i in range(num_images):
        image = images[i]
        label = labels[i]

        # If the image is a PyTorch tensor, convert it to a NumPy array
        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).cpu().numpy()  # CxHxW to HxWxC

        # Split the channels
        original_image = image[..., :3]
        transformed_image = image[..., 3:]

        # Check if the images are grayscale and remove the channel dimension if they are
        if original_image.shape[-1] == 1:
            original_image = original_image.squeeze(-1)
        if transformed_image.shape[-1] == 1:
            transformed_image = transformed_image.squeeze(-1)

        # Plot the original and transformed images side by side
        axes[i, 0].imshow(original_image)
        axes[i, 0].set_title(f'ID: {lidx[i]}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(transformed_image)
        axes[i, 1].set_title(f'ID: {ridx[i]}, Label: {label.item()}')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()
