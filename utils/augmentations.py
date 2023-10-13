import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image
from conf import *


class RandomScaleTransform:
    def __init__(self, min_scale=0.1, max_scale=2, sub_image_size=SUB_IMAGE_SIZE):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.sub_image_size = sub_image_size

    def __call__(self, image):
        scale_factor = random.uniform(self.min_scale, self.max_scale)
        new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
        image = transforms.Resize(new_size)(image)  # scale
        image = transforms.Resize(self.sub_image_size)(image)  # resize
        return image


class CircularTranslateTransform:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, image):
        # Ensure the input is a PIL Image
        if not isinstance(image, Image.Image):
            raise TypeError("Input should be a PIL Image.")

        # Convert PIL Image to numpy array
        image_np = np.array(image)

        # Randomly select shift value between L/2 and L/10
        shift_value = random.randint(self.image_size // 10, self.image_size // 2)

        # Randomly decide whether to shift horizontally or vertically
        axis = random.choice([0, 1])

        # Circular shift
        image_np = np.roll(image_np, shift=shift_value, axis=axis)

        # Convert back to PIL Image
        return Image.fromarray(image_np)


class RandomRotationTransform:
    def __init__(self, angles=None):
        if angles is None:
            angles = [90, 180, 270]
        self.angles = angles

    def __call__(self, image):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(image, angle)
