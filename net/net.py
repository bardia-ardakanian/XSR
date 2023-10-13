import torch
import torch.nn as nn
from torchvision import models, transforms
from conf import *


def build_net():
    # adapting the first and last layers of VGG16 to our desired input and output
    model = models.vgg16(pretrained=False)

    # changing the input channels from 3 to 6
    model.features[0] = nn.Conv2d(NUM_CHANNELS, L, kernel_size=KERNEL, stride=STRIDE, padding=PADDING)
    model.features[1] = nn.Conv2d(L, 64, kernel_size=KERNEL, stride=STRIDE, padding=PADDING)

    # changing the output to a 0 to 1 scale score
    model.classifier[-1] = nn.Sequential(
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(1024),  # Adding BatchNorm1d layer
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),  # Adding BatchNorm1d layer
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(128),  # Adding BatchNorm1d layer
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )

    # Move the model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(model.features)
    print(model.classifier)

    return model
