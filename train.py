import argparse
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn as nn

import eval
from data.div2k import *
from eval import eval
from net.net import build_net
from utils.generator import generate_dataset
from utils.plotter import *


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='XAI SR Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset_root', default=DIV2K_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--dataset_eval_root', default=DIV2K_VAL,
                    help='Dataset validation root directory path')
parser.add_argument('--batch_size', default=BATCH_SIZE, type=int,
                    help='Batch size for training')
parser.add_argument('--num_batches', default=BATCH_SIZE, type=int,
                    help='Number of batches for training')
parser.add_argument('--R', default=R, type=int,
                    help='Images H&W')
parser.add_argument('--L', default=L, type=int,
                    help='Sub-images H&W')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in data loading')
parser.add_argument('--cuda', default=False, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=LR, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--tensorboard', default=True, type=str2bool,
                    help='Use tensorboard for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():

    # Check if CUDA is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')

    try:
        # Define transformations
        transform = transforms.Compose([
            #     transforms.Resize(image_size),
            transforms.ToTensor()
        ])

        # Load the dataset
        div2k_dataset = DIV2KLoader(div2k_path=args.dataset_root, transform=transform)

        data, data_info = generate_dataset(div2k_dataset=div2k_dataset, num_batches=NUM_BATCHES,
                                                    num_images=BATCH_SIZE, image_size=SUB_IMAGE_SIZE)

        div2k_eval_dataset = DIV2KLoader(div2k_path=args.dataset_val_root, transform=transform)

        eval_data, eval_data_info = generate_dataset(div2k_dataset=div2k_eval_dataset, num_batches=NUM_EVAL_BATCHES,
                                                     num_images=BATCH_SIZE, image_size=SUB_IMAGE_SIZE)
    except:
        raise Exception('Dataset root is invalid')

    if args.tensorboard:
        from datetime import datetime
        from torch.utils.tensorboard import SummaryWriter
        run_name = f'{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}'
        writer = SummaryWriter(os.path.join('runs', 'XSR', 'tensorboard', run_name))

    net = build_net()

    vgg16 = net.features.eval()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    if args.cuda:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        net = net.cuda()

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        net.load_weights(args.resume)

    net.train()

    # Initialize the div2k dataset and eval
    div2k_dataset = DIV2KDataset(data, data_info)
    # div2k_eval = DIV2KDataset(eval_data, eval_data_info)

    # Initialize the data loader
    data_loader = DataLoader(div2k_dataset, batch_size=1, shuffle=True)
    # eval_data_loader = DataLoader(div2k_eval_dataset, batch_size=1, shuffle=True)
    eval_data_loader = data_loader

    # Testing dataset performance
    # ----------------------------------------------------------------------------------------------------
    # Retrieve additional information
    # ridx, lidx = 0, 1  # Replace with the IDs of the images you want
    # right_image_info, left_image_info = div2k_dataset.get_pair_info(ridx, lidx)
    #
    # plot_image(right_image_info)
    # plot_image(left_image_info)
    #
    # # Extract a batch of images and labels
    # lidx, ridx, images, labels = div2k_dataset[0]  # Get the first batch
    #
    # # Plot the images and labels
    # plot_images_with_labels(lidx, ridx, images, labels, num_images=BATCH_SIZE)
    # ----------------------------------------------------------------------------------------------------

    # Training
    # Arrays to store losses
    total_losses = []
    local_losses = []

    # Training loop
    for epoch in range(NUM_EPOCHS):
        net.train()
        epoch_loss = 0.0

        # Enumerate through the DataLoader
        for j, (lidx, ridx, images, labels) in enumerate(data_loader):

            # Ensure inputs and labels are torch tensors and send them to the device
            images = images.squeeze(0).to(device)
            labels = labels.squeeze().to(device).float()  # Ensure labels are float type

            # Forward pass
            outputs = net(images)  # No need to squeeze batch dimension
            loss = criterion(outputs.squeeze(), labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            local_losses.append(loss.item())

            # Print and log local loss
            if args.tensorboard and (j + 1) % ITER_UPDATE == 0:
                update_local_loss(data_loader, epoch, j, local_losses, writer)
                local_losses = []  # Reset local losses

            # Evaluation
            if (j + 1) % ITER_EVAL == 0:
                eval(net, eval_data_loader, criterion, writer, epoch * len(data_loader) + j, device)

            # Save weights
            if (j + 1) % ITER_SAVE == 0:
                torch.save(net.state_dict(), f'weights/weights_iter_{j}.pth')

        # Log and print total loss after each epoch
        update_total_loss(data_loader, epoch, epoch_loss, total_losses, writer)

        # Save weights
        if (epoch + 1) % EPOCH_SAVE == 0:
            eval(net, eval_data_loader, criterion, writer, epoch * len(data_loader) + j, device)
            torch.save(net.state_dict(), f'weights/weights_epoch_{epoch}.pth')

    # Finish training and save weights
    eval(net, eval_data_loader, criterion, writer, epoch * len(data_loader) + j, device)
    torch.save(net.state_dict(), f'weights/weights_epoch_{epoch}.pth')

    # Close the TensorBoard writer
    writer.close()


def update_total_loss(data_loader, epoch, epoch_loss, total_losses, writer):
    total_losses.append(epoch_loss / len(data_loader))
    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Total Loss: {total_losses[-1]:.4f}')
    writer.add_scalar('training loss (total)', total_losses[-1], epoch)


def update_local_loss(data_loader, epoch, j, local_losses, writer):
    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{j + 1}/{len(data_loader)}], Local Loss: {sum(local_losses) / len(local_losses):.4f}')
    writer.add_scalar('training loss (local)', sum(local_losses) / len(local_losses), epoch * len(data_loader) + j)


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    # training loop
    train()
