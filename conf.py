# Dataset Conf
DIV2K_ROOT = 'data/div2k-dataset/DIV2K_train_HR'
DIV2K_VAL = 'data/div2k-dataset/DIV2K_valid_HR'
NUM_BATCHES = 20
BATCH_SIZE = 10
NUM_EVAL_BATCHES = 10000
# Image Conf
R = 224
IMAGE_SIZE = (R, R)
L = 48
SUB_IMAGE_SIZE = (L, L)
# Network Conf
NUM_EPOCHS = 1000
LR = 1e-4
NUM_CHANNELS = 6
KERNEL = (3, 3)
STRIDE = (1, 1)
PADDING = (1, 1)
# Evaluation parameters
EPOCH_SAVE = 50
ITER_SAVE = 1000
ITER_UPDATE = 3
ITER_EVAL = 1000
