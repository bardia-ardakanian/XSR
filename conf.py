# Dataset Conf
DATA_FOLDER = "data/generated_batches"
DIV2K_ROOT = 'data/div2k-dataset/DIV2K_train_HR'
DIV2K_VAL = 'data/div2k-dataset/DIV2K_valid_HR'
NUM_BATCHES = 4000
BATCH_SIZE = 16
NUM_EVAL_BATCHES = 500
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
ITER_SAVE = 12500
ITER_UPDATE = 1000
ITER_EVAL = 1000
