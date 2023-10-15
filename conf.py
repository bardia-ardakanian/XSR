# Dataset Conf
DATA_FOLDER = "data/generated_batches"
DIV2K_ROOT = 'data/div2k-dataset/DIV2K_train_HR'
DIV2K_VAL = 'data/div2k-dataset/DIV2K_valid_HR'
NUM_BATCHES = 250
BATCH_SIZE = 10
NUM_EVAL_BATCHES = 100
# Image Conf
R = 224
IMAGE_SIZE = (R, R)
L = 96
SUB_IMAGE_SIZE = (L, L)
# Network Conf
NUM_EPOCHS = 2000
LR = 1e-4
NUM_CHANNELS = 6
KERNEL = (3, 3)
STRIDE = (1, 1)
PADDING = (1, 1)
# Evaluation parameters
EPOCH_SAVE = 100
ITER_SAVE = 100
ITER_UPDATE = 100
ITER_EVAL = 100
EPOCH_EVAL = 5
DATA_RENEWAL = 50
EVAL_RENEWAL = DATA_RENEWAL * 3

# Checkpoint
SAVE_ITER_FILE = f'weights/weights_net_{L}_iter'
SAVE_EPOCH_FILE = f'weights/weights_net_{L}_epoch'
SAVE_FINISH_FILE = f'weights/weights_final_net_{L}_epoch'
