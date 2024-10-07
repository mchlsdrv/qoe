import datetime
import pathlib
import torch

# -----------
# - GENERAL -
# -----------
EPSILON = 1e-8
DESCRIPTION = f'auto_encoder_4x128'
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# --------
# - DATA -
# --------
DATA_ROOT_DIR = pathlib.Path('./data')

CV_10_DATA_ROOT_DIR = DATA_ROOT_DIR / 'cv_10_folds'

TRAIN_DATA_FILE = DATA_ROOT_DIR / 'train_test/train_data.csv'
TEST_DATA_FILE = DATA_ROOT_DIR / 'train_test/test_data.csv'

ROOT_DIR = pathlib.Path('/Users/mchlsdrv/Desktop/BGU/PhD/QoE/Code/qoe')
OUTPUT_DIR = ROOT_DIR / 'outputs'

FEATURES = ['Bandwidth', 'pps', 'packets length', 'avg time between packets']
# FEATURES = ['Bandwidth', 'pps', 'Jitter', 'packets length', 'Interval start', 'Latency', 'avg time between packets']
LABELS = ['NIQE']
# LABELS = ['NIQE', 'Resolution', 'fps']
LABEL = 'fps'
OUTLIER_TH = 2

# ----------------
# - ARCHITECTURE -
# ----------------
N_LAYERS = 32
N_UNITS = 256
RBM_VISIBLE_UNITS = 784  # 28 X 28 IMAGES
RBM_HIDDEN_UNITS = 128
AUTO_ENCODER_CODE_LENGTH_PROPORTION = 32

# ------------
# - TRAINING -
# ------------
EPOCHS = 100
BATCH_SIZE = 64
VAL_PROP = 0.2
OPTIMIZER = torch.optim.Adam
LOSS_FUNCTION = torch.nn.MSELoss()
LAYER_ACTIVATION = torch.nn.SiLU
LR = 1e-3
LR_REDUCTION_FREQ = 50
LR_REDUCTION_FCTR = 0.5
MOMENTUM = 0.5
WEIGHT_DECAY = 1e-5
DROPOUT_START = 10
DROPOUT_DELTA = 20
DROPOUT_P = 0.1
RBM_K_GIBBS_STEPS = 10
