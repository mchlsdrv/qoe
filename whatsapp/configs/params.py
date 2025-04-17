import datetime
import pathlib
import torch

# -----------
# - GENERAL -
# -----------
EPSILON = 1e-8
RANDOM_SEED = 0

ROOT_DIR = pathlib.Path('./')
OUTPUT_DIR = ROOT_DIR / 'outputs'

DESCRIPTION = f'auto_encoder'
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# --------
# - DATA -
# --------
DATA_ROOT_DIR = pathlib.Path('./data')

CV_10_DATA_ROOT_DIR = DATA_ROOT_DIR / 'cv_10_folds'

TRAIN_DATA_FILE = DATA_ROOT_DIR / 'train_test/train_data.csv'
TEST_DATA_FILE = DATA_ROOT_DIR / 'train_test/test_data.csv'

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
EPOCHS = 50
BATCH_SIZE = 64
VAL_PROP = 0.2
OPTIMIZER = torch.optim.Adam
LOSS_FUNCTION = torch.nn.MSELoss()
LAYER_ACTIVATION = torch.nn.SiLU
LR = 1e-3
LR_REDUCTION_FREQ = 100
LR_REDUCTION_FCTR = 0.5
MOMENTUM = 0.5
WEIGHT_DECAY = 1e-5
DROPOUT_START = 100
DROPOUT_DELTA = 20
DROPOUT_P = 0.0
DROPOUT_P_MAX = 0.5
RBM_K_GIBBS_STEPS = 10
