import datetime
import pathlib
import torch

# -----------
# - GENERAL -
# -----------
EPSILON = 1e-9
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
DROPOUT_START = 50
DROPOUT_DELTA = 25
DROPOUT_P = 0.05
DROPOUT_P_MAX = 0.5
RBM_K_GIBBS_STEPS = 10

PACKET_SIZE_FEATURES = [
    'number_of_packet_sizes_in_time_window',
    'number_of_unique_packet_sizes_in_time_window',
    'min_packet_size',
    'max_packet_size',
    'mean_packet_size',
    'std_packet_size',
    'q1_packet_size',
    'q2_packet_size',
    'q3_packet_size',
]

PIAT_FEATURES = [
    'number_of_piats_in_time_window',
    'number_of_unique_piats_in_time_window',
    'min_piat',
    'max_piat',
    'mean_piat',
    'std_piat',
    'q1_piat',
    'q2_piat',
    'q3_piat',
]
