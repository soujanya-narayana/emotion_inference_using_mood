import os
import datetime
from pathlib import Path
import numpy as np
import pickle
import torch
import pandas as pd
import torch.optim as optim
from train import train_test
from utils import VisdomLinePlotter, get_logger, load_config, dump_config

torch.manual_seed(5)

# Load config file

cfg = load_config(str(Path(__file__).parent.joinpath('config.yaml')))

save_dir = os.path.join("experiments", datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + cfg['EXPERIMENT_NAME'])
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
global plotter
plotter = VisdomLinePlotter(env_name=os.path.basename(os.path.normpath(save_dir)), save_dir=save_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = cfg['DEVICE']
logger = get_logger(os.path.join(save_dir, 'everything.log'))

experiment_name = cfg['EXPERIMENT_NAME']  # name of the experiment. Will be used for saving model weights.

model_name = cfg['MODEL_NAME']  # model name obtained from get_model function

folds = cfg['FOLDS']
# img_dim = (cfg['IMAGE_SHAPE'], cfg['IMAGE_SHAPE'])
batch_size = cfg['BATCH_SIZE']
test_batch_size = cfg['TEST_BATCH_SIZE']
random_state = cfg['RANDOM_STATE']
learn_rate = cfg['LEARNING_RATE']
dropout_rate = cfg['DROPOUT_RATE']
log_interval = cfg['LOG_INTERVAL']
epochs = cfg['NUM_EPOCHS']
weight_decay = cfg['WEIGHT_DECAY']
num_workers = cfg['NUM_WORKERS']
lr_step_size = cfg['STEP_SIZE']
temporal_length = cfg['TEMPORAL_LENGTH']
temporal_step = cfg['TEMPORAL_STEP']
temporal_stride = cfg['TEMPORAL_STRIDE']

data_root = cfg['DATA_ROOT']
train_df = pd.read_csv(cfg['TRAIN_CSV'])
# train_df = train_df.head(500)
val_df = pd.read_csv(cfg['VAL_CSV'])
# val_df = val_df.head(500)
test_df = pd.read_csv(cfg['TEST_CSV'])
# test_df = test_df.head(500)
'''
logger.info(f'Loading train_data..')
with open(cfg['TRAIN_PICKLE'], 'rb') as f:
    train_data = pickle.load(f)
    logger.info(f'Train data: {len(train_data)}')

logger.info(f'Loading val_data..')
with open(cfg['VAL_PICKLE'], 'rb') as f:
    val_data = pickle.load(f)
    logger.info(f'Val data: {len(val_data)}')

logger.info(f'Loading test_data..')
with open(cfg['TEST_PICKLE'], 'rb') as f:
    test_data = pickle.load(f)
    logger.info(f'Test data: {len(test_data)}')
'''
final_dict = train_test(model_name, data_root, train_df, val_df, test_df, cfg, logger, device, plotter, save_dir)

pickle.dump(final_dict, open(os.path.join(save_dir, 'objects.p'), 'wb'))
logger.info(f'Test dict dumped at {os.path.join(save_dir, "objects.p")}')

dump_config(cfg, save_dir)
