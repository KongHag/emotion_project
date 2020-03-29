

from emotion import run
import sys
from log import setup_custom_logger
from dataset import MediaEval18

logger = setup_custom_logger("schedule_train")
logger.setLevel(20)

config = {
    'seq_len': 64,
    'num_hidden': 1,
    'hidden_size': 2,
    'lr': 0.0001,
    'batch_size': 64,
    'grad_clip': 1,
    'nb_epoch': 25,
    'optimizer': 'RMSprop',
    'crit': 'MSE',
    'weight_decay': 1e-5,
    'bidirect': True,
    'dropout': 0.3,
    'logger_level': 20,
    'fragment': 1,
    'features': 'cl',
    'overlapping': False,
    'model': 'FC'
}

try:
    run(config)
except Exception as exception:
    logger.critical(sys.exc_info())
"""
features = MediaEval18._features_len.keys()
for feature in features:
    print(feature)
    config['features'] = feature
    try:
        run(config)
    except Exception as exception:
        logger.critical(sys.exc_info())
"""
