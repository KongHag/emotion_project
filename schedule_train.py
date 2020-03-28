

from emotion import run
import sys
from log import setup_custom_logger

logger = setup_custom_logger("schedule_train")

config = {
    'seq_len': 64,
    'num_hidden': 1,
    'hidden_size': 512,
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
    'fragment': 0.001,
    'features': 'all',
    'overlapping': True,
    'model': 'FC'
}

for num_hidden in [0, 1, 2, 3]:
    config['num_hidden'] = num_hidden
    for hidden_size in [32, 128, 512]:
        config['hidden_size'] = hidden_size
        try:
            run(config)
        except Exception as exception:
            logger.critical(sys.exc_info())
