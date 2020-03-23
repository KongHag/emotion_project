from emotion import run, save_config_and_results
import logging
import sys
from dataset import MediaEval18

config = {
    'seq_len': 64,
    'num_hidden': 2,
    'hidden_size': 128,
    'lr': 0.00001,
    'batch_size': 16,
    'grad_clip': 1,
    'nb_epoch': 100,
    'optimizer': 'RMSprop',
    'crit': 'MSE',
    'weight_decay': 1e-7,
    'bidirect': True,
    'dropout': 0.5,
    'logger_level': 20,
    'fragment': 0.4,
    'features': 'all',
    'overlapping': False
}

# Parse args
logger = logging.getLogger()
logger.setLevel(config['logger_level'])

features = MediaEval18._features_len.keys()

for arg_name, arg in config.items():
    logger.info(
        "initialization -- {} - {}".format(arg_name, arg))

try:
    train_losses, test_losses = run(config)
    save_config_and_results(
        config, train_losses, test_losses)

except Exception as exception:
    logger.critical(sys.exc_info())
