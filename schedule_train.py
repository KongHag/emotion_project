from emotion import run, save_config_and_results
import logging
import sys

config = {
    'seq_len': 20,
    'num_hidden': 2,
    'hidden_size': 512,
    'lr': 0.001,
    'batch_size': 16,
    'grad_clip': 1,
    'nb_epoch': 50,
    'optimizer': 'RMSprop',
    'crit': 'MSE',
    'weight_decay': 1e-4,
    'bidirect': True,
    'dropout': 0.3,
    'logger_level': 20,
    'fragment': 1
}

# Parse args
logger = logging.getLogger()
logger.setLevel(config['logger_level'])


for arg_name, arg in config.items():
    logger.info(
        "initialization -- {} - {}".format(arg_name, arg))

try:
    train_losses, test_losses = run(config)
    save_config_and_results(
        config, train_losses, test_losses)

except Exception as exception:
    logger.critical(sys.exc_info())
