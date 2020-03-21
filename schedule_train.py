from emotion import run, save_config_and_results
import logging
import sys
from dataset import MediaEval18

config = {
    'seq_len': 64,
    'num_hidden': 2,
    'hidden_size': 128,
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
    'fragment': 0.01,
    'features': 'visual'
}

# Parse args
logger = logging.getLogger()
logger.setLevel(config['logger_level'])

features = MediaEval18._features_len.keys()

for feature in features:
    config['features']=feature

    for arg_name, arg in config.items():
        logger.info(
            "initialization -- {} - {}".format(arg_name, arg))

    try:
        train_losses, test_losses = run(config)
        save_config_and_results(
            config, train_losses, test_losses)

    except Exception as exception:
        logger.critical(sys.exc_info())
