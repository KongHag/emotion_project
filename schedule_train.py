

from emotion import run, save_config_and_results
import logging
import sys
from dataset import MediaEval18


configs = []

# ------------ LSTM 1 couche cachée  --------------------------

configs.append({
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
    'fragment': 1,
    'features': 'all',
    'overlapping': True,
    'model-with-CNN': False
})


# ------------ LSTM 2 couches cachées  --------------------------

configs.append({
    'seq_len': 64,
    'num_hidden': 2,
    'hidden_size': 32,
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
    'features': 'all',
    'overlapping': True,
    'model-with-CNN': False
})

configs.append({
    'seq_len': 64,
    'num_hidden': 2,
    'hidden_size': 128,
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
    'features': 'all',
    'overlapping': True,
    'model-with-CNN': False
})

configs.append({
    'seq_len': 64,
    'num_hidden': 2,
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
    'fragment': 1,
    'features': 'all',
    'overlapping': True,
    'model-with-CNN': False
})

# ------------ LSTM 3 couches cachées  --------------------------

configs.append({
    'seq_len': 64,
    'num_hidden': 3,
    'hidden_size': 32,
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
    'features': 'all',
    'overlapping': True,
    'model-with-CNN': False
})

configs.append({
    'seq_len': 64,
    'num_hidden': 3,
    'hidden_size': 128,
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
    'features': 'all',
    'overlapping': True,
    'model-with-CNN': False
})

configs.append({
    'seq_len': 64,
    'num_hidden': 3,
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
    'fragment': 1,
    'features': 'all',
    'overlapping': True,
    'model-with-CNN': False
})

for config in configs :
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

