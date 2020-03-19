from emotion import run, save_config_and_results
import logging
import sys

config = {
    'seq_len': 20,
    'num_hidden': 1,
    'hidden_size': 32,
    'lr': 0.001,
    'batch_size': 128,
    'grad_clip': None,
    'nb_epoch': 50,
    'optimizer': 'SGD',
    'crit': 'MSE',
    'weight_decay': 0,
    'dropout': 0,
    'logger_level': 20,
    'fragment': 1
}

# Parse args
logger = logging.getLogger()
logger.setLevel(config['logger_level'])


seq_lens = [2**5, 2**6, 2**7, 2**8, 2**9]
num_hiddens = [1, 2, 3, 4, 5]
weight_decays = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
dropouts = [0.1, 0.2, 0.3, 0.4, 0.5]
grad_clips = [None, 1e4, 1e3, 1e2, 1e1, 1e0]
hidden_sizes = [2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12]

for seq_len in seq_lens:
    config['seq_len'] = seq_len
    for num_hidden in num_hiddens:
        config['num_hidden'] = num_hidden
        for weight_decay in weight_decays:
            config['weight_decay'] = weight_decay
            for dropout in dropouts:
                config['dropout'] = dropout
                for grad_clip in grad_clips:
                    config['grad_clip'] = grad_clip
                    for hidden_size in hidden_sizes:
                        config['hidden_size'] = hidden_size

                        print(config)

                        for arg_name, arg in config.items():
                            logger.info(
                                "initialization -- {} - {}".format(arg_name, arg))

                        try:
                            train_losses, test_losses = run(config)
                            save_config_and_results(
                                config, train_losses, test_losses)

                        except Exception as exception:
                            logger.critical(sys.exc_info())
