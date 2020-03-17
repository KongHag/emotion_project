# -*- coding: utf-8 -*-
"""
TODO : write a docstring
"""

import logging
import time
import os


def setup_custom_logger(name):
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    if not os.path.exists('logs'):
        os.makedirs('logs')

    log_file_name = "logs/emotion_" + time.strftime("%Y-%m-%d") + ".log"

    fh = logging.FileHandler(log_file_name, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    logger = logging.getLogger()
    logger.addHandler(fh)

    return logger
