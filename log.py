# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:07:58 2020

@author: Tim
"""

import logging
import time
import os


# handler = logging.StreamHandler()
# handler.setFormatter(formatter)
# logger.addHandler(handler)

def setup_custom_logger(name):
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    if not os.path.exists('logs'):
        os.makedirs('logs')

    log_file_name = "logs/emotion_" + time.strftime("%Y-%m-%d_%H:%M") + ".log"

    fh = logging.FileHandler(log_file_name, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    return logger
