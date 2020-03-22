# -*- coding: utf-8 -*-
"""
Defines the the logger object to track execution

Logs are stored into logs/emotion_YYYY-mm-dd.log files
Default logger level is set to INFO, use --logger-level DEBUG tag
to get more information about execution trace.

To setup logger :
>>> from log import stetup_custom_logger
>>> logger = setup_custom_logger()

 To log log into file :
>>> logger.info("my first log message")

Content of logs/emotion_YYYY-mm-dd.log
> YYYY-mm-dd HH:MM:SS,sss - INFO - <module> - my first message
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
