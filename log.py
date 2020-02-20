# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:07:58 2020

@author: Tim
"""

import logging

def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

#    handler = logging.StreamHandler()
    fh = logging.FileHandler('test.log', mode = "w")
    fh.setLevel(logging.DEBUG)
    
#    handler.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
#    logger.addHandler(handler)
    logger.addHandler(fh)
    return logger

