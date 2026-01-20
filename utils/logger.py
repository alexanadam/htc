# def setup_logger(name, log_file, console=True):
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.INFO)

#     if not logger.handlers:
#         os.makedirs(os.path.dirname(log_file), exist_ok=True)
#         file_handler = logging.FileHandler(log_file)
#         file_handler.setLevel(logging.INFO)

#         formatter = logging.Formatter(
#             '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#         )
#         file_handler.setFormatter(formatter)
#         logger.addHandler(file_handler)

#         if console:
#             console_handler = logging.StreamHandler()
#             console_handler.setLevel(logging.INFO)
#             console_handler.setFormatter(formatter)
#             logger.addHandler(console_handler)

#     return logger

import logging
import sys
import os

from datetime import datetime

import configs.config as config


def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        format = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(format)
        logger.addHandler(handler)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"log_{name}_{timestamp}.log"
        os.makedirs(config.LOG_PATH, exist_ok=True)
        log_path = os.path.join(config.LOG_PATH, log_file)

        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setFormatter(format)
        logger.addHandler(file_handler)

    return logger
