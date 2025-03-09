import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir='logs'):
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w', encoding='utf-8'),  # <-- Add `encoding='utf-8'` here
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def debug(self, message):
        self.logger.debug(message)
        for handler in self.logger.handlers:
         handler.flush()  # <-- Forces the log entry to be written immediately


    def info(self, message):
        self.logger.info(message)
        for handler in self.logger.handlers:
            handler.flush()  # <-- Forces the log entry to be written immediately


    def warning(self, message):
        self.logger.warning(message)
        for handler in self.logger.handlers:
            handler.flush()  # <-- Forces the log entry to be written immediately


    def error(self, message):
        self.logger.error(message, exc_info=True)
        for handler in self.logger.handlers:
            handler.flush()  # <-- Forces the log entry to be written immediately


    def critical(self, message):
        self.logger.critical(message)
        for handler in self.logger.handlers:
            handler.flush()  # <-- Forces the log entry to be written immediately

