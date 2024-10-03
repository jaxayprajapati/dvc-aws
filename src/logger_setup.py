import logging
import os


class Logger:
    def __init__(self, logger_file_name, log_directory) -> None:
        # Ensure the "logs" directory exists
        self.log_dir = log_directory
        os.makedirs(self.log_dir, exist_ok=True)

        # Setting up logger
        self.__logger__ = logging.getLogger(logger_file_name)
        self.__logger__.setLevel('DEBUG')

        console_handler = logging.StreamHandler()
        console_handler.setLevel('DEBUG')

        log_file_path = os.path.join(self.log_dir, f'{logger_file_name}.log')
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel('DEBUG')

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        self.__logger__.addHandler(console_handler)
        self.__logger__.addHandler(file_handler)