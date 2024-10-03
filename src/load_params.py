import yaml
from logger_setup import Logger



class LoadParams(Logger):
    def __init__(self, params_path, log_file_name, log_directory):
        
        super().__init__(log_directory=log_directory, logger_file_name=log_file_name)

        """Load parameters from a YAML file."""
        try:
            with open(params_path, 'r') as file:
                self.__params__ = yaml.safe_load(file)
            self.__logger__.debug('Parameters retrieved from %s', params_path)
        except FileNotFoundError:
            self.__logger__.error('File not found: %s', params_path)
            raise
        except yaml.YAMLError as e:
            self.__logger__.error('YAML error: %s', e)
            raise
        except Exception as e:
            self.__logger__.error('Unexpected error: %s', e)
            raise
