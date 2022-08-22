import datetime
import logging
import pathlib
import sys
import os

import nero.constants as constants


def retrieve_main_module_name() -> str:
    try:
        main_module_file_path = sys.modules['__main__'].__file__
        main_module_name = os.path.basename(main_module_file_path).split(".")[0]
        return str(main_module_name)
    except AttributeError:
        return "jupyter"


def formatted_today() -> str:
    now = datetime.datetime.now()
    return now.strftime('%Y_%m_%d_%H_%M_%S')


def create_log_file_path(main_module_name: str) -> str:
    pathlib.Path(constants.LOGS_DIR).mkdir(parents=True, exist_ok=True)
    today = formatted_today()
    log_file = f"{main_module_name}_{today}.log"
    return os.path.join(constants.LOGS_DIR, log_file)


def get_configured_logger() -> logging.Logger:
    main_module_name = retrieve_main_module_name()
    result_logger = logging.getLogger(main_module_name)
    if "popen" in main_module_name:
        return result_logger
    if not hasattr(result_logger, 'initialized') or not getattr(result_logger, 'initialized'):
        result_logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(create_log_file_path(main_module_name))
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        result_logger.addHandler(file_handler)
        setattr(result_logger, 'initialized', True)
    return result_logger
