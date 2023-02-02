import logging

LOGGING_LEVEL = logging.DEBUG # usually INFO is good
class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""
    # see https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/7995762#7995762

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    cyan = "\x1b[1;36m" # dark green
    green = "\x1b[31;21m" # dark green
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    light_blue = "\x1b[1;36m"
    blue = "\x1b[1;34m"
    reset = "\x1b[0m"
    # File "{file}", line {max(line, 1)}'.replace("\\", "/")
    format = '[%(levelname)s]: %(asctime)s - %(name)s - %(message)s (File "%(pathname)s", line %(lineno)d, in %(funcName)s)'

    FORMATS = {
        logging.DEBUG: light_blue + format + reset,
        logging.INFO: cyan + format + reset,
        logging.WARNING: red + format + reset,
        logging.ERROR: bold_red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record).replace("\\", "/") #replace \ with / for pycharm links


def get_logger(name):
    """ Use get_logger to define a logger with useful color output and info and warning turned on according to the global LOGGING_LEVEL.

    :param name: the name of this logger. Use __name__ to give it the name of the module that instantiates it.

    :returns: the logger.
    """
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_LEVEL)
    # create console handler
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    return logger