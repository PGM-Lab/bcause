import logging
import sys


def get_logger(logname, level=logging.INFO, stream = sys.stdout, fmt=None, filename=None):
    log = logging.getLogger(logname)
    log.setLevel(level)
    stdout_handler = logging.StreamHandler(stream)
    if fmt is not None: stdout_handler.setFormatter(logging.Formatter(fmt))
    log.addHandler(stdout_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        if fmt is not None: file_handler.setFormatter(logging.Formatter(fmt))
        log.addHandler(file_handler)
    return log
