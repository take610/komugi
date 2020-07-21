from pathlib import Path
from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG


def create_logger(exp_version, base_path="./logs"):
    log_file = (base_path + "/{}.log".format(exp_version))

    # logger
    logger_ = getLogger(exp_version)
    logger_.setLevel(DEBUG)

    # formatter
    fmr = Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")

    # file handler
    fh = FileHandler(log_file)
    fh.setLevel(DEBUG)
    fh.setFormatter(fmr)

    # stream handler
    ch = StreamHandler()
    ch.setLevel(INFO)
    ch.setFormatter(fmr)

    logger_.addHandler(fh)
    logger_.addHandler(ch)


def get_logger(exp_version):
    return getLogger(exp_version)

## time_keeper.py
import time
from functools import wraps


def stop_watch(VERSION):
    def _stop_watch(func):
        @wraps(func)
        def wrapper(*args, **kargs):
            start = time.time()

            result = func(*args, **kargs)

            elapsed_time = int(time.time() - start)
            minits, sec = divmod(elapsed_time, 60)
            hour, minits = divmod(minits, 60)

            get_logger(VERSION).info("[elapsed_time]\t>> {:0>2}:{:0>2}:{:0>2}".format(hour, minits, sec))
        return wrapper

    return _stop_watch