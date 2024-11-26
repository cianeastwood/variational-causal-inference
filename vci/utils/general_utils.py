import os
import gzip
import json
import struct
import logging

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import wandb


class MyLogger:
    """ Wrapper to allow logging to both console and visual logger (tensorboard or wandb) """
    def __init__(self, config, log_dir, name=None, level='INFO', is_wandb=False):
        initialize_logger(log_dir, name=name, level=level)  # initialize logger for printing via logging.info
        self.is_wandb = is_wandb
        self.visual_logger = wandb if self.is_wandb else SummaryWriter(log_dir=log_dir)
        if is_wandb:
            defaults = dict(
                project=config["project"],
                entity=config["entity"],
                name=config["name"],
                config=config,
                dir=log_dir,
            )
            wandb.init(**defaults)
        
    def update(self, stats_dict, step, commit=False):
        # First, print to console
        logging.info(json.dumps(stats_dict))

        # Then, log to visual logger
        if self.is_wandb:
            # wandb
            wandb.log(stats_dict, commit=commit)
        else:
            # tensorboard
            for key, val in stats_dict.items():
                self.visual_logger.add_scalar(key, val, step)

    def close(self):
        if self.is_wandb:
            wandb.finish()
        else:
            self.visual_logger.close()
        logging.shutdown()


def initialize_logger(artifact_path, name=None, level='INFO'):
    logfile = os.path.join(artifact_path, 'log.txt')
    if name is None:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name)
    logger.setLevel(level)

    handler_console = logging.StreamHandler()
    handler_file    = logging.FileHandler(logfile)

    logger.addHandler(handler_console)
    logger.addHandler(handler_file)
    return logger

def pjson(s):
    print(json.dumps(s), flush=True)

def ljson(s):
    logging.info(json.dumps(s))

def unique_ind(records_array):
    # creates an array of indices, sorted by unique element
    idx_sort = np.argsort(records_array)

    # sorts records array so all unique elements are together 
    sorted_records_array = records_array[idx_sort]

    # returns the unique values, the index of the first occurrence of a value
    vals, idx_start = np.unique(sorted_records_array, return_index=True)

    # splits the indices into separate arrays
    res = np.split(idx_sort, idx_start[1:])

    return dict(zip(vals, res))

def _load_uint8(f):
    idx_dtype, ndim = struct.unpack('BBBB', f.read(4))[2:]
    shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data

def load_idx(path: str) -> np.ndarray:
    """Reads an array in IDX format from disk.

    Parameters
    ----------
    path : str
        Path of the input file. Will uncompress with `gzip` if path ends in '.gz'.

    Returns
    -------
    np.ndarray
        Output array of dtype ``uint8``.

    References
    ----------
    http://yann.lecun.com/exdb/mnist/
    """
    open_fcn = gzip.open if path.endswith('.gz') else open
    with open_fcn(path, 'rb') as f:
        return _load_uint8(f)
