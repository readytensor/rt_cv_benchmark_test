import os
import json
import random
import threading
import psutil
import torch as T
import time
import numpy as np
from pathlib import Path
from typing import Dict, Union
from config import paths


def read_json_as_dict(input_path: str) -> Dict:
    """
    Reads a JSON file and returns its content as a dictionary.
    If input_path is a directory, the first JSON file in the directory is read.
    If input_path is a file, the file is read.

    Args:
        input_path (str): The path to the JSON file or directory containing a JSON file.

    Returns:
        dict: The content of the JSON file as a dictionary.

    Raises:
        ValueError: If the input_path is neither a file nor a directory,
                    or if input_path is a directory without any JSON files.
    """
    if os.path.isdir(input_path):
        # Get all the JSON files in the directory
        json_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".json")
        ]

        # If there are no JSON files, raise a ValueError
        if not json_files:
            raise ValueError("No JSON files found in the directory")

        # Else, get the path of the first JSON file
        json_file_path = json_files[0]

    elif os.path.isfile(input_path):
        json_file_path = input_path
    else:
        raise ValueError("Input path is neither a file nor a directory")

    # Read the JSON file and return it as a dictionary
    with open(json_file_path, "r", encoding="utf-8") as file:
        json_data_as_dict = json.load(file)

    return json_data_as_dict


def set_seeds(seed_value: int) -> None:
    """
    Set the random seeds for Python, NumPy, etc. to ensure
    reproducibility of results.

    Args:
        seed_value (int): The seed value to use for random
            number generation. Must be an integer.

    Returns:
        None
    """
    if isinstance(seed_value, int):
        os.environ["PYTHONHASHSEED"] = str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        T.manual_seed(seed_value)
        T.cuda.manual_seed(seed_value)
        T.cuda.manual_seed_all(seed_value)
    else:
        raise ValueError(f"Invalid seed value: {seed_value}. Cannot set seeds.")


def get_model_parameters(
    model_name: str,
    hyperparameters_file_path: str = paths.HYPERPARAMETERS_FILE,
    hyperparameter_tuning: bool = False,
) -> dict:
    """
    Read hyperparameters from hyperparameters file.

    Args:
        model_name (str): Name of the model for which hyperparameters are read.
        hyperparameters_file_path (str): File path for hyperparameters.
        hyperparameter_tuning (bool): Whether hyperparameter tuning is used or not.


    """
    hyperparameters_dict = read_json_as_dict(hyperparameters_file_path)
    model_parameters = hyperparameters_dict[model_name]

    if not hyperparameter_tuning:
        hyperparameters = {i["name"]: i["default"] for i in model_parameters}

    else:
        # TODO: Read hyperparameters in case of tuning.
        pass

    return hyperparameters


def contains_subdirectories(folder_path):
    """Check if the given folder contains any subdirectories using pathlib."""
    folder = Path(folder_path)
    return any(entry.is_dir() for entry in folder.iterdir())


def get_optimizer(optimizer_str: str):
    optimizers = {
        "sgd": T.optim.SGD,
        "adam": T.optim.Adam,
    }

    if optimizer_str not in optimizers.keys():
        raise ValueError(f"{optimizer_str} is not a recognized optimizer.")

    return optimizers[optimizer_str]


def get_dataloader_parameters(config_dict: Dict) -> Dict:
    dataloader_params_names = [
        "batch_size",
        "num_workers",
        "image_size",
        "validation_size",
    ]

    return {
        k: config_dict[k] for k in dataloader_params_names if k in config_dict.keys()
    }


class MemoryMonitor:
    peak_memory = 0  # Class variable to store peak memory usage

    def __init__(self, interval=20.0, logger=print):
        self.interval = interval  # Time between executions in seconds
        self.timer = None  # Placeholder for the timer object
        self.logger = logger

    def monitor_memory(self):
        process = psutil.Process(os.getpid())
        children = process.children(recursive=True)
        total_memory = process.memory_info().rss

        for child in children:
            total_memory += child.memory_info().rss

        # Check if the current memory usage is a new peak and update accordingly
        MemoryMonitor.peak_memory = max(MemoryMonitor.peak_memory, total_memory)

    def _schedule_monitor(self):
        """Internal method to schedule the next execution"""
        self.monitor_memory()
        # Only reschedule if the timer has not been canceled
        if self.timer is not None:
            self.timer = threading.Timer(self.interval, self._schedule_monitor)
            self.timer.start()

    def start(self):
        """Starts the periodic monitoring"""
        if self.timer is not None:
            return  # Prevent multiple timers from starting
        self.timer = threading.Timer(self.interval, self._schedule_monitor)
        self.timer.start()

    def stop(self):
        """Stops the periodic monitoring"""
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        self.logger.info(
            f"CPU Memory allocated (peak): {MemoryMonitor.peak_memory / (1024**2):.2f} MB"
        )

    @classmethod
    def get_peak_memory(cls):
        """Returns the peak memory usage"""
        return cls.peak_memory


def get_peak_memory_usage() -> Union[float, None]:
    """
    Returns the peak memory usage by current cuda device if available
    """
    if not T.cuda.is_available():
        return None

    current_device = T.cuda.current_device()
    peak_memory = T.cuda.max_memory_allocated(current_device)
    return peak_memory / 1e6


class ResourceTracker(object):
    """
    This class serves as a context manager to track time and
    memory allocated by code executed inside it.
    """

    def __init__(self, logger, monitoring_interval):
        self.logger = logger
        self.monitor = MemoryMonitor(logger=logger, interval=monitoring_interval)

    def __enter__(self):
        self.start_time = time.time()
        if T.cuda.is_available():
            T.cuda.reset_peak_memory_stats()  # Reset CUDA memory stats
            T.cuda.empty_cache()  # Clear CUDA cache

        self.monitor.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.monitor.stop()
        cuda_peak = get_peak_memory_usage()
        if cuda_peak:
            self.logger.info(f"CUDA Memory allocated (peak): {cuda_peak:.2f} MB")

        elapsed_time = self.end_time - self.start_time

        self.logger.info(f"Execution time: {elapsed_time:.2f} seconds")
