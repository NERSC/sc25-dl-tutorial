import os
import logging

_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def slurm_filter(record):
    # Only the rank-0 process emits log records. Works under SLURM, PALS,
    # torchrun, or single-process — falls back to RANK/PMI_RANK if SLURM
    # isn't set.
    for var in ("SLURM_PROCID", "RANK", "PMI_RANK", "PMIX_RANK"):
        val = os.environ.get(var)
        if val is not None and val != "":
            try:
                return int(val) == 0
            except ValueError:
                continue
    return True


def config_logger(log_level=logging.INFO):
    logging.basicConfig(format=_format, level=log_level)
    root_logger = logging.getLogger()
    root_logger.addFilter(slurm_filter)


def log_to_file(
    logger_name=None, log_level=logging.INFO, log_filename="tensorflow.log"
):

    if not os.path.exists(os.path.dirname(log_filename)):
        os.makedirs(os.path.dirname(log_filename))

    if logger_name is not None:
        log = logging.getLogger(logger_name)
    else:
        log = logging.getLogger()

    fh = logging.FileHandler(log_filename)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter(_format))
    log.addHandler(fh)


def log_versions():
    import torch
    import subprocess

    logging.info("--------------- Versions ---------------")
    logging.info(
        "git branch: " + str(subprocess.check_output(["git", "branch"]).strip())
    )
    logging.info(
        "git hash: "
        + str(subprocess.check_output(["git", "rev-parse", "HEAD"]).strip())
    )
    logging.info("Torch: " + str(torch.__version__))
    logging.info("----------------------------------------")


class disable_logging(object):
    """
    A context manager to disable logging temporarily.
    """

    def __init__(self, level=logging.ERROR):  # pragma: no cover
        """
        Initialize the context manager.
        """
        logging.disable(level=level)

    def __enter__(self):  # pragma: no cover
        """
        Enter the context manager.
        """
        return self

    def __exit__(self, type, value, traceback):  # pragma: no cover
        """
        Exit the context manager and enable logging.
        """
        logging.disable(level=logging.NOTSET)
