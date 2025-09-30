# src/utils/logging.py


###
### Functions that handle logging and experiment tracking
###


# imports
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict


# implementations

def setup_logger(
    log_name: str = "experiment_logger", 
    log_file: str = None, 
    level: int = logging.INFO
) -> logging.Logger:
    """
    Sets up a logger that can write to a file and/or the console.

    Args:
        log_name: The name for the logger.
        log_file: Path to the file where logs should be saved.
        level: The minimum logging level to capture (e.g., logging.INFO).

    Returns:
        A configured logging.Logger object.
    """
    # Cria o logger
    logger = logging.getLogger(log_name)
    logger.setLevel(level)

    # Evita adicionar múltiplos handlers se a função for chamada várias vezes
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formato padrão para as mensagens de log
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Handler para o console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Handler para o arquivo, se um caminho for fornecido
    if log_file:
        # Garante que o diretório do log exista
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def log_experiment_config(logger: logging.Logger, config: Dict):
    """
    Logs the configuration parameters of an experiment in a readable format.

    Args:
        logger: The logger object to use.
        config: A dictionary containing the experiment's configuration.
    """
    logger.info("=" * 50)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 50)
    for key, value in config.items():
        logger.info(f"{key:<20}: {value}")
    logger.info("=" * 50)


def log_results(logger: logging.Logger, results: Dict):
    """
    Logs the final results of an experiment.

    Args:
        logger: The logger object to use.
        results: A dictionary containing the final metrics.
    """
    logger.info("=" * 50)
    logger.info("EXPERIMENT RESULTS")
    logger.info("=" * 50)
    for key, value in results.items():
        # Formata a chave para ser mais legível
        formatted_key = key.replace('_', ' ').title()
        logger.info(f"{formatted_key:<25}: {value:.4f}")
    logger.info("=" * 50)

