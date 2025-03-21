import logging
import os
import sys
from datetime import datetime
from typing import Optional

def setup_logger(name: str, log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with console and file handlers.
    
    Args:
        name: Logger name.
        log_dir: Directory to store logs. If None, only console logging is used.
        level: Logging level.
        
    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # Remove existing handlers
    for handler in logger.handlers[:]: 
        logger.removeHandler(handler)
        
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir is provided)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
    return logger
