"""
Logging configuration module for SCA Assist application.

This module sets up a centralized logging system that outputs to both
a file and the console for easy debugging and monitoring.
"""

import logging
import os
import sys

# Get the project root directory (parent of app folder)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Directory where log files will be stored (in project root)
LOG_DIR = os.path.join(ROOT_DIR, "logs")

# Create logs directory if it doesn't exist
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Path to the main application log file
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"

# Create file handler with UTF-8 encoding
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Create console handler with UTF-8 encoding for Windows compatibility
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,  # Minimum level to capture (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format=LOG_FORMAT,
    handlers=[file_handler, console_handler]
)

# Application logger instance - use this throughout the app
logger = logging.getLogger("my_app_logger")