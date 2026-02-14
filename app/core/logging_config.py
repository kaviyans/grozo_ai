"""
Simple logging configuration for the application.
Provides loggers for graphs and ingestion with consistent formatting.
"""

import logging
import sys
from functools import lru_cache


# ---------- LOG FORMAT ----------
LOG_FORMAT = "[%(asctime)s] %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---------- LOGGER FACTORY ----------
@lru_cache
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get or create a logger with the specified name."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        
        logger.addHandler(handler)
        logger.propagate = False
    
    return logger


# ---------- GRAPH LOGGER ----------
def get_graph_logger(graph_name: str = "graph") -> logging.Logger:
    """Get a logger for graph nodes."""
    return get_logger(f"graph.{graph_name}")


# ---------- INGESTION LOGGER ----------
def get_ingestion_logger() -> logging.Logger:
    """Get a logger for ingestion operations."""
    return get_logger("ingestion")


# ---------- HELPER FUNCTIONS ----------
def log_node_entry(logger: logging.Logger, node_name: str, state_keys: list = None, extra: str = None):
    """Log entry into a graph node."""
    msg = f"[{node_name}] ENTER"
    if state_keys:
        msg += f" | state_keys: {state_keys}"
    if extra:
        msg += f" | {extra}"
    logger.debug(msg)


def log_node_exit(logger: logging.Logger, node_name: str, result_keys: list = None, extra: str = None):
    """Log exit from a graph node."""
    msg = f"[{node_name}] EXIT"
    if result_keys:
        msg += f" | result_keys: {result_keys}"
    if extra:
        msg += f" | {extra}"
    logger.debug(msg)


def log_node_error(logger: logging.Logger, node_name: str, error: Exception, context: str = None):
    """Log an error in a graph node."""
    msg = f"[{node_name}] ERROR: {type(error).__name__}: {str(error)}"
    if context:
        msg += f" | {context}"
    logger.error(msg, exc_info=True)
