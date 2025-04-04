import logging
import traceback
from functools import wraps
from typing import Type, Callable, Any

logger = logging.getLogger(__name__)

class DiffusionLMError(Exception):
    """Base exception class for DiffusionLM package"""
    pass

def handle_errors(error_class: Type[Exception] = DiffusionLMError, 
                 reraise: bool = True, 
                 logger: logging.Logger = logger) -> Callable:
    """
    Decorator for handling errors in functions.
    
    Args:
        error_class: Exception class to wrap the error in
        reraise: Whether to reraise the exception after logging
        logger: Logger instance to use
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Error in {func.__name__}: {str(e)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                
                if reraise:
                    raise error_class(error_msg) from e
        return wrapper
    return decorator

def setup_logging(log_file: str = None, 
                 level: int = logging.INFO,
                 format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s') -> None:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
        format: Log message format
    """
    logging.basicConfig(
        level=level,
        format=format,
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
