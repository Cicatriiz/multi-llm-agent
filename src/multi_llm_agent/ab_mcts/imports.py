"""Import helper for optional dependencies like PyMC."""

import warnings
from contextlib import contextmanager
from typing import Generator


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency is not available."""
    pass


class ImportManager:
    """Manages optional import state."""
    def __init__(self):
        self.failed = False
        self.error = None


@contextmanager
def try_import() -> Generator[ImportManager, None, None]:
    """
    Context manager for importing optional dependencies.
    
    Usage:
        with try_import() as _import:
            import pymc as pm  # This will be handled gracefully if not available
        
        if _import.failed:
            # Handle the case where imports failed
            pass
    """
    manager = ImportManager()
    try:
        yield manager
    except ImportError as e:
        manager.failed = True
        manager.error = e
        # For now, we'll just issue a warning and continue
        # In production, you might want to fall back to simpler algorithms
        warnings.warn(
            f"Optional dependency not available: {e}. "
            "AB-MCTS-M will use simplified logic instead of full PyMC implementation.",
            UserWarning
        )
