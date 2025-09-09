"""
Memory management utilities for CAD processing system.
"""

import gc
import psutil
import os
from typing import Optional
from contextlib import contextmanager
import numpy as np


class MemoryManager:
    """Memory usage monitoring and management for image processing."""

    def __init__(self, max_memory_mb: Optional[float] = None):
        self.max_memory_mb = max_memory_mb or (psutil.virtual_memory().total / (1024 * 1024) * 0.8)  # 80% of total RAM
        self.process = psutil.Process(os.getpid())

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / (1024 * 1024)

    def is_memory_limit_exceeded(self) -> bool:
        """Check if memory usage exceeds the limit."""
        return self.get_memory_usage_mb() > self.max_memory_mb

    def force_garbage_collection(self) -> None:
        """Force garbage collection to free memory."""
        gc.collect()

    @contextmanager
    def memory_guard(self, operation_name: str = "operation"):
        """Context manager to monitor memory usage during operations."""
        initial_memory = self.get_memory_usage_mb()
        try:
            yield
        finally:
            final_memory = self.get_memory_usage_mb()
            memory_delta = final_memory - initial_memory
            if memory_delta > 50:  # More than 50MB increase
                print(".1f")
                self.force_garbage_collection()


def optimize_image_for_processing(image: np.ndarray, target_max_dim: int = 4096) -> np.ndarray:
    """Optimize image dimensions for processing while maintaining aspect ratio.

    Args:
        image: Input image array
        target_max_dim: Maximum dimension for processing

    Returns:
        Optimized image array
    """
    h, w = image.shape[:2]

    # If image is already within limits, return as-is
    if max(h, w) <= target_max_dim:
        return image

    # Calculate scaling factor
    scale = target_max_dim / max(h, w)

    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image
    if len(image.shape) == 3:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return resized


def calculate_optimal_workers(memory_limit_mb: float, estimated_mb_per_worker: float = 512) -> int:
    """Calculate optimal number of workers based on memory constraints.

    Args:
        memory_limit_mb: Total memory limit in MB
        estimated_mb_per_worker: Estimated memory usage per worker in MB

    Returns:
        Optimal number of workers
    """
    # Reserve 20% for system and other processes
    available_mb = memory_limit_mb * 0.8
    optimal_workers = max(1, int(available_mb / estimated_mb_per_worker))

    # Cap at CPU count
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    return min(optimal_workers, cpu_count)


# Import cv2 here to avoid circular imports
import cv2

# Handle imports for both package and direct execution
try:
    from .exceptions import CADProcessingError
except ImportError:
    try:
        from exceptions import CADProcessingError
    except ImportError:
        CADProcessingError = Exception  # Fallback
