import numpy as np
import time
from collections import deque


class PathSelector:
    def __init__(self, multipath_manager, config):
        self.multipath_manager = multipath_manager
        self.config = config
        self.switching_threshold = config['mpet']['path_switching_threshold']
        self.current_paths = []
        self.path_history = {}
        self.last_switch_time = time.time()
        self.min_switch_interval = 5  # seconds

    def evaluate_path_quality(self, path_id):
        """
        Calculate a quality score for a path based on its characteristics

        Args:
            path_id: identifier for the path

        Returns:
            quality score (higher is better)
        """
        stats = self.multipath_manager.path_stats.get(path_id, {})

        if not stats:
            return 0

        # Normalize metrics to 0-1 range where 1 is best
        bandwidth_norm = min(1.0, stats.get('bandwidth', 0) / self.config['network']['max_bandwidth'])
        latency_norm = max(0, 1.0 - stats.get('latency', 200) / 200)  # Assuming 200ms is the worst acceptable latency
        loss_norm = max(0, 1.0 - stats.get('loss_rate', 0.05) / 0.05)  # Assuming 5% loss is the worst acceptable

        # Weighted sum of normalized metrics
        quality = 0.6 * bandwidth_norm + 0.3 * latency_norm + 0.1 * loss_norm

        return quality

    def should_switch_paths(self):
        """
        Determine if we should switch paths based on quality changes

        Returns:
            bool: True if paths should be switched
        """
        # Don't switch too frequently
        if time.time() - self.last_switch_time < self.min_switch_interval:
            return False

        # If we don't have current paths, we should select some
        if not self.current_paths:
            return True

        # Calculate current average path quality
        current_quality = np.mean([
            self.evaluate_path_quality(path)
            for path in self.current_paths
        ])

        # Get potential new paths
        potential_paths = self.multipath_manager.select_paths()

        # Calculate potential new average path quality
        potential_quality = np.mean([
            self.evaluate_path_quality(path)
            for path in potential_paths
        ])

        # Switch if potential paths are significantly better
        if potential_quality > current_quality * (1 + self.switching_threshold):
            return True

        return False

    def select_paths(self):
        """
        Select the best paths for transmission

        Returns:
            list of selected path IDs
        """
        if self.should_switch_paths():
            self.current_paths = self.multipath_manager.select_paths()
            self.last_switch_time = time.time()

        return self.current_paths

    def update_path_history(self, path_id, quality_metrics):
        """
        Update history of path performance

        Args:
            path_id: identifier for the path
            quality_metrics: dict with path quality metrics
        """
        if path_id not in self.path_history:
            self.path_history[path_id] = {
                'quality_scores': deque(maxlen=20),
                'last_used': time.time()
            }

        quality = self.evaluate_path_quality(path_id)
        self.path_history[path_id]['quality_scores'].append(quality)
        self.path_history[path_id]['last_used'] = time.time()

    def get_path_stability(self, path_id):
        """
        Calculate stability score for a path based on its history

        Args:
            path_id: identifier for the path

        Returns:
            stability score (0-1, higher is more stable)
        """
        if path_id not in self.path_history:
            return 0.5  # Default medium stability for unknown paths

        scores = list(self.path_history[path_id]['quality_scores'])

        if len(scores) < 2:
            return 0.5

        # Calculate coefficient of variation (lower means more stable)
        cv = np.std(scores) / max(np.mean(scores), 0.001)

        # Convert to stability score (1 - normalized cv)
        stability = max(0, min(1, 1 - cv / 0.5))  # Assuming cv > 0.5 is very unstable

        return stability
