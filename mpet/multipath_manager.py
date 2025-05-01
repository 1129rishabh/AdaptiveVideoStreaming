import numpy as np
from collections import defaultdict
import time


class MultipathManager:
    def __init__(self, config):
        self.config = config
        self.max_paths = config['mpet']['max_paths']
        self.active_paths = []
        self.path_stats = defaultdict(dict)
        self.path_selection_strategy = config['mpet']['path_selection_strategy']
        self.last_path_update = time.time()

    def discover_paths(self):
        """
        Discover available network paths
        This is a simplified simulation - in a real system this would involve
        network probing and discovery protocols

        Returns:
            list of discovered path IDs
        """
        # Simulate discovering 1-5 paths with different characteristics
        num_paths = np.random.randint(1, min(5, self.max_paths + 1))
        paths = []

        for i in range(num_paths):
            path_id = f"path_{i}"
            # Simulate path characteristics
            self.path_stats[path_id] = {
                'bandwidth': np.random.uniform(1000, 8000),  # kbps
                'latency': np.random.uniform(10, 200),  # ms
                'loss_rate': np.random.uniform(0, 0.05),  # 0-5%
                'last_updated': time.time()
            }
            paths.append(path_id)

        return paths

    def update_path_stats(self, path_id, stats):
        """
        Update statistics for a specific path

        Args:
            path_id: identifier for the path
            stats: dict containing 'bandwidth', 'latency', 'loss_rate'
        """
        if path_id in self.path_stats:
            self.path_stats[path_id].update(stats)
            self.path_stats[path_id]['last_updated'] = time.time()

    def select_paths(self):
        """
        Select the best paths based on the configured strategy

        Returns:
            list of selected path IDs
        """
        # First, discover available paths if none are active
        if not self.active_paths or time.time() - self.last_path_update > 30:
            discovered_paths = self.discover_paths()
            self.last_path_update = time.time()

            # If we have new paths, update active paths
            if discovered_paths:
                self.active_paths = discovered_paths[:self.max_paths]

        # If we still don't have paths, return empty list
        if not self.active_paths:
            return []

        # Apply path selection strategy
        if self.path_selection_strategy == 'bandwidth_weighted':
            # Sort paths by bandwidth (highest first)
            sorted_paths = sorted(
                self.active_paths,
                key=lambda p: self.path_stats[p].get('bandwidth', 0),
                reverse=True
            )
            return sorted_paths[:self.max_paths]

        elif self.path_selection_strategy == 'lowest_latency':
            # Sort paths by latency (lowest first)
            sorted_paths = sorted(
                self.active_paths,
                key=lambda p: self.path_stats[p].get('latency', float('inf'))
            )
            return sorted_paths[:self.max_paths]

        elif self.path_selection_strategy == 'round_robin':
            # Just return all active paths for round-robin scheduling
            return self.active_paths[:self.max_paths]

        # Default: return all active paths
        return self.active_paths[:self.max_paths]

    def get_total_bandwidth(self):
        """
        Calculate total available bandwidth across selected paths

        Returns:
            total bandwidth in kbps
        """
        selected_paths = self.select_paths()
        total_bandwidth = sum(
            self.path_stats[path].get('bandwidth', 0)
            for path in selected_paths
        )
        return total_bandwidth

    def get_path_weights(self):
        """
        Calculate weight for each path based on its bandwidth share

        Returns:
            dict mapping path_id to weight (0-1)
        """
        selected_paths = self.select_paths()
        total_bandwidth = self.get_total_bandwidth()

        if total_bandwidth == 0:
            # Equal weights if no bandwidth info
            return {path: 1.0 / len(selected_paths) for path in selected_paths}

        return {
            path: self.path_stats[path].get('bandwidth', 0) / total_bandwidth
            for path in selected_paths
        }
