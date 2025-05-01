from collections import deque


class BitrateAdapter:
    def __init__(self, config, bitrate_predictor=None):
        self.config = config
        self.available_bitrates = sorted(config['video']['available_bitrates'])
        self.default_bitrate = config['video']['default_bitrate']
        self.bitrate_predictor = bitrate_predictor
        self.last_bitrates = deque(maxlen=5)  # Track recent bitrate decisions
        self.safety_factor = 0.8  # Conservative factor for throughput-based decisions
        self.switch_penalty = 0.2  # Penalty for switching bitrates (for stability)

    def select_bitrate(self, network_stats, buffer_level):
        """
        Select the optimal bitrate based on network conditions and buffer level

        Args:
            network_stats: dict with network statistics
            buffer_level: current buffer level in seconds

        Returns:
            selected bitrate
        """
        # If we have a ML predictor, use it
        if self.bitrate_predictor is not None:
            try:
                predicted_bitrate = self.bitrate_predictor.predict(network_stats)
                if predicted_bitrate in self.available_bitrates:
                    self._update_history(predicted_bitrate)
                    return predicted_bitrate
            except Exception as e:
                print(f"Error using bitrate predictor: {e}")

        # Fall back to heuristic approach
        return self._heuristic_bitrate_selection(network_stats, buffer_level)

    def _heuristic_bitrate_selection(self, network_stats, buffer_level):
        """
        Select bitrate using a heuristic approach

        Args:
            network_stats: dict with network statistics
            buffer_level: current buffer level in seconds

        Returns:
            selected bitrate
        """
        # Get estimated bandwidth in kbps
        bandwidth = network_stats.get('bandwidth', 0)

        # Apply safety factor to bandwidth
        safe_bandwidth = bandwidth * self.safety_factor

        # Get current/last bitrate
        current_bitrate = self.default_bitrate
        if self.last_bitrates:
            current_bitrate = self.last_bitrates[-1]

        # Adjust based on buffer level
        buffer_size = self.config['network']['buffer_size']

        # If buffer is low, be conservative
        if buffer_level < buffer_size * 0.2:
            # Select a lower bitrate
            for bitrate in sorted(self.available_bitrates):
                if bitrate < current_bitrate and bitrate * 1.5 < safe_bandwidth:
                    selected_bitrate = bitrate
                    break
            else:
                # If no suitable lower bitrate found, select the lowest
                selected_bitrate = min(self.available_bitrates)

        # If buffer is high, be more aggressive
        elif buffer_level > buffer_size * 0.8:
            # Try to select a higher bitrate
            selected_bitrate = current_bitrate
            for bitrate in sorted(self.available_bitrates, reverse=True):
                if bitrate > current_bitrate and bitrate < safe_bandwidth:
                    selected_bitrate = bitrate
                    break

        # Otherwise, select the highest bitrate below safe_bandwidth
        else:
            selected_bitrate = self.default_bitrate
            for bitrate in sorted(self.available_bitrates, reverse=True):
                if bitrate < safe_bandwidth:
                    selected_bitrate = bitrate
                    break

        # Apply switch penalty for stability
        if selected_bitrate != current_bitrate:
            # If the improvement is marginal, stick with current bitrate
            if selected_bitrate > current_bitrate:
                improvement = (selected_bitrate - current_bitrate) / current_bitrate
                if improvement < self.switch_penalty:
                    selected_bitrate = current_bitrate

        self._update_history(selected_bitrate)
        return selected_bitrate

    def _update_history(self, bitrate):
        """Update the history of selected bitrates"""
        self.last_bitrates.append(bitrate)

    def get_bitrate_switches(self):
        """
        Count the number of bitrate switches in recent history

        Returns:
            number of switches
        """
        if len(self.last_bitrates) <= 1:
            return 0

        switches = 0
        for i in range(1, len(self.last_bitrates)):
            if self.last_bitrates[i] != self.last_bitrates[i - 1]:
                switches += 1

        return switches

    def reset(self):
        """Reset the adapter state"""
        self.last_bitrates.clear()
