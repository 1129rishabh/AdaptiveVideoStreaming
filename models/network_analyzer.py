import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import deque


class NetworkAnalyzer:
    def __init__(self, config):
        self.config = config
        self.lookback_window = config['ml']['lookback_window']
        self.history = {
            'bandwidth': deque(maxlen=self.lookback_window),
            'latency': deque(maxlen=self.lookback_window),
            'loss_rate': deque(maxlen=self.lookback_window),
            'buffer_level': deque(maxlen=self.lookback_window)
        }
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False

    def update(self, metrics):
        """
        Update network statistics history

        Args:
            metrics: dict containing 'bandwidth', 'latency', 'loss_rate', 'buffer_level'
        """
        for key in self.history:
            if key in metrics:
                self.history[key].append(metrics[key])

    def get_features(self):
        """
        Get the current network features for prediction

        Returns:
            numpy array of shape (lookback_window, n_features)
        """
        # Check if we have enough data
        if len(self.history['bandwidth']) < self.lookback_window:
            # Fill with zeros if not enough history
            return np.zeros((self.lookback_window, len(self.history)))

        # Convert history to numpy array
        features = np.column_stack([
            list(self.history[feature]) for feature in self.config['ml']['features']
        ])

        # Scale features if scaler is fitted
        if self.is_scaler_fitted:
            features = self.scaler.transform(features)

        return features

    def fit_scaler(self, data):
        """
        Fit the scaler on historical data

        Args:
            data: pandas DataFrame containing network statistics
        """
        self.scaler.fit(data[self.config['ml']['features']])
        self.is_scaler_fitted = True

    def predict_future_bandwidth(self, horizon=5):
        """
        Simple prediction of future bandwidth based on recent trends

        Args:
            horizon: number of future steps to predict

        Returns:
            list of predicted bandwidth values
        """
        if len(self.history['bandwidth']) < 3:
            return [self.config['network']['default_bandwidth']] * horizon

        # Simple linear extrapolation
        recent_bandwidth = list(self.history['bandwidth'])
        slope = (recent_bandwidth[-1] - recent_bandwidth[-3]) / 2

        predictions = []
        last_value = recent_bandwidth[-1]

        for _ in range(horizon):
            next_value = last_value + slope
            # Ensure bandwidth is within reasonable limits
            next_value = max(self.config['network']['min_bandwidth'],
                             min(self.config['network']['max_bandwidth'], next_value))
            predictions.append(next_value)
            last_value = next_value

        return predictions

    def detect_network_change(self, threshold=0.2):
        """
        Detect if there's a significant change in network conditions

        Args:
            threshold: relative change threshold to trigger detection

        Returns:
            bool: True if significant change detected
        """
        if len(self.history['bandwidth']) < self.lookback_window:
            return False

        # Calculate coefficient of variation for bandwidth
        bandwidth_array = np.array(list(self.history['bandwidth']))
        cv = np.std(bandwidth_array) / np.mean(bandwidth_array)

        # Check if recent change is significant
        recent_avg = np.mean(bandwidth_array[-3:])
        older_avg = np.mean(bandwidth_array[:-3])

        if older_avg > 0:
            relative_change = abs(recent_avg - older_avg) / older_avg
            return relative_change > threshold or cv > threshold

        return False
