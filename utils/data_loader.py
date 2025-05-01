import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()

    def load_network_logs(self, file_path):
        """
        Load network logs from CSV file

        Args:
            file_path: path to the CSV file

        Returns:
            pandas DataFrame with network logs
        """
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            print(f"Error loading network logs: {e}")
            # Create a synthetic dataset if file doesn't exist
            return self._create_synthetic_data()

    def _create_synthetic_data(self, n_samples=10000):
        """
        Create synthetic network data for testing

        Args:
            n_samples: number of samples to generate

        Returns:
            pandas DataFrame with synthetic data
        """
        # Time series index
        timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='10s')

        # Generate bandwidth with realistic patterns (daily cycles, random drops)
        time_of_day = np.sin(np.pi * (timestamps.hour / 12))
        base_bandwidth = 5000 + 3000 * time_of_day  # Higher during day, lower at night

        # Add some random variation and occasional drops
        random_var = np.random.normal(0, 500, n_samples)
        drops = np.random.binomial(1, 0.05, n_samples) * -2000  # 5% chance of bandwidth drop
        bandwidth = np.maximum(500, base_bandwidth + random_var + drops)

        # Generate latency (inversely related to bandwidth with some noise)
        base_latency = 10000 / (bandwidth + 500)  # Higher latency when bandwidth is lower
        latency = base_latency * (1 + 0.3 * np.random.random(n_samples))

        # Generate packet loss (higher when bandwidth is lower)
        loss_rate = 0.05 * (1 - np.minimum(1, bandwidth / 8000)) + 0.01 * np.random.random(n_samples)

        # Buffer level simulation
        buffer_level = np.zeros(n_samples)
        buffer_size = self.config['network']['buffer_size']

        for i in range(1, n_samples):
            # Buffer increases based on bandwidth and decreases at playback rate
            download_rate = bandwidth[i] / 3000  # Simplified: bitrate / bandwidth
            buffer_change = 1 - download_rate  # 1 second of playback time
            buffer_level[i] = np.clip(buffer_level[i - 1] + buffer_change, 0, buffer_size)

        # Determine optimal bitrate based on bandwidth and buffer
        available_bitrates = np.array(self.config['video']['available_bitrates'])
        optimal_bitrate = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            # Conservative bitrate selection based on bandwidth and buffer level
            safe_bandwidth = bandwidth[i] * 0.8  # 80% of available bandwidth

            # More conservative when buffer is low
            if buffer_level[i] < 5:
                safe_bandwidth *= 0.7

            # Select highest bitrate below safe bandwidth
            suitable_bitrates = available_bitrates[available_bitrates < safe_bandwidth]
            if len(suitable_bitrates) > 0:
                optimal_bitrate[i] = np.max(suitable_bitrates)
            else:
                optimal_bitrate[i] = np.min(available_bitrates)

        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'bandwidth': bandwidth,
            'latency': latency,
            'loss_rate': loss_rate,
            'buffer_level': buffer_level,
            'optimal_bitrate': optimal_bitrate
        })

        return df

    def prepare_training_data(self, df, lookback_window=None):
        """
        Prepare data for training the bitrate prediction model

        Args:
            df: pandas DataFrame with network logs
            lookback_window: number of past observations to include (default: from config)

        Returns:
            X_train, X_test, y_train, y_test
        """
        if lookback_window is None:
            lookback_window = self.config['ml']['lookback_window']

        features = self.config['ml']['features']
        target = self.config['ml']['target']

        # Check if dataframe is empty or too small
        if len(df) <= lookback_window:
            print(
                f"Warning: DataFrame has only {len(df)} rows, which is not enough for lookback window of {lookback_window}")
            # Generate more synthetic data
            df = self._create_synthetic_data(n_samples=lookback_window * 10)

        # Scale features
        X = df[features].values
        X_scaled = self.scaler.fit_transform(X)

        # Create sequences
        X_sequences = []
        y = []

        for i in range(lookback_window, len(df)):
            X_sequences.append(X_scaled[i - lookback_window:i])

            # Get target value
            target_value = df[target].iloc[i]

            # Check if target_value exists in available_bitrates
            if target_value in self.config['video']['available_bitrates']:
                target_idx = self.config['video']['available_bitrates'].index(target_value)
            else:
                # Use closest available bitrate
                available_bitrates = np.array(self.config['video']['available_bitrates'])
                target_idx = np.argmin(np.abs(available_bitrates - target_value))

            # Convert target to one-hot encoding
            target_onehot = np.zeros(len(self.config['video']['available_bitrates']))
            target_onehot[target_idx] = 1

            y.append(target_onehot)

        # Check if we have enough sequences
        if len(X_sequences) == 0:
            print("Warning: No sequences were created. Check your data and lookback window.")
            # Create dummy data for demonstration
            n_features = len(features)
            n_outputs = len(self.config['video']['available_bitrates'])
            X_sequences = np.random.random((100, lookback_window, n_features))
            y = np.random.random((100, n_outputs))
            # Normalize y to be valid probabilities
            y = y / y.sum(axis=1, keepdims=True)

        X_sequences = np.array(X_sequences)
        y = np.array(y)

        # Split into train and test sets
        split_ratio = self.config['ml']['train_test_split']
        X_train, X_test, y_train, y_test = train_test_split(
            X_sequences, y, test_size=1 - split_ratio, shuffle=False
        )

        return X_train, X_test, y_train, y_test

