import numpy as np
from keras import optimizers
from keras import callbacks
from keras import models
from keras import layers


class QoEEstimator:
    def __init__(self, config):
        self.config = config
        self.model = self._build_model()

    def _build_model(self):
        """Build a neural network to estimate QoE"""
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(5,)),
            # Input: bitrate, bitrate changes, rebuffering, latency, packet loss
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)  # Output: QoE score
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error'
        )

        return model

    def train(self, X_train, y_train, X_val, y_val):
        """Train the QoE estimation model"""
        return self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[
                callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ]
        )

    def estimate_qoe(self, metrics):
        """
        Estimate QoE based on streaming metrics

        Args:
            metrics: dict containing 'bitrate', 'bitrate_changes', 'rebuffering_time', 'latency', 'packet_loss'

        Returns:
            Estimated QoE score
        """
        # For simple calculation without using the ML model
        bitrate_utility = metrics['bitrate'] / np.max(self.config['video']['available_bitrates'])
        rebuffering_penalty = self.config['qoe']['rebuffering_penalty'] * metrics['rebuffering_time']
        bitrate_switch_penalty = self.config['qoe']['bitrate_switch_penalty'] * metrics['bitrate_changes']

        qoe = self.config['qoe']['bitrate_weight'] * bitrate_utility - rebuffering_penalty - bitrate_switch_penalty

        return max(0, qoe)  # QoE should not be negative

    def predict(self, features):
        """
        Predict QoE using the trained model

        Args:
            features: numpy array of shape (n_samples, 5) containing the features

        Returns:
            Predicted QoE scores
        """
        return self.model.predict(features)

    def save(self, path):
        """Save the model to disk"""
        self.model.save(path)

    def load(self, path):
        """Load the model from disk"""
        self.model = models.load_model(path)
