# Correct imports for standalone Keras
import keras
from keras import Sequential
from keras import layers
from keras import optimizers
from keras import callbacks
import numpy as np


class BitratePredictor:
    def __init__(self, config):
        self.config = config
        self.model = self._build_model()

    def _build_model(self):
        model_type = self.config['ml']['model_type']
        lookback = self.config['ml']['lookback_window']
        n_features = len(self.config['ml']['features'])
        n_outputs = len(self.config['video']['available_bitrates'])

        model = Sequential()

        if model_type == 'lstm':
            model.add(layers.LSTM(128, activation='relu', input_shape=(lookback, n_features), return_sequences=True))
            model.add(layers.LSTM(64, activation='relu'))
        elif model_type == 'gru':
            model.add(layers.GRU(128, activation='relu', input_shape=(lookback, n_features), return_sequences=True))
            model.add(layers.GRU(64, activation='relu'))
        elif model_type == 'cnn':
            model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(lookback, n_features)))
            model.add(layers.MaxPooling1D(pool_size=2))
            model.add(layers.Flatten())
            model.add(layers.Dense(128, activation='relu'))

        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(n_outputs, activation='softmax'))

        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['ml']['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X_train, y_train, X_val, y_val):
        """Train the bitrate prediction model"""
        return self.model.fit(
            X_train, y_train,
            epochs=self.config['ml']['epochs'],
            batch_size=self.config['ml']['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=[
                callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )

    def predict(self, network_stats):
        """Predict the optimal bitrate based on network statistics"""
        # Ensure input is properly shaped
        if network_stats.ndim == 2:
            network_stats = np.expand_dims(network_stats, axis=0)
        elif network_stats.ndim == 1:
            network_stats = np.reshape(network_stats, (1, 1, len(network_stats)))

        # Get probability distribution over bitrates
        bitrate_probs = self.model.predict(network_stats, verbose=0)

        # Get the index of the highest probability
        bitrate_index = np.argmax(bitrate_probs, axis=1)[0]

        # Return the corresponding bitrate
        return self.config['video']['available_bitrates'][bitrate_index]

    def save(self, path):
        """Save the model to disk"""
        self.model.save(path)

    def load(self, path):
        """Load the model from disk"""
        self.model = keras.models.load_model(path)
