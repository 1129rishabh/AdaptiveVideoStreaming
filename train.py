import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from models.bitrate_predictor import BitratePredictor
from models.qoe_estimator import QoEEstimator
from utils.data_loader import DataLoader
from utils.visualization import Visualizer


def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("Loading and preparing data...")

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models/saved', exist_ok=True)

    # Load data
    data_loader = DataLoader(config)
    network_logs = data_loader.load_network_logs('data/network_logs.csv')

    # Save synthetic data if it was generated
    if not os.path.exists('data/network_logs.csv'):
        network_logs.to_csv('data/network_logs.csv', index=False)
        print("Generated synthetic network data and saved to data/network_logs.csv")

    # Visualize network conditions
    visualizer = Visualizer(config)
    visualizer.plot_network_conditions(network_logs, 'data/network_conditions.png')
    print("Network conditions visualization saved to data/network_conditions.png")

    # Prepare data for training
    print("Preparing training data...")
    X_train, X_test, y_train, y_test = data_loader.prepare_training_data(network_logs)

    # Train bitrate prediction model
    print("Training bitrate prediction model...")
    bitrate_predictor = BitratePredictor(config)
    history = bitrate_predictor.train(X_train, y_train, X_test, y_test)

    # Save the model
    bitrate_predictor.save('models/saved/bitrate_predictor.h5')
    print("Bitrate prediction model saved to models/saved/bitrate_predictor.h5")

    # Visualize learning curves
    visualizer.plot_learning_curves(history, 'data/learning_curves.png')
    print("Learning curves saved to data/learning_curves.png")

    # Train QoE estimation model
    print("Training QoE estimation model...")

    # Prepare data for QoE model
    # We need to create features: bitrate, bitrate changes, rebuffering, latency, packet loss
    qoe_features = []
    qoe_targets = []

    for i in range(1, len(network_logs)):
        bitrate = network_logs['optimal_bitrate'].iloc[i]
        bitrate_change = abs(network_logs['optimal_bitrate'].iloc[i] - network_logs['optimal_bitrate'].iloc[i - 1])
        rebuffering = max(0, 1 - network_logs['bandwidth'].iloc[i] / bitrate)  # Simplified rebuffering estimation
        latency = network_logs['latency'].iloc[i]
        loss_rate = network_logs['loss_rate'].iloc[i]

        # Feature vector
        qoe_features.append([bitrate, bitrate_change, rebuffering, latency, loss_rate])

        # Calculate simplified QoE score as target
        bitrate_utility = bitrate / max(config['video']['available_bitrates'])
        rebuffering_penalty = config['qoe']['rebuffering_penalty'] * rebuffering
        bitrate_switch_penalty = config['qoe']['bitrate_switch_penalty'] * bitrate_change

        qoe = config['qoe']['bitrate_weight'] * bitrate_utility - rebuffering_penalty - bitrate_switch_penalty
        qoe_targets.append(max(0, qoe))

    # Convert to numpy arrays
    qoe_features = np.array(qoe_features)
    qoe_targets = np.array(qoe_targets)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        qoe_features, qoe_targets, test_size=0.2, random_state=42
    )

    # Train QoE model
    qoe_estimator = QoEEstimator(config)
    qoe_history = qoe_estimator.train(X_train, y_train, X_test, y_test)

    # Save the model
    qoe_estimator.save('models/saved/qoe_estimator.h5')
    print("QoE estimation model saved to models/saved/qoe_estimator.h5")

    print("Training complete!")


if __name__ == "__main__":
    main()
