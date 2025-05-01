import os
import yaml
import time
import numpy as np
# import pandas as pd
import argparse
# from threading import Thread
# import matplotlib.pyplot as plt

# Import project modules
from models.bitrate_predictor import BitratePredictor
from models.qoe_estimator import QoEEstimator
from models.network_analyzer import NetworkAnalyzer
from mpet.multipath_manager import MultipathManager
from mpet.path_selector import PathSelector
from mpet.packet_scheduler import PacketScheduler
from streaming.video_segmenter import VideoSegmenter
from streaming.buffer_manager import BufferManager
from streaming.bitrate_adapter import BitrateAdapter
from utils.metrics import QoEMetrics
from utils.data_loader import DataLoader
from utils.visualization import Visualizer


class AdaptiveStreamingSystem:
    def __init__(self, config, use_ml=True, use_mpet=True):
        self.config = config
        self.use_ml = use_ml
        self.use_mpet = use_mpet

        # Initialize components
        self.network_analyzer = NetworkAnalyzer(config)
        self.buffer_manager = BufferManager(config)

        # Initialize ML models if enabled
        if use_ml:
            self.bitrate_predictor = BitratePredictor(config)
            try:
                self.bitrate_predictor.load('models/saved/bitrate_predictor.h5')
                print("Loaded bitrate prediction model")
            except:
                print("Could not load bitrate prediction model, using heuristic approach")
                self.use_ml = False

        # Initialize MPET components if enabled
        if use_mpet:
            self.multipath_manager = MultipathManager(config)
            self.path_selector = PathSelector(self.multipath_manager, config)
            self.packet_scheduler = PacketScheduler(self.path_selector, config)

        # Initialize bitrate adapter
        self.bitrate_adapter = BitrateAdapter(
            config,
            bitrate_predictor=self.bitrate_predictor if use_ml else None
        )

        # Initialize metrics collector
        self.metrics = QoEMetrics(config)

        # Simulation state
        self.current_segment = 0
        self.total_segments = 100  # Default, can be changed
        self.is_running = False
        self.network_conditions = []
        self.segment_download_times = []

    def simulate_network_conditions(self):
        """Simulate changing network conditions"""
        # Load synthetic data or create it
        data_loader = DataLoader(self.config)
        network_logs = data_loader.load_network_logs('data/network_logs.csv')

        # Extract network conditions
        self.network_conditions = []
        for _, row in network_logs.iterrows():
            self.network_conditions.append({
                'bandwidth': row['bandwidth'],
                'latency': row['latency'],
                'loss_rate': row['loss_rate']
            })

        # Limit to the number of segments we're simulating
        self.network_conditions = self.network_conditions[:self.total_segments]

    def simulate_segment_download(self, segment_idx, bitrate):
        """
        Simulate downloading a video segment

        Args:
            segment_idx: index of the segment
            bitrate: bitrate of the segment in p (resolution)

        Returns:
            tuple: (download_time, success)
        """
        # Get current network conditions
        network_idx = min(segment_idx, len(self.network_conditions) - 1)
        network = self.network_conditions[network_idx]

        # Calculate segment size (in bits) based on bitrate
        # Assuming 1 second of video at resolution p needs approximately p*1000 bits
        segment_size = bitrate * 1000 * self.config['video']['segment_duration']

        # Calculate download time based on bandwidth
        if self.use_mpet:
            # With MPET, use combined bandwidth from multiple paths
            total_bandwidth = self.multipath_manager.get_total_bandwidth()
            if total_bandwidth == 0:  # No paths available
                return float('inf'), False
        else:
            # Without MPET, use single path bandwidth
            total_bandwidth = network['bandwidth']

        # Calculate base download time
        download_time = segment_size / (total_bandwidth * 1000)  # Convert kbps to bps

        # Add latency effect
        download_time += network['latency'] / 1000  # Convert ms to seconds

        # Add packet loss effect (retransmissions)
        download_time *= (1 + 5 * network['loss_rate'])  # Simple model: each 1% loss adds 5% time

        # Add some randomness
        download_time *= np.random.uniform(0.9, 1.1)

        # Check if download was successful (timeout if too long)
        success = download_time < 30  # Timeout after 30 seconds

        return download_time, success

    def run_simulation(self):
        """Run the adaptive streaming simulation"""
        self.is_running = True

        # Simulate network conditions
        self.simulate_network_conditions()

        print(f"Starting simulation with {self.total_segments} segments")
        print(f"Using ML: {self.use_ml}, Using MPET: {self.use_mpet}")

        # Main streaming loop
        while self.current_segment < self.total_segments and self.is_running:
            # Update playback and check for rebuffering
            is_rebuffering, rebuffer_time = self.buffer_manager.update_playback()

            # Get current network conditions
            network_idx = min(self.current_segment, len(self.network_conditions) - 1)
            network = self.network_conditions[network_idx]

            # Update network analyzer with current conditions
            self.network_analyzer.update({
                'bandwidth': network['bandwidth'],
                'latency': network['latency'],
                'loss_rate': network['loss_rate'],
                'buffer_level': self.buffer_manager.get_buffer_level()
            })

            # Record metrics
            self.metrics.record_network_metrics(
                network['bandwidth'],
                network['latency'],
                network['loss_rate']
            )
            self.metrics.record_buffer_level(self.buffer_manager.get_buffer_level())

            # If using MPET, update path statistics and select paths
            if self.use_mpet:
                for path_id in self.multipath_manager.active_paths:
                    # Simulate different conditions for each path
                    path_bandwidth = network['bandwidth'] * np.random.uniform(0.5, 1.5)
                    path_latency = network['latency'] * np.random.uniform(0.8, 1.2)
                    path_loss = network['loss_rate'] * np.random.uniform(0.7, 1.3)

                    self.multipath_manager.update_path_stats(path_id, {
                        'bandwidth': path_bandwidth,
                        'latency': path_latency,
                        'loss_rate': path_loss
                    })

                # Select paths for transmission
                selected_paths = self.path_selector.select_paths()
                if not selected_paths:
                    print(f"No paths available at segment {self.current_segment}")
                    time.sleep(0.1)  # Small delay before retrying
                    continue

            # Get next segment to request
            next_segment = self.buffer_manager.get_next_segment_to_request()
            if next_segment is None:
                # All segments are buffered
                time.sleep(0.1)  # Small delay
                continue

            # Select bitrate for next segment
            if self.use_ml:
                # Get features for prediction
                features = self.network_analyzer.get_features()
                selected_bitrate = self.bitrate_adapter.select_bitrate(
                    features,
                    self.buffer_manager.get_buffer_level()
                )
            else:
                # Use heuristic approach
                selected_bitrate = self.bitrate_adapter.select_bitrate(
                    network,
                    self.buffer_manager.get_buffer_level()
                )

            print(
                f"Segment {next_segment}: Selected bitrate {selected_bitrate}p, Buffer: {self.buffer_manager.get_buffer_level():.1f}s")

            # Simulate segment download
            start_time = time.time()
            download_time, success = self.simulate_segment_download(next_segment, selected_bitrate)

            if success:
                # Add segment to buffer
                self.buffer_manager.add_segment(next_segment, selected_bitrate)

                # Record metrics for this segment
                self.metrics.record_segment_metrics(
                    next_segment,
                    selected_bitrate,
                    download_time,
                    rebuffer_time
                )

                self.current_segment += 1
            else:
                print(f"Segment {next_segment} download failed, retrying with lower bitrate")
                # Try with a lower bitrate next time
                self.bitrate_adapter.switch_penalty *= 2  # Make it more conservative

            # Small delay to simulate real-time
            time.sleep(0.01)

        self.is_running = False
        print("Simulation complete!")

        # Calculate final QoE
        qoe = self.metrics.calculate_qoe()
        print(f"Final QoE score: {qoe:.2f}")

        # Return summary metrics
        return self.metrics.get_summary()


def main():
    parser = argparse.ArgumentParser(description='Adaptive Video Streaming Simulation')
    parser.add_argument('--no-ml', action='store_true', help='Disable machine learning')
    parser.add_argument('--no-mpet', action='store_true', help='Disable multipath transport')
    parser.add_argument('--segments', type=int, default=100, help='Number of segments to simulate')
    args = parser.parse_args()

    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create output directories
    os.makedirs('results', exist_ok=True)

    # Run different configurations for comparison
    results = {}

    # Configuration 1: ML + MPET (full system)
    if not args.no_ml and not args.no_mpet:
        print("\n=== Running simulation with ML + MPET ===")
        system = AdaptiveStreamingSystem(config, use_ml=True, use_mpet=True)
        system.total_segments = args.segments
        results['ML+MPET'] = system.run_simulation()

    # Configuration 2: ML only
    if not args.no_ml:
        print("\n=== Running simulation with ML only ===")
        system = AdaptiveStreamingSystem(config, use_ml=True, use_mpet=False)
        system.total_segments = args.segments
        results['ML'] = system.run_simulation()

    # Configuration 3: MPET only
    if not args.no_mpet:
        print("\n=== Running simulation with MPET only ===")
        system = AdaptiveStreamingSystem(config, use_ml=False, use_mpet=True)
        system.total_segments = args.segments
        results['MPET'] = system.run_simulation()

    # Configuration 4: Baseline (no ML, no MPET)
    print("\n=== Running baseline simulation (no ML, no MPET) ===")
    system = AdaptiveStreamingSystem(config, use_ml=False, use_mpet=False)
    system.total_segments = args.segments
    results['Baseline'] = system.run_simulation()

    # Visualize comparison results
    visualizer = Visualizer(config)
    visualizer.plot_qoe_comparison(results, 'results/qoe_comparison.png')
    print("QoE comparison saved to results/qoe_comparison.png")

    # Save detailed results
    with open('results/simulation_results.txt', 'w') as f:
        for name, metrics in results.items():
            f.write(f"=== {name} ===\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
            f.write("\n")

    print("Detailed results saved to results/simulation_results.txt")


if __name__ == "__main__":
    main()
