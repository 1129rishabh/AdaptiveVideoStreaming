import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
from matplotlib.animation import FuncAnimation


class Visualizer:
    def __init__(self, config):
        self.config = config

    def plot_network_conditions(self, df, output_path=None):
        """
        Plot network conditions over time

        Args:
            df: pandas DataFrame with network metrics
            output_path: path to save the plot, if None, display the plot
        """
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Plot bandwidth
        axs[0].plot(df['timestamp'], df['bandwidth'], 'b-')
        axs[0].set_ylabel('Bandwidth (kbps)')
        axs[0].set_title('Network Bandwidth Over Time')
        axs[0].grid(True)

        # Plot latency
        axs[1].plot(df['timestamp'], df['latency'], 'r-')
        axs[1].set_ylabel('Latency (ms)')
        axs[1].set_title('Network Latency Over Time')
        axs[1].grid(True)

        # Plot packet loss
        axs[2].plot(df['timestamp'], df['loss_rate'] * 100, 'g-')
        axs[2].set_ylabel('Packet Loss (%)')
        axs[2].set_xlabel('Time')
        axs[2].set_title('Packet Loss Rate Over Time')
        axs[2].grid(True)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()

    def plot_qoe_comparison(self, results, output_path=None):
        """
        Plot QoE comparison between different algorithms

        Args:
            results: dict mapping algorithm name to QoE metrics
            output_path: path to save the plot, if None, display the plot
        """
        algorithms = list(results.keys())
        qoe_scores = [results[alg]['qoe_score'] for alg in algorithms]
        rebuffering = [results[alg]['total_rebuffering'] for alg in algorithms]
        avg_bitrate = [results[alg]['avg_bitrate'] for alg in algorithms]

        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        # Plot QoE scores
        axs[0].bar(algorithms, qoe_scores, color='blue')
        axs[0].set_ylabel('QoE Score')
        axs[0].set_title('Quality of Experience Comparison')

        # Plot rebuffering time
        axs[1].bar(algorithms, rebuffering, color='red')
        axs[1].set_ylabel('Total Rebuffering (s)')
        axs[1].set_title('Rebuffering Time Comparison')

        # Plot average bitrate
        axs[2].bar(algorithms, avg_bitrate, color='green')
        axs[2].set_ylabel('Average Bitrate (p)')
        axs[2].set_title('Average Video Quality Comparison')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()

    def create_streaming_animation(self, metrics_history, output_path=None):
        """
        Create an animation of streaming metrics over time

        Args:
            metrics_history: list of dicts with metrics at each time step
            output_path: path to save the animation, if None, display it
        """
        # Convert metrics history to DataFrame
        df = pd.DataFrame(metrics_history)

        # Create figure and subplots
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

        # Initialize lines
        bandwidth_line, = axs[0].plot([], [], 'b-', label='Bandwidth')
        bitrate_line, = axs[0].plot([], [], 'r-', label='Selected Bitrate')
        buffer_line, = axs[1].plot([], [], 'g-')
        latency_line, = axs[2].plot([], [], 'm-')
        qoe_line, = axs[3].plot([], [], 'c-')

        # Set up axes
        axs[0].set_ylabel('Bandwidth / Bitrate (kbps)')
        axs[0].set_title('Bandwidth and Selected Bitrate')
        axs[0].grid(True)
        axs[0].legend()

        axs[1].set_ylabel('Buffer Level (s)')
        axs[1].set_title('Playback Buffer Level')
        axs[1].grid(True)

        axs[2].set_ylabel('Latency (ms)')
        axs[2].set_title('Network Latency')
        axs[2].grid(True)

        axs[3].set_ylabel('QoE Score')
        axs[3].set_xlabel('Time (s)')
        axs[3].set_title('Quality of Experience')
        axs[3].grid(True)

        # Set time limits
        max_time = len(df)
        axs[3].set_xlim(0, max_time)

        # Set y-axis limits
        axs[0].set_ylim(0, max(df['bandwidth'].max() * 1.1, df['bitrate'].max() * 1.1))
        axs[1].set_ylim(0, self.config['network']['buffer_size'] * 1.1)
        axs[2].set_ylim(0, df['latency'].max() * 1.1)
        axs[3].set_ylim(0, 5)  # QoE score typically 0-5

        def init():
            bandwidth_line.set_data([], [])
            bitrate_line.set_data([], [])
            buffer_line.set_data([], [])
            latency_line.set_data([], [])
            qoe_line.set_data([], [])
            return bandwidth_line, bitrate_line, buffer_line, latency_line, qoe_line

        def update(frame):
            # Get data up to current frame
            current_df = df.iloc[:frame]
            times = np.arange(len(current_df))

            bandwidth_line.set_data(times, current_df['bandwidth'])
            bitrate_line.set_data(times, current_df['bitrate'])
            buffer_line.set_data(times, current_df['buffer_level'])
            latency_line.set_data(times, current_df['latency'])
            qoe_line.set_data(times, current_df['qoe_score'])

            return bandwidth_line, bitrate_line, buffer_line, latency_line, qoe_line

        ani = FuncAnimation(fig, update, frames=len(df), init_func=init, blit=True, interval=50)

        plt.tight_layout()

        if output_path:
            ani.save(output_path, writer='ffmpeg')
        else:
            plt.show()

    def plot_bitrate_distribution(self, bitrate_history, output_path=None):
        """
        Plot distribution of selected bitrates

        Args:
            bitrate_history: list of selected bitrates
            output_path: path to save the plot, if None, display the plot
        """
        plt.figure(figsize=(10, 6))

        # Count occurrences of each bitrate
        bitrates = self.config['video']['available_bitrates']
        counts = [bitrate_history.count(br) for br in bitrates]

        # Calculate percentages
        total = sum(counts)
        percentages = [count / total * 100 for count in counts]

        # Create bar chart
        plt.bar(bitrates, percentages)
        plt.xlabel('Bitrate (p)')
        plt.ylabel('Percentage (%)')
        plt.title('Distribution of Selected Bitrates')
        plt.grid(axis='y')

        # Add percentage labels on top of bars
        for i, percentage in enumerate(percentages):
            plt.text(bitrates[i], percentage + 1, f'{percentage:.1f}%', ha='center')

        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()

    def plot_learning_curves(self, history, output_path=None):
        """
        Plot learning curves from model training

        Args:
            history: training history object from Keras
            output_path: path to save the plot, if None, display the plot
        """
        plt.figure(figsize=(12, 5))

        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        plt.grid(True)

        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.grid(True)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()
