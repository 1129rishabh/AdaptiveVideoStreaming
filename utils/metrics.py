import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


class QoEMetrics:
    def __init__(self, config):
        self.config = config
        self.metrics = defaultdict(list)
        self.segment_metrics = []

    def record_segment_metrics(self, segment_idx, bitrate, download_time, rebuffering_time):
        """
        Record metrics for a downloaded segment

        Args:
            segment_idx: index of the segment
            bitrate: bitrate of the segment
            download_time: time taken to download the segment
            rebuffering_time: rebuffering time experienced
        """
        self.segment_metrics.append({
            'segment_idx': segment_idx,
            'bitrate': bitrate,
            'download_time': download_time,
            'rebuffering_time': rebuffering_time,
            'timestamp': pd.Timestamp.now()
        })

        # Update running metrics
        self.metrics['bitrate'].append(bitrate)
        self.metrics['rebuffering_time'].append(rebuffering_time)

        # Calculate bitrate switch if possible
        if len(self.metrics['bitrate']) > 1:
            switch = abs(self.metrics['bitrate'][-1] - self.metrics['bitrate'][-2])
            self.metrics['bitrate_switches'].append(switch)
        else:
            self.metrics['bitrate_switches'].append(0)

    def record_network_metrics(self, bandwidth, latency, loss_rate):
        """
        Record network metrics

        Args:
            bandwidth: current bandwidth in kbps
            latency: current latency in ms
            loss_rate: current packet loss rate
        """
        self.metrics['bandwidth'].append(bandwidth)
        self.metrics['latency'].append(latency)
        self.metrics['loss_rate'].append(loss_rate)

    def record_buffer_level(self, buffer_level):
        """
        Record buffer level

        Args:
            buffer_level: current buffer level in seconds
        """
        self.metrics['buffer_level'].append(buffer_level)

    def calculate_qoe(self):
        """
        Calculate overall QoE based on recorded metrics

        Returns:
            QoE score
        """
        if not self.segment_metrics:
            return 0

        # Get config parameters
        rebuffering_penalty = self.config['qoe']['rebuffering_penalty']
        bitrate_weight = self.config['qoe']['bitrate_weight']
        bitrate_switch_penalty = self.config['qoe']['bitrate_switch_penalty']

        # Calculate metrics
        avg_bitrate = np.mean(self.metrics['bitrate'])
        total_rebuffering = sum(self.metrics['rebuffering_time'])
        avg_bitrate_switch = np.mean(self.metrics['bitrate_switches']) if self.metrics['bitrate_switches'] else 0

        # Normalize bitrate by max available bitrate
        max_bitrate = max(self.config['video']['available_bitrates'])
        normalized_bitrate = avg_bitrate / max_bitrate

        # Calculate QoE
        qoe = (bitrate_weight * normalized_bitrate) - (rebuffering_penalty * total_rebuffering) - (
                    bitrate_switch_penalty * avg_bitrate_switch)

        return max(0, qoe)  # QoE should not be negative

    def get_summary(self):
        """
        Get summary of all metrics

        Returns:
            dict with summary statistics
        """
        summary = {}

        for metric, values in self.metrics.items():
            if values:
                summary[f'avg_{metric}'] = np.mean(values)
                summary[f'min_{metric}'] = np.min(values)
                summary[f'max_{metric}'] = np.max(values)
                summary[f'std_{metric}'] = np.std(values)

        summary['total_rebuffering'] = sum(self.metrics['rebuffering_time'])
        summary['qoe_score'] = self.calculate_qoe()

        return summary

    def plot_metrics(self, output_path=None):
        """
        Plot recorded metrics

        Args:
            output_path: path to save the plot, if None, display the plot
        """
        if not self.segment_metrics:
            print("No metrics to plot")
            return

        # Convert segment metrics to DataFrame
        df = pd.DataFrame(self.segment_metrics)

        # Create figure with subplots
        fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

        # Plot bitrate
        axs[0].plot(df['timestamp'], df['bitrate'], 'b-')
        axs[0].set_ylabel('Bitrate (p)')
        axs[0].set_title('Video Bitrate Over Time')
        axs[0].grid(True)

        # Plot rebuffering time
        axs[1].bar(df['timestamp'], df['rebuffering_time'], color='r', width=0.01)
        axs[1].set_ylabel('Rebuffering Time (s)')
        axs[1].set_title('Rebuffering Events')
        axs[1].grid(True)

        # Plot download time
        axs[2].plot(df['timestamp'], df['download_time'], 'g-')
        axs[2].set_ylabel('Download Time (s)')
        axs[2].set_title('Segment Download Time')
        axs[2].grid(True)

        # Plot buffer level if available
        if 'buffer_level' in self.metrics and len(self.metrics['buffer_level']) > 0:
            buffer_times = pd.date_range(
                start=df['timestamp'].min(),
                end=df['timestamp'].max(),
                periods=len(self.metrics['buffer_level'])
            )
            axs[3].plot(buffer_times, self.metrics['buffer_level'], 'm-')
            axs[3].set_ylabel('Buffer Level (s)')
            axs[3].set_title('Buffer Level Over Time')
            axs[3].grid(True)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()
