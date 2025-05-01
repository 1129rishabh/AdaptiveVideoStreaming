import time
import numpy as np
from collections import deque


class BufferManager:
    def __init__(self, config):
        self.config = config
        self.buffer_size_seconds = config['network']['buffer_size']
        self.initial_buffer_seconds = config['network']['initial_buffer']
        self.buffer = deque()  # Will store (segment_idx, bitrate, timestamp) tuples
        self.playback_position = 0  # Current playback position in seconds
        self.last_playback_update = time.time()
        self.is_buffering = True
        self.segment_duration = config['video']['segment_duration']
        self.buffer_history = []  # For tracking buffer level over time

    def add_segment(self, segment_idx, bitrate):
        """
        Add a segment to the buffer

        Args:
            segment_idx: index of the segment
            bitrate: bitrate of the segment

        Returns:
            bool: True if segment was added, False if buffer is full
        """
        # Check if buffer is full
        if self.get_buffer_level() >= self.buffer_size_seconds:
            return False

        # Check if segment is already in buffer
        for idx, br, _ in self.buffer:
            if idx == segment_idx:
                return False

        # Add segment to buffer
        self.buffer.append((segment_idx, bitrate, time.time()))

        # Update buffer history
        self.buffer_history.append((time.time(), self.get_buffer_level()))

        # Check if we can start playback
        if self.is_buffering and self.get_buffer_level() >= self.initial_buffer_seconds:
            self.is_buffering = False
            self.last_playback_update = time.time()

        return True

    def get_buffer_level(self):
        """
        Get current buffer level in seconds

        Returns:
            buffer level in seconds
        """
        return len(self.buffer) * self.segment_duration

    def update_playback(self):
        """
        Update playback position based on elapsed time

        Returns:
            tuple: (is_rebuffering, rebuffer_time)
        """
        current_time = time.time()
        elapsed = current_time - self.last_playback_update
        rebuffer_time = 0

        # If we're buffering, don't update playback position
        if self.is_buffering:
            if self.get_buffer_level() >= self.initial_buffer_seconds:
                self.is_buffering = False
                self.last_playback_update = current_time
            else:
                return True, elapsed

        # Update playback position
        self.playback_position += elapsed
        self.last_playback_update = current_time

        # Calculate how many segments to consume (convert to integer)
        prev_position = self.playback_position - elapsed
        prev_segments = int(prev_position / self.segment_duration)
        current_segments = int(self.playback_position / self.segment_duration)
        segments_to_consume = current_segments - prev_segments

        # Consume segments that have been played
        for _ in range(min(segments_to_consume, len(self.buffer))):
            if self.buffer:
                self.buffer.popleft()

        # Check if buffer is empty - need to rebuffer
        if not self.buffer:
            self.is_buffering = True
            rebuffer_time = elapsed

        # Update buffer history
        self.buffer_history.append((current_time, self.get_buffer_level()))

        return self.is_buffering, rebuffer_time

    def get_next_segment_to_request(self):
        """
        Get the index of the next segment that should be requested

        Returns:
            next segment index or None if all segments are buffered
        """
        if not self.buffer:
            # If buffer is empty, start from the current playback position
            next_segment = int(self.playback_position / self.segment_duration)
            return max(0, next_segment)

        # Get the highest segment index in the buffer
        highest_idx = max(idx for idx, _, _ in self.buffer)

        # Next segment is the one after the highest
        return highest_idx + 1

    def get_recent_bitrates(self, count=3):
        """
        Get the bitrates of the most recently added segments

        Args:
            count: number of recent segments to consider

        Returns:
            list of recent bitrates
        """
        recent_segments = list(self.buffer)[-count:]
        return [br for _, br, _ in recent_segments]

    def get_buffer_stability(self):
        """
        Calculate buffer stability metric (0-1, higher is more stable)

        Returns:
            stability score
        """
        if len(self.buffer_history) < 10:
            return 0.5  # Default medium stability if not enough history

        # Get recent buffer levels
        recent_levels = [level for _, level in self.buffer_history[-10:]]

        # Calculate coefficient of variation
        if np.mean(recent_levels) > 0:
            cv = np.std(recent_levels) / np.mean(recent_levels)

            # Convert to stability score (1 - normalized cv)
            stability = max(0, min(1, 1 - cv / 0.5))  # Assuming cv > 0.5 is very unstable
            return stability
        else:
            return 0  # Empty buffer is unstable
