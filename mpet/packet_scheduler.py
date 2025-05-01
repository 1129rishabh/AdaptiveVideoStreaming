import heapq
from collections import deque


class PacketScheduler:
    def __init__(self, path_selector, config):
        self.path_selector = path_selector
        self.config = config
        self.algorithm = config['mpet']['packet_scheduling_algorithm']
        self.packet_queue = []  # Priority queue for EDF
        self.path_queues = {}  # Per-path queues for WFQ
        self.sequence_number = 0
        self.path_credits = {}  # For weighted fair queuing

    def enqueue_packet(self, packet_data, deadline=None, priority=0):
        """
        Add a packet to the scheduling queue

        Args:
            packet_data: the actual packet data or reference
            deadline: deadline for packet delivery (for EDF)
            priority: priority level (higher is more important)

        Returns:
            sequence number assigned to the packet
        """
        self.sequence_number += 1

        if self.algorithm == 'earliest_deadline_first':
            # Use deadline for priority in EDF
            if deadline is None:
                deadline = float('inf')
            heapq.heappush(self.packet_queue, (deadline, priority, self.sequence_number, packet_data))

        else:  # weighted_fair_queuing or other
            # Just add to general queue, will be distributed to path queues during scheduling
            heapq.heappush(self.packet_queue, (priority, self.sequence_number, packet_data))

        return self.sequence_number

    def schedule_packets(self):
        """
        Schedule packets across available paths

        Returns:
            dict mapping path_id to list of packets for that path
        """
        # Get available paths
        paths = self.path_selector.select_paths()
        if not paths:
            return {}

        # Get path weights based on bandwidth
        path_weights = self.path_selector.multipath_manager.get_path_weights()

        # Initialize result and path queues if needed
        scheduled_packets = {path: [] for path in paths}

        if self.algorithm == 'earliest_deadline_first':
            return self._schedule_edf(paths, path_weights)
        elif self.algorithm == 'weighted_fair_queuing':
            return self._schedule_wfq(paths, path_weights)
        else:
            # Default round-robin scheduling
            return self._schedule_round_robin(paths)

    def _schedule_edf(self, paths, path_weights):
        """Earliest Deadline First scheduling algorithm"""
        scheduled_packets = {path: [] for path in paths}

        # Sort paths by quality (higher weight = better path)
        sorted_paths = sorted(paths, key=lambda p: path_weights.get(p, 0), reverse=True)

        # Assign packets to paths, starting with the most critical deadlines
        temp_queue = []
        while self.packet_queue:
            deadline, priority, seq_num, packet = heapq.heappop(self.packet_queue)

            # Find the best path that has capacity
            assigned = False
            for path in sorted_paths:
                # Simple capacity check - in reality would be more complex
                if len(scheduled_packets[path]) < 10:  # Arbitrary limit for simulation
                    scheduled_packets[path].append((seq_num, packet))
                    assigned = True
                    break

            # If no path had capacity, put back in queue for next round
            if not assigned:
                temp_queue.append((deadline, priority, seq_num, packet))

        # Restore unscheduled packets
        for packet_info in temp_queue:
            heapq.heappush(self.packet_queue, packet_info)

        return scheduled_packets

    def _schedule_wfq(self, paths, path_weights):
        """Weighted Fair Queuing scheduling algorithm"""
        scheduled_packets = {path: [] for path in paths}

        # Initialize path credits if needed
        for path in paths:
            if path not in self.path_credits:
                self.path_credits[path] = 0

        # Initialize path queues if needed
        for path in paths:
            if path not in self.path_queues:
                self.path_queues[path] = deque()

        # First, distribute packets to path queues based on weights
        temp_queue = []
        while self.packet_queue:
            priority, seq_num, packet = heapq.heappop(self.packet_queue)

            # Find path with the highest credit
            best_path = max(paths, key=lambda p: self.path_credits[p])
            self.path_queues[best_path].append((seq_num, packet))

            # Update credits - selected path loses credit, others gain
            for path in paths:
                if path == best_path:
                    self.path_credits[path] -= 1.0
                else:
                    self.path_credits[path] += path_weights.get(path, 1.0 / len(paths))

        # Now take packets from path queues based on capacity
        for path in paths:
            # Simple capacity check - in reality would be more complex
            capacity = int(10 * path_weights.get(path, 1.0 / len(paths)))  # Arbitrary limit for simulation

            # Take up to 'capacity' packets from this path's queue
            for _ in range(min(capacity, len(self.path_queues[path]))):
                scheduled_packets[path].append(self.path_queues[path].popleft())

        return scheduled_packets

    def _schedule_round_robin(self, paths):
        """Simple round-robin scheduling algorithm"""
        scheduled_packets = {path: [] for path in paths}

        # Distribute packets in round-robin fashion
        path_index = 0
        temp_queue = []

        while self.packet_queue:
            priority, seq_num, packet = heapq.heappop(self.packet_queue)

            # Get current path
            path = paths[path_index]

            # Simple capacity check
            if len(scheduled_packets[path]) < 10:  # Arbitrary limit for simulation
                scheduled_packets[path].append((seq_num, packet))
            else:
                temp_queue.append((priority, seq_num, packet))

            # Move to next path
            path_index = (path_index + 1) % len(paths)

        # Restore unscheduled packets
        for packet_info in temp_queue:
            heapq.heappush(self.packet_queue, packet_info)

        return scheduled_packets
